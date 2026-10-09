"""Algorithm 1 of the paper: brewing a poisoned subset by gradient matching.

The loop below is a direct transcription of the published pseudocode (paper p. 5):

    Require pretrained clean network F(., theta), training set, target (x_t, y_adv),
            poison budget P < N, perturbation bound eps, restarts R, optimization steps M
    1  select P training images with label y_adv
    2  for r = 1..R:
    3      randomly initialize perturbations Delta^r in C
    4      for j = 1..M:
    5          apply data augmentation to all poisoned samples
    6          compute the average cost B(Delta^r, theta) over all poisoned samples
    7          update Delta^r with a step of signed Adam and project onto ||Delta^r||_inf <= eps
    8  choose Delta* as the Delta^r with minimal B(Delta^r, theta)
    9  return the poisoned dataset

with lines 6-7 run in the stochastic mini-batch setting noted at the end of section 3.4.

Defaults tau=0.1, R=8, M=250 are the values section 5 states for every experiment in the paper.
"""

from __future__ import annotations

from typing import Callable

import torch

from ..data.augment import RandomTransform
from ..data.datasets import denormalization_tensors
from .objectives import (
    bullseye_loss,
    gradient_matching_loss,
    poison_frogs_loss,
    target_gradient,
)


class PoisonBounds:
    """Converts the l_inf budget into the normalized space the models actually see.

    `eps` is given in 0..255 pixel units (section 3.1), while inputs are normalized per channel, so
    the bound becomes `eps/255/std` per channel. The second constraint -- that `x + Delta` stay a
    valid image -- is not written in Algorithm 1 but is required for the perturbation to survive
    being written out as an 8-bit image; it comes from the reference implementation and is flagged in
    DESIGN_DECISIONS.md section 3.
    """

    def __init__(self, eps: float, dataset: str, device: torch.device) -> None:
        dm, ds = denormalization_tensors(dataset, device)
        self.dm, self.ds = dm, ds
        self.eps_normalized = eps / 255.0 / ds  # [1, C, 1, 1]

    def clip_eps(self, delta: torch.Tensor) -> torch.Tensor:
        """The eps-ball alone, used at initialization before any image is known."""
        return torch.max(torch.min(delta, self.eps_normalized), -self.eps_normalized)

    def project(self, delta: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Project onto the intersection of the eps-ball and the valid pixel range.

        The two constraints are intersected rather than applied in sequence, so the eps bound of
        section 3.1 holds unconditionally -- clipping to the pixel range afterwards could otherwise
        push a coordinate back outside the ball. For a genuine image both intervals contain 0, so the
        intersection is non-empty; the final `maximum` is only a guard for degenerate inputs.
        """
        eps = self.eps_normalized.expand_as(clean)
        # Both endpoints are clamped into the eps-ball first, so neither can ever leave it.
        low = torch.clamp(-self.dm / self.ds - clean, min=-eps, max=eps)
        high = torch.clamp((1 - self.dm) / self.ds - clean, min=-eps, max=eps)
        high = torch.maximum(high, low)
        return torch.max(torch.min(delta, high), low)


def initialize_delta(
    shape: tuple[int, ...],
    bounds: PoisonBounds,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Line 3 of Algorithm 1: "randomly initialize perturbations Delta^r in C".

    The paper does not fix the distribution; `randn` scaled into the bound is the reference default
    (DESIGN_DECISIONS.md section 3).
    """
    if mode == "zero":
        delta = torch.zeros(shape, device=device)
    elif mode == "rand":
        delta = (torch.rand(shape, device=device) * 2 - 1) * bounds.eps_normalized
    elif mode == "randn":
        delta = torch.randn(shape, device=device) * bounds.eps_normalized
    else:
        raise ValueError(f"unknown initialization: {mode}")
    return bounds.clip_eps(delta)


def _scheduler_milestones(steps: int) -> list[int]:
    """x0.1 decay points. The paper says "decaying step size"; the ratios are the reference's."""
    return [int(steps // 2.667), int(steps // 1.6), int(steps // 1.142)]


class Brewer:
    """Runs Algorithm 1 for one case.

    Args:
        models: the clean networks providing `theta`. One model is the paper's default; a list of
            `K` models is the ensemble of section 3.4.
        clean_poisons: the unperturbed poison images, normalized, shape [P, C, H, W].
        poison_labels: their (unchanged) labels -- the attack is clean-label.
        target_inputs / target_labels_adv: the target images and the adversarial labels `y_adv`.
        augment: differentiable augmentation module, or None to disable (Table 1 ablation).
        gradient_hook: optional transform applied to the poison gradient before the objective,
            used by the attacker-aware differential-privacy experiment (appendix F.3).
    """

    def __init__(
        self,
        cfg,
        models: list[torch.nn.Module],
        clean_poisons: torch.Tensor,
        poison_labels: torch.Tensor,
        target_inputs: torch.Tensor,
        target_labels_adv: torch.Tensor,
        device: torch.device,
        augment: RandomTransform | None = None,
        gradient_hook: Callable[[list[torch.Tensor]], list[torch.Tensor]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.models = models
        self.device = device
        self.clean = clean_poisons
        self.labels = poison_labels
        self.target_inputs = target_inputs
        self.target_labels = target_labels_adv
        self.augment = augment
        self.gradient_hook = gradient_hook
        self.bounds = PoisonBounds(cfg.eps, cfg.dataset, device)
        self.criterion = torch.nn.CrossEntropyLoss()
        # Projection needs the clean images on the compute device every step; cache one copy
        # instead of re-sending the whole stack each time.
        self._clean_device = clean_poisons.to(device)

        # Section 3.3 keeps theta fixed. eval() also stops batch-norm running statistics from being
        # updated by poison batches, which would silently change theta.
        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(True)

        if cfg.recipe == "gradient-matching":
            self.target_grads = []
            self.target_gnorms = []
            for model in self.models:
                grads, norm = target_gradient(
                    model, target_inputs, target_labels_adv, cfg.target_criterion
                )
                self.target_grads.append(grads)
                self.target_gnorms.append(norm)
        else:
            self.target_grads, self.target_gnorms = [], []

    # ------------------------------------------------------------------
    def _objective(self, model_index: int, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        recipe = self.cfg.recipe
        model = self.models[model_index]
        if recipe == "gradient-matching":
            return gradient_matching_loss(
                model,
                inputs,
                labels,
                self.target_grads[model_index],
                self.target_gnorms[model_index],
                objective=self.cfg.objective,
                cosine_mode=self.cfg.cosine_mode,
                train_criterion=self.criterion,
                gradient_hook=self.gradient_hook,
            )
        if recipe == "poison-frogs":
            return poison_frogs_loss(model, inputs, self.target_inputs)
        if recipe == "bullseye":
            return bullseye_loss(model, inputs, self.target_inputs)
        raise ValueError(f"unknown recipe: {recipe}")

    # ------------------------------------------------------------------
    def _batches(self) -> list[torch.Tensor]:
        """Index batches over the poisons (section 3.4: lines 6-7 run on mini-batches)."""
        count = self.clean.shape[0]
        order = torch.randperm(count) if self.cfg.poison_shuffle else torch.arange(count)
        size = min(self.cfg.poison_batch_size, count)
        return [order[i : i + size] for i in range(0, count, size)]

    def _step_objective(self, delta: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One pass over the poison mini-batches; returns the accumulated loss value."""
        total = 0.0
        delta.grad = None
        batches = self._batches()
        for index in batches:
            clean = self._clean_device[index]
            labels = self.labels[index].to(self.device, non_blocking=True)
            inputs = clean + delta[index]
            if self.augment is not None:
                # Alg. 1 line 5: augmentation is applied *after* Delta is added, and the update
                # differentiates through it (section 3.4).
                inputs = self.augment(inputs)
            losses = [self._objective(k, inputs, labels) for k in range(len(self.models))]
            loss = torch.stack(losses).mean()  # ensemble average (section 3.4)
            loss.backward()
            total += float(loss.detach())
        return delta, total / max(1, len(batches))

    def _measure(self, delta: torch.Tensor) -> float:
        """Evaluate `B(Delta, theta)` without updating anything.

        Algorithm 1 line 10 selects the restart with the minimal value of `B` *at the perturbation
        that is returned*. The value accumulated during the last optimization step belongs to the
        perturbation as it was before that step's update, so the comparison is made here instead,
        after the final projection. Gradients w.r.t. the model parameters are still needed -- that
        is what `B` is built from -- so this cannot run under `no_grad`; only the result is kept.
        """
        total = 0.0
        batches = self._batches()
        for index in batches:
            clean = self._clean_device[index]
            labels = self.labels[index].to(self.device, non_blocking=True)
            inputs = clean + delta[index]
            if self.augment is not None:
                inputs = self.augment(inputs)
            values = [self._objective(k, inputs, labels) for k in range(len(self.models))]
            total += float(torch.stack(values).mean().detach())
        return total / max(1, len(batches))

    def _run_trial(self) -> tuple[torch.Tensor, float]:
        cfg = self.cfg
        delta = initialize_delta(tuple(self.clean.shape), self.bounds, cfg.init, self.device)
        delta.requires_grad_(True)

        # Section 3.3 / Alg. 1 line 9: signed Adam. The reference scales tau by the poison batch
        # size and the ensemble size so that tau=0.1 keeps its meaning; flagged in DESIGN_DECISIONS.
        scale = (cfg.poison_batch_size / 512) / max(1, len(self.models))
        lr = cfg.tau * max(scale, 1e-3)
        optimizer = torch.optim.Adam([delta], lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=_scheduler_milestones(cfg.attack_steps), gamma=0.1
        )

        steps = 1 if cfg.dryrun else cfg.attack_steps
        for _ in range(steps):
            delta, _ = self._step_objective(delta)
            if delta.grad is None:
                break
            if cfg.attack_optimizer == "signAdam":
                delta.grad = delta.grad.sign()
            optimizer.step()
            if cfg.attack_scheduling:
                scheduler.step()
            with torch.no_grad():
                delta.data = self.bounds.project(delta.data, self._clean_device)
        final = delta.detach()
        return final, self._measure(final)

    def brew(self) -> tuple[torch.Tensor, float, list[float]]:
        """Lines 2-8: run `R` restarts and keep the perturbation with the lowest final `B`."""
        best_delta, best_value = None, float("inf")
        per_restart = []
        restarts = 1 if self.cfg.dryrun else self.cfg.restarts
        for _ in range(restarts):
            delta, value = self._run_trial()
            per_restart.append(value)
            if value < best_value:
                best_delta, best_value = delta, value
        assert best_delta is not None
        return best_delta, best_value, per_restart
