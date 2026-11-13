"""Victim and surrogate training.

Recipe, verbatim from appendix B.2: Nesterov SGD, batch size 128, momentum 0.9, weight decay 5e-4,
40 epochs, learning rate dropped by a factor of 10 at epochs 14, 24 and 35; initial learning rate
0.1 for residual architectures and 0.01 for the ConvNet, MobileNet-v2 and VGG-16. The dataset is
shuffled every epoch "so that where poisoned images appear in mini-batches is random and not known
to the attacker".

The same function trains the clean surrogate (`poison_delta=None`) and the victim, so the attacker's
`theta` is produced by exactly the recipe the victim uses -- the paper's gray-box assumption is about
knowing the architecture, not about using a different training procedure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data.augment import RandomTransform
from .data.datasets import denormalization_tensors, make_loader
from .defenses.purification import victim_purifier_normalized
from .utils import available_workers


@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    valid_acc: list[float] = field(default_factory=list)
    target_adv_acc: list[float] = field(default_factory=list)
    target_adv_loss: list[float] = field(default_factory=list)
    target_orig_acc: list[float] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "valid_acc": self.valid_acc,
            "target_adv_acc": self.target_adv_acc,
            "target_adv_loss": self.target_adv_loss,
            "target_orig_acc": self.target_orig_acc,
            "diagnostics": self.diagnostics,
        }


def build_optimizer(model: nn.Module, cfg) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma
    )
    return optimizer, scheduler


def apply_privacy_defense(model: nn.Module, cfg) -> None:
    """Differentially private SGD defense of Fig. 4b / appendix F.3.

    Appendix F.3: gradient clipping is fixed at 1 and only the noise is swept, and both act "on the
    mini-batch level, not the example level". The noise scale `clip * noise` follows the reference
    implementation (flagged in DESIGN_DECISIONS.md section 8).
    """
    if cfg.dp_clip is None and cfg.dp_noise is None:
        return
    with torch.no_grad():
        if cfg.dp_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.dp_clip)
        if cfg.dp_noise is not None:
            scale = (cfg.dp_clip if cfg.dp_clip is not None else 1.0) * cfg.dp_noise
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.add_(torch.randn_like(param) * scale)


def _poison_lookup(poison_ids: list[int] | None) -> dict[int, int]:
    """Map training-set index -> row in the perturbation tensor."""
    if not poison_ids:
        return {}
    return {int(pid): row for row, pid in enumerate(poison_ids)}


def inject_poison(
    inputs: torch.Tensor,
    indices,
    delta: torch.Tensor | None,
    lookup: dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Add `Delta_i` to the mini-batch entries whose dataset index is poisoned.

    The victim never knows which images these are: it simply trains on the dataset it was given,
    reshuffled every epoch (appendix B.2).
    """
    if delta is None or not lookup:
        return inputs
    rows = [lookup.get(int(i), -1) for i in indices]
    if all(row < 0 for row in rows):
        return inputs
    mask = torch.tensor([row >= 0 for row in rows], device=device)
    selected = torch.tensor([row for row in rows if row >= 0], device=device)
    inputs = inputs.clone()
    inputs[mask] = inputs[mask] + delta[selected]
    return inputs


def ablated_subset(trainset, fraction: float, seed: int):
    """Table 7: the victim keeps only a random fraction of the (clean + poisoned) training data."""
    if fraction >= 1.0:
        return trainset
    rng = np.random.default_rng(seed % (2**32 - 1))
    keep = rng.choice(len(trainset), size=int(len(trainset) * fraction), replace=False)
    return Subset(trainset, sorted(int(i) for i in keep))


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    exclude: set[int] | None = None,
) -> float:
    """Top-1 accuracy. `exclude` drops the target images (appendix F.9)."""
    model.eval()
    correct = total = 0
    for inputs, labels, indices in loader:
        if exclude:
            mask = torch.tensor([int(i) not in exclude for i in indices])
            if not bool(mask.any()):
                continue
            inputs, labels = inputs[mask], labels[mask]
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / max(1, total)


@torch.no_grad()
def check_targets(
    model: nn.Module,
    target_inputs: torch.Tensor,
    adv_labels: torch.Tensor,
    true_labels: torch.Tensor,
) -> dict[str, float]:
    """Success criterion of appendix B: the target must be classified *as the adversarial class*."""
    model.eval()
    outputs = model(target_inputs)
    predictions = outputs.argmax(dim=1)
    return {
        "adv_acc": float((predictions == adv_labels).float().mean()),
        "adv_loss": float(nn.functional.cross_entropy(outputs, adv_labels)),
        "orig_acc": float((predictions == true_labels).float().mean()),
    }


def train_model(
    cfg,
    model: nn.Module,
    trainset,
    validset,
    device: torch.device,
    poison_delta: torch.Tensor | None = None,
    poison_ids: list[int] | None = None,
    target_inputs: torch.Tensor | None = None,
    adv_labels: torch.Tensor | None = None,
    true_labels: torch.Tensor | None = None,
    target_ids: list[int] | None = None,
    augment: RandomTransform | None = None,
    diagnostics_fn=None,
    seed: int = 0,
) -> TrainHistory:
    """Train one model, optionally on poisoned data, and record the paper's measurements."""
    history = TrainHistory()
    lookup = _poison_lookup(poison_ids)
    delta = poison_delta.to(device) if poison_delta is not None else None
    # Victim-side purification defense (E2/E3): Psi is applied to the poisoned images at ingest, the
    # way PureGen/PureEBM purify the dataset before training. None keeps the paper path untouched.
    dm, ds = denormalization_tensors(cfg.dataset, device)
    victim_purifier = victim_purifier_normalized(cfg, dm, ds)

    data = ablated_subset(trainset, cfg.data_ablation, seed)
    workers = available_workers(cfg.workers)
    train_loader = make_loader(data, cfg.batch_size, shuffle=True, workers=workers)
    valid_loader = make_loader(validset, 256, shuffle=False, workers=workers)
    # Appendix F.9 measures clean accuracy on the validation set *besides* the targets.
    exclude = {int(i) for i in (target_ids or [])}

    optimizer, scheduler = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    epochs = 1 if cfg.dryrun else cfg.epochs

    def prepare(batch_inputs: torch.Tensor, batch_indices) -> torch.Tensor:
        """Exactly what the victim feeds its network: poisons added, purified, then augmentation."""
        prepared = inject_poison(batch_inputs, batch_indices, delta, lookup, device)
        if victim_purifier is not None:
            # The defense purifies the stored (poisoned) image before normal training proceeds, so
            # Psi sits before augmentation -- the PureGen pipeline order.
            with torch.no_grad():
                prepared = victim_purifier(prepared)
        if augment is not None and cfg.victim_augment:
            prepared = augment(prepared)
        return prepared

    for epoch in range(epochs):
        model.train()
        running_loss = correct = seen = 0
        for inputs, labels, indices in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs = prepare(inputs, indices)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            apply_privacy_defense(model, cfg)
            optimizer.step()

            running_loss += float(loss.detach()) * labels.numel()
            correct += int((outputs.argmax(dim=1) == labels).sum())
            seen += labels.numel()
            if cfg.dryrun:
                break
        scheduler.step()

        history.epochs.append(epoch)
        history.train_loss.append(running_loss / max(1, seen))
        history.train_acc.append(correct / max(1, seen))

        last = epoch == epochs - 1
        if last:
            history.valid_acc.append(evaluate_accuracy(model, valid_loader, device, exclude))
        else:
            history.valid_acc.append(float("nan"))

        if target_inputs is not None:
            stats = check_targets(model, target_inputs, adv_labels, true_labels)
            history.target_adv_acc.append(stats["adv_acc"])
            history.target_adv_loss.append(stats["adv_loss"])
            history.target_orig_acc.append(stats["orig_acc"])

        if diagnostics_fn is not None:
            # Fig. 2 measures the alignment with the gradient of each mini-batch *as the victim
            # sees it*, so the same poisoning and augmentation are applied.
            history.diagnostics.append(diagnostics_fn(model, train_loader, epoch, prepare))

    return history
