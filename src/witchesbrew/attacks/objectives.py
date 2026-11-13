"""The objectives that can be placed inside the brewing loop.

Table 2 of the paper swaps exactly this component and keeps everything else fixed: "we fix our
'brewing' framework of efficient data poisoning, with only a single network and diff. data
augmentation. We evaluate the discussed gradient matching cost function, replacing it with either
the feature-collision objective of Poison Frogs or the bullseye objective of Aghakhani et al.
(2020)". So all four objectives below share the optimizer, restarts, augmentation and evaluation.

* `gradient_matching` with `cosine`    -- eq. 3, the paper's method.
* `gradient_matching` with `euclidean` -- the section 3.3 Remark / Table 8 ablation, which directly
  optimizes eq. 2 and is reported as stronger for thin models (Fig. 12 left).
* `poison_frogs`  -- feature collision (Shafahi et al. 2018).
* `bullseye`      -- feature centroid matching (Aghakhani et al. 2020).
"""

from __future__ import annotations

import torch

from ..models.factory import bypass_last_layer
from .losses import get_target_criterion


def target_gradient(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion_name: str = "cross-entropy",
) -> tuple[list[torch.Tensor], float]:
    """Gradient of the adversarial loss w.r.t. all model parameters, plus its norm.

    Section 3.3: the clean model's parameters `theta` are kept fixed, so this is computed once per
    case and then treated as a constant. That is what makes one attack step cost a single
    differentiation instead of an unrolled training trajectory.
    """
    criterion = get_target_criterion(criterion_name)
    params = [p for p in model.parameters() if p.requires_grad]
    loss = criterion(model(inputs), labels)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    grads = [g.detach() for g in grads]
    norm = torch.sqrt(sum(g.pow(2).sum() for g in grads)).item()
    return grads, norm


def _cosine_global(
    poison_grad: list[torch.Tensor], target_grad: list[torch.Tensor]
) -> torch.Tensor:
    """1 - cosine similarity computed over the concatenation of all parameters (eq. 3 literally)."""
    inner = sum((p * t).sum() for p, t in zip(poison_grad, target_grad))
    poison_norm = torch.sqrt(sum(p.pow(2).sum() for p in poison_grad))
    target_norm = torch.sqrt(sum(t.pow(2).sum() for t in target_grad))
    return 1 - inner / (poison_norm * target_norm + 1e-12)


def _cosine_per_layer(
    poison_grad: list[torch.Tensor], target_grad: list[torch.Tensor]
) -> torch.Tensor:
    """Mean of the per-parameter-tensor cosine similarities (the reference's `cosine1` option)."""
    total = poison_grad[0].new_zeros(())
    for p, t in zip(poison_grad, target_grad):
        total = total + torch.nn.functional.cosine_similarity(p.flatten(), t.flatten(), dim=0)
    return 1 - total / len(poison_grad)


def _euclidean(poison_grad: list[torch.Tensor], target_grad: list[torch.Tensor]) -> torch.Tensor:
    """Half the squared l2 distance between the gradients, i.e. eq. 2 optimized directly."""
    return 0.5 * sum((p - t).pow(2).sum() for p, t in zip(poison_grad, target_grad))


def gradient_matching_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    target_grad: list[torch.Tensor],
    target_gnorm: float,
    objective: str = "cosine",
    cosine_mode: str = "global",
    train_criterion: torch.nn.Module | None = None,
    gradient_hook=None,
    purifier=None,
    eot: int = 1,
) -> torch.Tensor:
    """`B(Delta, theta)` of eq. 3, differentiable w.r.t. `inputs`.

    `create_graph=True` is the whole trick: the poison gradient stays part of the autograd graph, so
    one more backward pass gives d B / d Delta. Section 3.3: "Each optimization step of this attack
    requires only a single differentiation of the parameter gradient w.r.t to its input".

    Reduction: the criterion averages over the poison batch, so the poison gradient is the *mean* of
    the per-example gradients. Eq. 3 writes a sum and eq. 2 writes the mean `(1/P) sum`; for the
    cosine objective the difference is a positive scale factor and therefore invisible, and for the
    Euclidean objective the mean is what eq. 2 asks for.

    E2 (WB-Psi): when `purifier` is given the poison gradient is taken on `Psi(inputs)` and averaged
    over `eot` fresh draws of Psi -- the expectation `E_xi [ sum_i grad l(Psi_xi(x_i+delta_i)) ]`
    sits *inside* the cosine, so the brewer matches the gradient the purifying victim will actually
    produce. The graph is kept through Psi (BPDA for jpeg), so dB/dDelta propagates through it.
    """
    criterion = train_criterion or torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if purifier is None:
        loss = criterion(model(inputs), labels)
        poison_grad = list(torch.autograd.grad(loss, params, create_graph=True))
    else:
        accumulated: list[torch.Tensor] | None = None
        draws = max(1, eot)
        for _ in range(draws):
            loss = criterion(model(purifier(inputs)), labels)
            grad = torch.autograd.grad(loss, params, create_graph=True)
            accumulated = (
                list(grad) if accumulated is None else [a + g for a, g in zip(accumulated, grad)]
            )
        poison_grad = [g / draws for g in accumulated]

    if gradient_hook is not None:
        # Appendix F.3's attacker-aware counter-attack: the attacker applies the defense's own
        # clipping and freshly drawn noise to the poison gradient every time B is evaluated.
        poison_grad = gradient_hook(poison_grad)

    if objective == "cosine":
        value = _cosine_global(poison_grad, target_grad) if cosine_mode == "global" else _cosine_per_layer(
            poison_grad, target_grad
        )
    elif objective == "euclidean":
        # Normalizing by ||g_target|| keeps the reported value comparable across cases; it is a
        # positive constant, so it does not change the argmin (reference code does the same).
        value = _euclidean(poison_grad, target_grad) / max(target_gnorm, 1e-12)
    else:
        raise ValueError(f"unknown gradient matching objective: {objective}")
    return value


def poison_frogs_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_inputs: torch.Tensor,
) -> torch.Tensor:
    """Feature collision: squared distance between the mean poison feature and the target feature."""
    feature_model, _ = bypass_last_layer(model)
    poison_features = feature_model(inputs)
    with torch.no_grad():
        target_features = feature_model(target_inputs)
    return (poison_features.mean(dim=0, keepdim=True) - target_features).pow(2).mean()


def bullseye_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_inputs: torch.Tensor,
) -> torch.Tensor:
    """Bullseye polytope: distance between the poison feature centroid and the target centroid."""
    feature_model, _ = bypass_last_layer(model)
    poison_features = feature_model(inputs)
    with torch.no_grad():
        target_features = feature_model(target_inputs)
    gap = poison_features.mean(dim=0) - target_features.mean(dim=0)
    return gap.pow(2).mean()
