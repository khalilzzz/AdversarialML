"""Integration tests for the brewing loop itself, on a tiny model so they run on CPU in seconds.

What is checked here is the part of the paper that is actually falsifiable without a full training
run: that Algorithm 1 decreases `B(Delta, theta)` — i.e. that the poisons' gradient really does get
aligned with the adversarial gradient — and that the perturbation it returns satisfies the threat
model of section 3.1.
"""

from __future__ import annotations

import torch

from witchesbrew.attacks.brew import Brewer
from witchesbrew.attacks.objectives import gradient_matching_loss, target_gradient
from witchesbrew.config import load_config
from witchesbrew.data.augment import build_augmentation


def tiny_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    )


def tiny_setup(steps: int = 12, **overrides):
    cfg = load_config(
        None,
        {
            "attack_steps": str(steps),
            "restarts": "2",
            "poison_batch_size": "8",
            "cases": "1",
            "victim_runs": "1",
            **{k: str(v) for k, v in overrides.items()},
        },
    )
    torch.manual_seed(1)
    clean = torch.randn(8, 3, 32, 32) * 0.5
    labels = torch.full((8,), 3)
    targets = torch.randn(1, 3, 32, 32) * 0.5
    adv_labels = torch.tensor([3])
    return cfg, clean, labels, targets, adv_labels


def current_objective(cfg, model, clean, delta, labels, targets, adv_labels) -> float:
    grads, norm = target_gradient(model, targets, adv_labels, cfg.target_criterion)
    value = gradient_matching_loss(
        model,
        clean + delta,
        labels,
        grads,
        norm,
        objective=cfg.objective,
        cosine_mode=cfg.cosine_mode,
    )
    return float(value.detach())


def test_brewing_reduces_the_alignment_loss():
    """Eq. 3 is what Algorithm 1 minimizes, so the returned Delta must beat the initialization."""
    cfg, clean, labels, targets, adv_labels = tiny_setup()
    model = tiny_model()
    device = torch.device("cpu")

    before = current_objective(
        cfg, model, clean, torch.zeros_like(clean), labels, targets, adv_labels
    )
    brewer = Brewer(cfg, [model], clean, labels, targets, adv_labels, device, augment=None)
    delta, loss, per_restart = brewer.brew()
    after = current_objective(cfg, model, clean, delta, labels, targets, adv_labels)

    assert after < before
    assert loss == min(per_restart)  # Alg. 1 line 10 keeps the best restart
    assert len(per_restart) == cfg.restarts


def test_brewing_respects_the_threat_model():
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=6)
    brewer = Brewer(
        cfg, [tiny_model()], clean, labels, targets, adv_labels, torch.device("cpu"), augment=None
    )
    delta, _, _ = brewer.brew()
    assert torch.all(delta.abs() <= brewer.bounds.eps_normalized + 1e-6)
    assert delta.shape == clean.shape


def test_brewing_works_through_differentiable_augmentation():
    """Section 3.4: the perturbation is updated through the augmentation, not around it."""
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=10)
    brewer = Brewer(
        cfg,
        [tiny_model()],
        clean,
        labels,
        targets,
        adv_labels,
        torch.device("cpu"),
        augment=build_augmentation("CIFAR10"),
    )
    delta, loss, _ = brewer.brew()
    assert delta.abs().sum() > 0
    assert torch.isfinite(delta).all()
    assert loss < 1.0  # some alignment achieved despite the random transforms


def test_euclidean_objective_also_decreases():
    """The section 3.3 Remark / Table 8 variant has to optimize too, not just exist."""
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=12, objective="euclidean")
    model = tiny_model()
    before = current_objective(
        cfg, model, clean, torch.zeros_like(clean), labels, targets, adv_labels
    )
    brewer = Brewer(cfg, [model], clean, labels, targets, adv_labels, torch.device("cpu"), augment=None)
    delta, _, _ = brewer.brew()
    assert current_objective(cfg, model, clean, delta, labels, targets, adv_labels) < before


def test_ensemble_averages_over_models():
    """Section 3.4's ensemble: the objective is the mean over K independently initialized models."""
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=6, ensemble=2)
    models = [tiny_model(0), tiny_model(1)]
    brewer = Brewer(cfg, models, clean, labels, targets, adv_labels, torch.device("cpu"), augment=None)
    assert len(brewer.target_grads) == 2
    delta, loss, _ = brewer.brew()
    averages = [
        current_objective(cfg, model, clean, delta, labels, targets, adv_labels) for model in models
    ]
    # Both models are attacked, so neither should be left at the trivial value of 1.
    assert all(value < 1.0 for value in averages)


def test_poison_frogs_recipe_runs_and_lowers_its_own_objective():
    """Table 2 swaps only the objective; the framework must accept it unchanged."""
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=10, recipe="poison-frogs")
    from witchesbrew.attacks.objectives import poison_frogs_loss

    model = tiny_model()
    before = float(poison_frogs_loss(model, clean, targets).detach())
    brewer = Brewer(cfg, [model], clean, labels, targets, adv_labels, torch.device("cpu"), augment=None)
    delta, _, _ = brewer.brew()
    assert float(poison_frogs_loss(model, clean + delta, targets).detach()) < before


# --- numerical correctness of the objective itself ------------------------
def double_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(6, 10),
    ).double()


def small_problem(seed: int = 0):
    torch.manual_seed(seed)
    model = double_model(seed)
    targets = torch.randn(1, 3, 8, 8, dtype=torch.float64)
    adv_labels = torch.tensor([3])
    poisons = torch.randn(5, 3, 8, 8, dtype=torch.float64)
    labels = torch.full((5,), 3)
    grads, norm = target_gradient(model, targets, adv_labels)
    return model, poisons, labels, grads, norm


def test_objective_equals_one_minus_cosine_computed_independently():
    """B must be exactly eq. 3 and not merely something that decreases."""
    model, poisons, labels, target_grad, gnorm = small_problem()
    value = gradient_matching_loss(model, poisons, labels, target_grad, gnorm)

    params = list(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(poisons), labels)
    poison_grad = torch.autograd.grad(loss, params)
    flat_p = torch.cat([g.reshape(-1) for g in poison_grad])
    flat_t = torch.cat([g.reshape(-1) for g in target_grad])
    expected = 1 - torch.dot(flat_p, flat_t) / (flat_p.norm() * flat_t.norm())
    assert abs(float(value.detach()) - float(expected)) < 1e-10


def test_objective_gradient_matches_finite_differences():
    """The attack differentiates the parameter gradient w.r.t. the input (section 3.3)."""
    model, poisons, labels, target_grad, gnorm = small_problem()
    delta = torch.zeros_like(poisons, requires_grad=True)
    value = gradient_matching_loss(model, poisons + delta, labels, target_grad, gnorm)
    analytic = torch.autograd.grad(value, delta)[0]

    step, index = 1e-6, (2, 1, 4, 5)
    probe = torch.zeros_like(poisons)
    probe[index] = step
    plus = gradient_matching_loss(model, poisons + probe, labels, target_grad, gnorm)
    probe[index] = -step
    minus = gradient_matching_loss(model, poisons + probe, labels, target_grad, gnorm)
    numeric = (float(plus.detach()) - float(minus.detach())) / (2 * step)
    assert abs(float(analytic[index]) - numeric) <= 1e-5 * max(abs(numeric), 1e-9)


def test_cosine_is_invariant_to_the_gradient_reduction():
    """Eq. 3 writes a sum and eq. 2 a mean; for the cosine that is the same objective."""
    model, poisons, labels, target_grad, gnorm = small_problem()
    with_mean = gradient_matching_loss(model, poisons, labels, target_grad, gnorm)
    with_sum = gradient_matching_loss(
        model, poisons, labels, target_grad, gnorm,
        train_criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    )
    assert abs(float(with_mean.detach()) - float(with_sum.detach())) < 1e-10


def test_exported_perturbation_stays_within_eps_in_pixel_units():
    """Section 3.1 states the bound in 0..255 pixel values, which is what a victim would receive."""
    from witchesbrew.attacks.brew import PoisonBounds
    from witchesbrew.data.datasets import denormalization_tensors
    from witchesbrew.export import quantize, to_pixel_space

    device = torch.device("cpu")
    bounds = PoisonBounds(16.0, "CIFAR10", device)
    dm, ds = denormalization_tensors("CIFAR10", device)
    pixels = torch.rand(16, 3, 32, 32)
    clean = (pixels - dm) / ds
    delta = bounds.project(torch.randn(16, 3, 32, 32) * 99, clean)

    stored = quantize(to_pixel_space(clean + delta, "CIFAR10"))
    assert float((stored - quantize(pixels)).abs().max() * 255) <= 16.0 + 1e-4
    assert float(stored.min()) >= 0.0 and float(stored.max()) <= 1.0


def test_restart_score_describes_the_returned_perturbation():
    """Alg. 1 line 10 compares B at the perturbation each restart actually returns."""
    cfg, clean, labels, targets, adv_labels = tiny_setup(steps=4)
    model = tiny_model()
    brewer = Brewer(cfg, [model], clean, labels, targets, adv_labels, torch.device("cpu"), augment=None)
    delta, loss, _ = brewer.brew()
    recomputed = current_objective(cfg, model, clean, delta, labels, targets, adv_labels)
    assert abs(loss - recomputed) < 1e-6
