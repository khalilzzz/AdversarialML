"""Unit tests for the parts of the attack whose correctness is checkable without training.

These are the properties the paper's method depends on:

* the perturbation stays inside the threat model (section 3.1);
* the augmentation is differentiable, because Delta is updated *through* it (section 3.4);
* the gradient matching objective is really 1 - cosine similarity, so it vanishes when the poison
  gradient points along the target gradient (eq. 3);
* the published cases are loaded verbatim (Table 4).
"""

from __future__ import annotations

import torch

from witchesbrew.attacks.brew import PoisonBounds, initialize_delta
from witchesbrew.attacks.losses import carlini_wagner, cross_entropy
from witchesbrew.attacks.objectives import _cosine_global, _euclidean
from witchesbrew.config import load_config
from witchesbrew.consts import CIFAR10_CASES, CIFAR10_CLASSES
from witchesbrew.data.augment import RandomTransform, build_augmentation
from witchesbrew.models.convnet import convnet
from witchesbrew.models.factory import bypass_last_layer, build_model
from witchesbrew.models.resnet import resnet18

DEVICE = torch.device("cpu")


# --- threat model ---------------------------------------------------------
def test_eps_bound_is_in_pixel_units():
    """eps is given in 0..255 pixel values, so the normalized bound is eps/255/std per channel."""
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=DEVICE)
    expected = 16.0 / 255.0 / torch.tensor([0.24703224, 0.24348514, 0.26158786]).view(1, 3, 1, 1)
    assert torch.allclose(bounds.eps_normalized, expected, atol=1e-6)


def test_projection_enforces_both_constraints():
    """After a step, Delta obeys the eps ball and x + Delta is still a valid image."""
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=DEVICE)
    mean = torch.tensor([0.49146724, 0.48226175, 0.44677013]).view(1, 3, 1, 1)
    std = torch.tensor([0.24703224, 0.24348514, 0.26158786]).view(1, 3, 1, 1)
    # Real images include saturated pixels, where the valid-range clip is the binding constraint.
    pixels = torch.rand(8, 3, 32, 32)
    pixels[0] = 0.0
    pixels[1] = 1.0
    clean = (pixels - mean) / std
    delta = torch.randn(8, 3, 32, 32) * 10
    projected = bounds.project(delta, clean)

    assert torch.all(projected.abs() <= bounds.eps_normalized + 1e-6)
    recovered = (clean + projected) * std + mean
    assert recovered.min() >= -1e-5 and recovered.max() <= 1 + 1e-5
    # A saturated image can only be perturbed inwards.
    assert torch.all(projected[0] >= -1e-6) and torch.all(projected[1] <= 1e-6)


def test_initialization_respects_the_bound():
    bounds = PoisonBounds(eps=8.0, dataset="CIFAR10", device=DEVICE)
    for mode in ("zero", "rand", "randn"):
        delta = initialize_delta((4, 3, 32, 32), bounds, mode, DEVICE)
        assert torch.all(delta.abs() <= bounds.eps_normalized + 1e-6), mode


# --- differentiable augmentation ------------------------------------------
def test_augmentation_is_differentiable_wrt_the_perturbation():
    """Alg. 1 line 7 augments the poisons and line 9 still updates Delta, so gradients must flow."""
    augment = build_augmentation("CIFAR10")
    clean = torch.rand(2, 3, 32, 32)
    delta = torch.zeros(2, 3, 32, 32, requires_grad=True)
    augment(clean + delta).sum().backward()
    assert delta.grad is not None
    assert torch.isfinite(delta.grad).all()
    assert delta.grad.abs().sum() > 0


def test_zero_shift_and_no_flip_is_the_identity():
    """randgen = 0.5 means no translation, so the sampling grid must reproduce the input exactly."""
    augment = RandomTransform(size=32, shift=4, flip=False)
    x = torch.rand(1, 3, 32, 32)
    out = augment(x, randgen=torch.full((1, 3), 0.5))
    assert torch.allclose(out, x, atol=1e-5)


def test_flip_is_a_horizontal_mirror():
    augment = RandomTransform(size=32, shift=4, flip=True)
    x = torch.rand(1, 3, 32, 32)
    randgen = torch.tensor([[0.5, 0.5, 0.9]])  # no translation, flip triggered
    assert torch.allclose(augment(x, randgen), torch.flip(x, dims=[3]), atol=1e-5)


def test_translation_budget_matches_the_paper_padding():
    """Appendix B.2 uses random crops with zero-padding of 4 on CIFAR-10, i.e. +-4 pixels."""
    augment = build_augmentation("CIFAR10")
    assert augment.shift == 4
    assert build_augmentation("ImageNet").shift == 28
    x = torch.ones(1, 3, 32, 32)
    shifted = augment(x, randgen=torch.tensor([[1.0, 0.5, 0.0]]))  # maximum shift in x
    # Zero padding shifted in from the border: exactly `shift` columns become zero.
    zero_columns = int((shifted[0, 0].abs().sum(dim=0) == 0).sum())
    assert zero_columns == 4


def test_augmentation_preserves_shape():
    for dataset, size in (("CIFAR10", 32), ("ImageNet", 224)):
        augment = build_augmentation(dataset)
        x = torch.rand(2, 3, size, size)
        assert augment(x).shape == x.shape


# --- objectives -----------------------------------------------------------
def test_cosine_objective_vanishes_on_aligned_gradients():
    target = [torch.randn(5, 4), torch.randn(3)]
    aligned = [t * 2.5 for t in target]  # same direction, different magnitude
    assert float(_cosine_global(aligned, target)) < 1e-5


def test_cosine_objective_is_two_on_opposed_gradients():
    target = [torch.randn(6)]
    assert abs(float(_cosine_global([-target[0]], target)) - 2.0) < 1e-5


def test_cosine_objective_is_bounded():
    target = [torch.randn(10), torch.randn(4, 4)]
    for _ in range(20):
        poison = [torch.randn_like(t) for t in target]
        value = float(_cosine_global(poison, target))
        assert -1e-6 <= value <= 2 + 1e-6


def test_euclidean_objective_vanishes_only_on_equality():
    target = [torch.randn(7)]
    assert float(_euclidean([target[0].clone()], target)) < 1e-10
    # Unlike the cosine, a rescaled gradient is not optimal: eq. 2 asks for equality, not alignment.
    assert float(_euclidean([target[0] * 2], target)) > 1e-3


def test_carlini_wagner_is_negative_when_the_intended_class_wins():
    outputs = torch.tensor([[5.0, 0.0, 0.0]])
    labels = torch.tensor([0])
    assert float(carlini_wagner(outputs, labels)) < 0
    assert float(cross_entropy(outputs, labels)) < float(cross_entropy(outputs, torch.tensor([1])))


# --- models ---------------------------------------------------------------
def test_convnet_penultimate_width_matches_the_paper():
    """Appendix B.2 lists the output widths as 64, 128, 128, 256, 256, 2304."""
    model = convnet(num_classes=10)
    features, head = bypass_last_layer(model)
    out = features(torch.zeros(1, 3, 32, 32))
    assert out.shape[1] == 2304
    assert head.in_features == 2304


def test_cifar_resnet_stem_has_no_downsampling():
    """Appendix B.2 replaces the ImageNet stem with a 3x3 conv + BN + ReLU."""
    model = resnet18()
    stem_conv = model.stem[0]
    assert stem_conv.kernel_size == (3, 3)
    assert stem_conv.stride == (1, 1)
    assert not any(isinstance(m, torch.nn.MaxPool2d) for m in model.stem)


def test_resnet_width_argument_scales_the_network():
    narrow = sum(p.numel() for p in resnet18(width=16).parameters())
    wide = sum(p.numel() for p in resnet18(width=64).parameters())
    assert wide > 10 * narrow  # Fig. 12 left varies exactly this


def test_models_build_and_run_on_cifar_shapes():
    for name in ("ResNet18", "ConvNet", "VGG11", "MobileNetV2"):
        model = build_model(name, "CIFAR10", num_classes=10)
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


# --- cases and config -----------------------------------------------------
def test_published_case_table_is_loaded_verbatim():
    """First row of the paper's Table 4."""
    target, poison, target_id, seed = CIFAR10_CASES[0]
    assert (target, poison, target_id, seed) == ("dog", "frog", 8745, 2000000000)
    assert len(CIFAR10_CASES) == 10
    for target, poison, _, _ in CIFAR10_CASES:
        assert target in CIFAR10_CLASSES and poison in CIFAR10_CLASSES


def test_base_config_matches_the_published_hyperparameters():
    cfg = load_config()
    assert (cfg.tau, cfg.restarts, cfg.attack_steps) == (0.1, 8, 250)  # section 5
    assert cfg.epochs == 40 and cfg.lr_milestones == [14, 24, 35]  # appendix B.2
    assert cfg.batch_size == 128 and cfg.momentum == 0.9 and cfg.weight_decay == 5e-4
    assert cfg.lr == 0.1  # ResNet default
    assert cfg.victim_runs == 8 and cfg.cases == 10  # appendix B
    assert cfg.is_paper_setting


def test_overrides_mark_the_run_as_non_paper():
    cfg = load_config(None, {"attack_steps": "10"})
    assert cfg.attack_steps == 10
    assert not cfg.is_paper_setting


def test_convnet_learning_rate_default_is_lower():
    """Appendix B.2: 0.01 for the ConvNet, MobileNet-v2 and VGG-16; 0.1 for residual nets."""
    assert load_config(None, {"net": "ConvNet"}).lr == 0.01
    assert load_config(None, {"net": "VGG11"}).lr == 0.01
