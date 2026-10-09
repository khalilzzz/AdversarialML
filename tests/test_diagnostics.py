"""Tests for the analysis measurements of section 4 / appendix F.4.

These exist because Fig. 2 and Fig. 5 are the paper's evidence for *why* the attack works, so the
quantity plotted has to be the one the paper defines, measured without disturbing the model it is
measured on.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from witchesbrew.attacks.objectives import gradient_matching_loss, target_gradient
from witchesbrew.diagnostics import epoch_alignment


class OneBatch(Dataset):
    """A loader whose single batch is known, so the measurement can be checked in closed form."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images, self.labels = images, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index], index


def plain_model(seed: int = 0) -> torch.nn.Module:
    """No batch norm, so train and eval mode compute the same function."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    )


def toy_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    )


def test_alignment_equals_one_minus_the_brewing_objective():
    """Fig. 2's cosine and eq. 3's `B` are the same quantity, so they must agree numerically.

    Uses a batch-norm-free model so the comparison isolates the formula: with batch norm the two
    differ by design, because the target gradient is taken in eval mode (see the test below).
    """
    device = torch.device("cpu")
    model = plain_model()
    torch.manual_seed(5)
    images, labels = torch.randn(6, 3, 16, 16), torch.randint(0, 10, (6,))
    targets, adv_labels = torch.randn(1, 3, 16, 16), torch.tensor([4])

    loader = DataLoader(OneBatch(images, labels), batch_size=6, shuffle=False)
    stats = epoch_alignment(model, loader, device, targets, adv_labels)

    grads, gnorm = target_gradient(model, targets, adv_labels)
    value = gradient_matching_loss(model, images, labels, grads, gnorm)
    assert abs(stats["cosine_adv"] - (1 - float(value.detach()))) < 1e-5


def test_eq4_bound_uses_the_published_expression():
    """Eq. 4's right-hand side is `(1 - B) * ||grad L|| / ||grad L_adv||` (beta folded out)."""
    device = torch.device("cpu")
    model = toy_model(1)
    torch.manual_seed(6)
    images, labels = torch.randn(4, 3, 16, 16), torch.randint(0, 10, (4,))
    targets, adv_labels = torch.randn(1, 3, 16, 16), torch.tensor([2])

    loader = DataLoader(OneBatch(images, labels), batch_size=4, shuffle=False)
    stats = epoch_alignment(model, loader, device, targets, adv_labels)

    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    adv_grad = torch.cat(
        [
            g.reshape(-1)
            for g in torch.autograd.grad(
                torch.nn.functional.cross_entropy(model(targets), adv_labels), params
            )
        ]
    )
    model.train()
    batch_grad = torch.cat(
        [
            g.reshape(-1)
            for g in torch.autograd.grad(
                torch.nn.functional.cross_entropy(model(images), labels), params
            )
        ]
    )
    expected = stats["cosine_adv"] * float(batch_grad.norm()) / float(adv_grad.norm())
    assert abs(stats["bound_eq4"] - expected) < 1e-4


def test_measurement_does_not_disturb_the_model_being_measured():
    """Train mode is needed for the victim's own gradients; batch-norm state must survive it."""
    device = torch.device("cpu")
    model = toy_model(2)
    torch.manual_seed(7)
    images, labels = torch.randn(8, 3, 16, 16), torch.randint(0, 10, (8,))
    targets, adv_labels = torch.randn(1, 3, 16, 16), torch.tensor([1])

    model.train()
    model(torch.randn(8, 3, 16, 16))  # give the running statistics a non-default value
    before = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
    weights_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    loader = DataLoader(OneBatch(images, labels), batch_size=4, shuffle=False)
    epoch_alignment(model, loader, device, targets, adv_labels, true_labels=torch.tensor([3]))

    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, before[name]), f"{name} was modified by the measurement"
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights_before[key]), f"{key} was modified by the measurement"
    assert all(p.grad is None for p in model.parameters())


def test_original_label_alignment_is_reported_when_requested():
    """Fig. 11c plots the same metric for the target's original label."""
    device = torch.device("cpu")
    model = toy_model(3)
    torch.manual_seed(8)
    images, labels = torch.randn(4, 3, 16, 16), torch.randint(0, 10, (4,))
    loader = DataLoader(OneBatch(images, labels), batch_size=4, shuffle=False)
    stats = epoch_alignment(
        model,
        loader,
        device,
        torch.randn(1, 3, 16, 16),
        torch.tensor([4]),
        true_labels=torch.tensor([7]),
    )
    assert "cosine_orig" in stats
    assert -1.0 <= stats["cosine_orig"] <= 1.0
    assert -1.0 <= stats["cosine_adv"] <= 1.0


def test_prepare_hook_changes_the_measured_batches():
    """The victim's batches carry the perturbation, so the measurement must use them."""
    device = torch.device("cpu")
    model = toy_model(4)
    torch.manual_seed(9)
    images, labels = torch.randn(4, 3, 16, 16), torch.randint(0, 10, (4,))
    loader = DataLoader(OneBatch(images, labels), batch_size=4, shuffle=False)
    args = (model, loader, device, torch.randn(1, 3, 16, 16), torch.tensor([4]))

    plain = epoch_alignment(*args)
    shifted = epoch_alignment(*args, prepare=lambda x, idx: x + 1.0)
    assert plain["cosine_adv"] != shifted["cosine_adv"]


def test_target_gradient_is_taken_in_eval_mode():
    """A single target image has no usable batch statistics, so batch norm must use running ones.

    The mini-batch gradients keep train mode, because those are the updates the victim actually
    applies; only the adversarial gradient switches. With batch norm present the two modes give
    different gradients, which is why this choice is explicit rather than incidental.
    """
    from witchesbrew.diagnostics import adversarial_gradient

    model = toy_model(11)
    targets, adv_labels = torch.randn(1, 3, 16, 16), torch.tensor([5])

    model.train()
    in_eval = adversarial_gradient(model, targets, adv_labels)
    assert model.training, "the model's mode must be restored after the measurement"
    assert torch.isfinite(in_eval).all()

    model.eval()
    again = adversarial_gradient(model, targets, adv_labels)
    assert torch.allclose(in_eval, again, atol=1e-6)
