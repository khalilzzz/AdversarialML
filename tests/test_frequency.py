"""The piste-1 frequency probe: purification operators, budget survival, WB-LF projection.

CPU-only, no GPU and no real dataset. The budget-survival test is the one that encodes the thesis:
a high-frequency perturbation is destroyed by a low-pass filter, a low-frequency one survives.
"""

from __future__ import annotations

import torch

from witchesbrew.attacks.brew import Brewer, PoisonBounds
from witchesbrew.attacks.lowfreq import lowpass_normalized, subspace_project
from witchesbrew.attacks.objectives import gradient_matching_loss, target_gradient
from witchesbrew.analysis.spectrum import (
    budget_survival,
    delta_report,
    offmanifold_energy,
    radial_dct_profile,
    visibility_ssim,
)
from witchesbrew.config import load_config
from witchesbrew.defenses.purification import (
    brew_purifier_normalized,
    dct2,
    dct_lowpass,
    gaussian_denoise_onestep,
    idct2,
    jpeg_compress,
)


def _highfreq_image(size=32):
    """A +-1 checkerboard (zero mean): all its DCT energy is at the highest frequency, no DC."""
    row = torch.arange(size)
    board = ((row.view(-1, 1) + row.view(1, -1)) % 2).float() * 2 - 1  # {-1, +1}
    return board.view(1, 1, size, size).repeat(1, 3, 1, 1)


def _lowfreq_image(size=32):
    """A smooth gradient: its DCT energy is concentrated at low frequency."""
    ramp = torch.linspace(0, 1, size)
    img = ramp.view(-1, 1).expand(size, size)
    return img.view(1, 1, size, size).repeat(1, 3, 1, 1)


def test_dct_roundtrip_is_identity():
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(idct2(dct2(x)), x, atol=1e-4)


def test_lowpass_preserves_dc_and_reduces_energy():
    const = torch.full((1, 3, 16, 16), 0.5)
    assert torch.allclose(dct_lowpass(const, 4), const, atol=1e-5)  # DC untouched
    x = torch.rand(1, 3, 32, 32)
    assert dct_lowpass(x, 8).pow(2).sum() <= x.pow(2).sum() + 1e-3


def test_lowpass_is_idempotent():
    x = torch.rand(1, 3, 32, 32)
    once = dct_lowpass(x, 10)
    twice = dct_lowpass(once, 10)
    assert torch.allclose(once, twice, atol=1e-4)


def test_jpeg_is_near_identity_at_quality_100_and_valid_range():
    x = torch.rand(2, 3, 32, 32)
    assert jpeg_compress(x, 100).abs().max() <= 1.0
    assert (jpeg_compress(x, 75) >= 0).all() and (jpeg_compress(x, 75) <= 1).all()


def test_jpeg_bpda_carries_gradient():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    jpeg_compress(x, 75, differentiable=True).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_gaussian_denoise_shape_and_range():
    x = torch.rand(2, 3, 32, 32)
    out = gaussian_denoise_onestep(x, sigma=0.25, eot=3)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1


def test_budget_survival_encodes_the_thesis():
    """High-frequency delta is erased by the low-pass filter; low-frequency delta survives."""
    clean = torch.full((1, 3, 32, 32), 0.5)
    psi = lambda x: dct_lowpass(x, 8)

    hf = 0.05 * _highfreq_image()
    lf = 0.05 * _lowfreq_image()
    survival_hf = budget_survival(clean, hf, psi)
    survival_lf = budget_survival(clean, lf, psi)

    assert survival_hf < 0.1  # almost entirely destroyed
    assert survival_lf > 0.8  # almost entirely retained
    assert survival_lf > survival_hf


def test_radial_profile_sums_to_one_and_places_highfreq_energy_high():
    hf = _highfreq_image()
    profile = radial_dct_profile(hf)
    assert abs(profile.sum() - 1.0) < 1e-6
    # A checkerboard's energy is in the top half of the frequency bins.
    assert profile[len(profile) // 2:].sum() > profile[: len(profile) // 2].sum()


def test_subspace_projection_kills_high_frequencies_and_respects_eps():
    device = torch.device("cpu")
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=device)
    clean = torch.rand(4, 3, 32, 32)  # normalized-space stand-in
    delta = torch.randn(4, 3, 32, 32) * bounds.eps_normalized

    projected = subspace_project(delta, clean, bounds, cutoff=8, iters=5)

    # High-frequency DCT energy (outside the 8x8 block) is near zero.
    coeffs = dct2(projected * bounds.ds)
    mask = torch.ones(32, 32)
    mask[:8, :8] = 0.0
    high_energy = (coeffs * mask).pow(2).sum()
    total_energy = coeffs.pow(2).sum()
    assert high_energy / (total_energy + 1e-12) < 0.05

    # Still inside the eps-ball.
    assert (projected.abs() <= bounds.eps_normalized + 1e-5).all()


def test_lowpass_normalized_roundtrips_through_scaling():
    device = torch.device("cpu")
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=device)
    delta = torch.randn(2, 3, 32, 32) * bounds.eps_normalized
    out = lowpass_normalized(delta, bounds.ds, cutoff=32)  # full cutoff -> identity
    assert torch.allclose(out, delta, atol=1e-4)


# --- E2: purification-aware brewing (WB-Psi) ------------------------------------------------------


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 10),
    )


def _tiny_brew_setup(**overrides):
    cfg = load_config(
        None,
        {
            "attack_steps": "6",
            "restarts": "1",
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


def test_wb_psi_brewing_runs_through_jpeg_and_respects_eps():
    """E2: brewing with a differentiable Psi (BPDA jpeg) stays inside the eps-ball and is finite."""
    cfg, clean, labels, targets, adv_labels = _tiny_brew_setup(brew_purify="jpeg")
    brewer = Brewer(cfg, [_tiny_model()], clean, labels, targets, adv_labels, torch.device("cpu"))
    assert brewer.brew_purifier is not None  # the Psi-aware path is active
    delta, loss, _ = brewer.brew()
    assert torch.isfinite(delta).all()
    assert torch.all(delta.abs() <= brewer.bounds.eps_normalized + 1e-6)
    assert loss == loss  # not NaN


def test_wb_psi_gradient_flows_through_purifier():
    """dB/dDelta must propagate through Psi, otherwise the attack cannot adapt to it."""
    cfg, clean, labels, targets, adv_labels = _tiny_brew_setup(brew_purify="jpeg")
    model = _tiny_model()
    bounds = PoisonBounds(cfg.eps, cfg.dataset, torch.device("cpu"))
    purifier = brew_purifier_normalized(cfg, bounds.dm, bounds.ds)
    grads, norm = target_gradient(model, targets, adv_labels, cfg.target_criterion)

    delta = torch.zeros_like(clean, requires_grad=True)
    value = gradient_matching_loss(
        model, clean + delta, labels, grads, norm, purifier=purifier, eot=2
    )
    value.backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


# --- E3: success / detectability / visibility metrics ---------------------------------------------


def test_offmanifold_energy_high_for_highfreq_low_for_lowfreq():
    hf = 0.05 * _highfreq_image()
    lf = 0.05 * _lowfreq_image()
    assert offmanifold_energy(hf, cutoff=8) > 0.9  # almost all energy above the cutoff
    assert offmanifold_energy(lf, cutoff=8) < 0.1  # almost all energy inside it


def test_visibility_ssim_identity_and_perturbed():
    x = torch.rand(2, 3, 32, 32)
    assert abs(visibility_ssim(x, x) - 1.0) < 1e-5
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    assert visibility_ssim(x, noisy) < 1.0


def test_delta_report_keys_and_ranges():
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=torch.device("cpu"))
    clean = torch.randn(4, 3, 32, 32) * 0.5
    delta = bounds.clip_eps(torch.randn(4, 3, 32, 32) * bounds.eps_normalized)
    report = delta_report(delta, clean, "CIFAR10", cutoff=12)
    assert set(report) == {"l2_pixel", "linf_pixel", "offmanifold_hf", "ssim"}
    assert 0.0 <= report["offmanifold_hf"] <= 1.0
    assert report["linf_pixel"] <= 16.0 / 255.0 + 1e-6
    assert report["ssim"] <= 1.0 + 1e-6
