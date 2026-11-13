"""Where does Witches' Brew spend its budget, and how much survives purification (E0).

This answers the piste-1 question quantitatively, on perturbations that have already been brewed:

* `radial_dct_profile`  -- the DCT energy of delta binned by radial frequency. The thesis predicts
  the energy sits at high frequencies.
* `budget_survival`     -- ||Psi(x+delta) - Psi(x)||^2 / ||delta||^2: the fraction of the l_inf
  budget that Psi lets through. 1 - survival is the budget destroyed.
* `alignment_survival`  -- the matching objective B before and after Psi. B is what the attack
  minimizes; if it jumps back toward 1 after Psi, the attack's signal has been erased.
* `rho_subspace`        -- ||P x_t|| / ||x_t||, the fraction of the target's energy inside the
  preserved subspace. For the orthogonal low-pass projector this is the exact upper bound on the
  achievable alignment from the linear Proposition; for jpeg/gaussian it is the analogous
  energy-retention ratio.

Everything works in pixel space ([0, 1]); a normalized perturbation delta is scaled to pixel space by
the per-channel std (no mean added -- delta is a difference).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..data.datasets import denormalization_tensors
from ..defenses.purification import dct2
from ..diagnostics import _flat_grad
from ..attacks.objectives import target_gradient


def delta_to_pixel(delta: torch.Tensor, dataset: str) -> torch.Tensor:
    """Scale a normalized perturbation into pixel units (multiply by std, no mean)."""
    _, ds = denormalization_tensors(dataset, delta.device)
    return delta * ds


def radial_dct_profile(delta_pixel: torch.Tensor, bins: int | None = None) -> np.ndarray:
    """Mean DCT energy of delta binned by radial frequency sqrt(u^2 + v^2), averaged over N and C."""
    coeffs = dct2(delta_pixel)  # [N, C, H, W]
    energy = coeffs.pow(2).mean(dim=(0, 1))  # [H, W]
    n = energy.shape[-1]
    u = torch.arange(n, device=energy.device).view(-1, 1).expand(n, n)
    v = torch.arange(n, device=energy.device).view(1, -1).expand(n, n)
    radius = torch.sqrt((u.float()) ** 2 + (v.float()) ** 2)
    nbins = bins or n
    edges = torch.linspace(0, float(radius.max()) + 1e-6, nbins + 1, device=energy.device)
    profile = np.zeros(nbins, dtype=np.float64)
    for b in range(nbins):
        m = (radius >= edges[b]) & (radius < edges[b + 1])
        if m.any():
            profile[b] = float(energy[m].sum())
    total = profile.sum()
    return profile / total if total > 0 else profile


def budget_survival(
    clean_pixel: torch.Tensor, delta_pixel: torch.Tensor, psi: Callable[[torch.Tensor], torch.Tensor]
) -> float:
    """Fraction of the perturbation's energy that passes through Psi.

    For a stochastic Psi (e.g. the Gaussian denoiser) the two evaluations must share the same noise
    realization -- common random numbers -- or the noise variance swamps the perturbation and the
    ratio blows up. The global RNG state is reset identically before each call. The value can mildly
    exceed 1 for a non-linear Psi (JPEG quantization can push a coefficient across a bin boundary).
    """
    poisoned = (clean_pixel + delta_pixel).clamp(0.0, 1.0)
    state = torch.random.get_rng_state()
    out_poisoned = psi(poisoned)
    torch.random.set_rng_state(state)
    out_clean = psi(clean_pixel)
    diff = out_poisoned - out_clean
    denom = float(delta_pixel.pow(2).sum())
    if denom <= 0:
        return 0.0
    return float(diff.pow(2).sum()) / denom


def rho_subspace(x_t_pixel: torch.Tensor, psi: Callable[[torch.Tensor], torch.Tensor]) -> float:
    """||Psi(x_t)|| / ||x_t||: the fraction of the target's energy the subspace retains."""
    num = float(psi(x_t_pixel).pow(2).sum()) ** 0.5
    den = float(x_t_pixel.pow(2).sum()) ** 0.5
    return num / den if den > 0 else 0.0


def offmanifold_energy(delta_pixel: torch.Tensor, cutoff: int) -> float:
    """Fraction of delta's DCT energy outside the top-left cutoff x cutoff block.

    A synthetic stand-in for the EBM-energy detectability of E3: a purification defence flags the
    high-frequency, off-manifold component, so the larger this fraction the more detectable (and the
    more of the budget a low-pass Psi will erase). 0 = all energy inside the preserved subspace.
    """
    coeffs = dct2(delta_pixel)  # [N, C, H, W]
    energy = coeffs.pow(2)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    n = coeffs.shape[-1]
    c = max(1, min(int(cutoff), n))
    low = float(energy[..., :c, :c].sum())
    return max(0.0, 1.0 - low / total)


def _ssim_single(a: torch.Tensor, b: torch.Tensor) -> float:
    """Global single-scale SSIM between two pixel-space images (per channel, then averaged)."""
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(num / den)


def visibility_ssim(clean_pixel: torch.Tensor, poisoned_pixel: torch.Tensor) -> float:
    """Mean SSIM over the batch between the clean and poisoned images. 1 = invisible perturbation."""
    scores = [
        _ssim_single(clean_pixel[i], poisoned_pixel[i]) for i in range(clean_pixel.shape[0])
    ]
    return sum(scores) / max(1, len(scores))


def delta_report(delta_norm: torch.Tensor, clean_norm: torch.Tensor, dataset: str, cutoff: int) -> dict:
    """Per-case perturbation summary for the E3 success/detectability/visibility trade-off.

    Everything is averaged over the poison batch. `l2_pixel`/`linf_pixel` are in 0..1 pixel units;
    `offmanifold_hf` is the detectability proxy; `ssim` is the visibility (1 = invisible).
    """
    dm, ds = denormalization_tensors(dataset, delta_norm.device)
    delta_pixel = delta_norm * ds
    clean_pixel = (clean_norm * ds + dm).clamp(0.0, 1.0)
    poisoned_pixel = (clean_pixel + delta_pixel).clamp(0.0, 1.0)
    per_image_l2 = delta_pixel.flatten(1).pow(2).sum(dim=1).sqrt().mean()
    return {
        "l2_pixel": float(per_image_l2),
        "linf_pixel": float(delta_pixel.abs().max()),
        "offmanifold_hf": offmanifold_energy(delta_pixel, cutoff),
        "ssim": visibility_ssim(clean_pixel, poisoned_pixel),
    }


def _poison_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Flat gradient of the training loss on the poisons w.r.t. the model parameters."""
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    return _flat_grad(model, loss)


def _to_normalized(pixel: torch.Tensor, dataset: str) -> torch.Tensor:
    dm, ds = denormalization_tensors(dataset, pixel.device)
    return (pixel - dm) / ds


def alignment_survival(
    model: torch.nn.Module,
    clean_norm: torch.Tensor,
    delta_norm: torch.Tensor,
    labels: torch.Tensor,
    target_inputs: torch.Tensor,
    adv_labels: torch.Tensor,
    psi: Callable[[torch.Tensor], torch.Tensor],
    dataset: str,
    criterion_name: str = "cross-entropy",
) -> dict:
    """The matching objective B = 1 - cos(g_target, g_poison) before and after Psi.

    g_poison is taken on the poisoned images as the victim would ingest them -- raw for `before`,
    purified for `after` -- so a large jump in B is the attack's signal being filtered out.
    """
    model.eval()
    tgt_grads, tgt_norm = target_gradient(model, target_inputs, adv_labels, criterion_name)
    g_target = torch.cat([g.reshape(-1) for g in tgt_grads])

    def cos_to_target(inputs_norm: torch.Tensor) -> float:
        g = _poison_gradient(model, inputs_norm, labels)
        return float(torch.dot(g, g_target) / (g.norm() * g_target.norm() + 1e-12))

    poisoned_norm = clean_norm + delta_norm
    cos_before = cos_to_target(poisoned_norm)

    dm, ds = denormalization_tensors(dataset, clean_norm.device)
    poisoned_pixel = (poisoned_norm * ds + dm).clamp(0.0, 1.0)
    purified_norm = _to_normalized(psi(poisoned_pixel), dataset)
    cos_after = cos_to_target(purified_norm)

    return {
        "B_before": 1.0 - cos_before,
        "B_after": 1.0 - cos_after,
        "cos_before": cos_before,
        "cos_after": cos_after,
        "alignment_retained": (cos_after / cos_before) if cos_before != 0 else 0.0,
    }
