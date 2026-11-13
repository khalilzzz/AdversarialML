"""Synthetic purification operators Psi (research extension, NOT the paper).

Piste 1 tests the thesis that denoising defenses beat Witches' Brew because the attack spends its
l_inf budget in a subspace -- high frequencies, off the image manifold -- that purification erases.
To measure that without any external model, three synthetic Psi are provided, all operating in pixel
space ([0, 1], shape [N, C, H, W]):

* `dct_lowpass`            -- the orthogonal low-pass projector P of the linear Proposition: keep the
                             top-left cutoff x cutoff 2D-DCT coefficients, drop the rest.
* `jpeg_compress`          -- block-wise 8x8 DCT, quantization by the standard luminance table scaled
                             by `quality`, dequantization, inverse DCT. This is the filter PureGen
                             reports takes WB to 0%. Quantization is non-differentiable; a
                             straight-through (BPDA) variant is provided for gradient use.
* `gaussian_denoise_onestep` -- EOT mean of `blur(x + sigma*eta)`: a synthetic stand-in for the
                             optimal one-step denoiser, which near the manifold is locally a linear
                             projection onto the tangent plane (Tweedie). It is an approximation, not
                             a trained DDPM; the real-DDPM setting of Hong et al. (2024) is left as a
                             documented hook (E2).

The DCT here is the orthonormal DCT-II, so `idct2(dct2(x)) == x` and the low-pass map is a genuine
orthogonal projection, which is what the linear bound rho_S(x_t) = ||P x_t|| / ||x_t|| needs.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn.functional as F

# Standard JPEG luminance quantization table (ITU-T T.81, Annex K.1).
_JPEG_LUMA = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)


@functools.lru_cache(maxsize=8)
def _dct_matrix(n: int, device: str, dtype_str: str) -> torch.Tensor:
    """Orthonormal DCT-II matrix D of size [n, n], so `D @ x` is the 1D DCT along a length-n axis."""
    dtype = getattr(torch, dtype_str)
    k = torch.arange(n, dtype=dtype).view(-1, 1)  # frequency index
    m = torch.arange(n, dtype=dtype).view(1, -1)  # spatial index
    basis = torch.cos(torch.pi * (2 * m + 1) * k / (2 * n))
    scale = torch.full((n, 1), (2.0 / n) ** 0.5, dtype=dtype)
    scale[0, 0] = (1.0 / n) ** 0.5
    return (basis * scale).to(device)


def _basis_for(x: torch.Tensor, n: int) -> torch.Tensor:
    return _dct_matrix(n, str(x.device), str(x.dtype).rsplit(".", 1)[-1])


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D DCT-II over the last two axes (assumed square)."""
    d = _basis_for(x, x.shape[-1])
    return d @ x @ d.transpose(-1, -2)


def idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `dct2` (the transpose, since the basis is orthonormal)."""
    d = _basis_for(x, x.shape[-1])
    return d.transpose(-1, -2) @ x @ d


def dct_lowpass(x: torch.Tensor, cutoff: int) -> torch.Tensor:
    """Keep the top-left `cutoff x cutoff` DCT coefficients, drop the rest. Orthogonal projector P."""
    coeffs = dct2(x)
    n = coeffs.shape[-1]
    c = max(1, min(cutoff, n))
    mask = torch.zeros(n, n, device=x.device, dtype=x.dtype)
    mask[:c, :c] = 1.0
    return idct2(coeffs * mask)


def _blockify(x: torch.Tensor, block: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reshape [N, C, H, W] into [..., block, block] tiles, padding H/W up to a multiple of block."""
    n, c, h, w = x.shape
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    xp = F.pad(x, (0, pad_w, 0, pad_h))
    H, W = xp.shape[-2:]
    tiles = (
        xp.reshape(n, c, H // block, block, W // block, block)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(-1, block, block)
    )
    return tiles, (h, w)


def _unblockify(tiles: torch.Tensor, shape: tuple[int, int, int, int], block: int, crop: tuple[int, int]) -> torch.Tensor:
    n, c, h, w = shape
    H = ((h + block - 1) // block) * block
    W = ((w + block - 1) // block) * block
    x = (
        tiles.reshape(n, c, H // block, W // block, block, block)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(n, c, H, W)
    )
    return x[..., : crop[0], : crop[1]]


def _jpeg_quant_table(quality: int, device, dtype) -> torch.Tensor:
    quality = int(max(1, min(100, quality)))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = torch.floor((_JPEG_LUMA.to(device=device, dtype=dtype) * scale + 50) / 100)
    return table.clamp_min(1.0)


def jpeg_compress(x: torch.Tensor, quality: int = 75, differentiable: bool = False) -> torch.Tensor:
    """Block-wise 8x8 DCT quantization. `differentiable=True` uses a straight-through round (BPDA)."""
    block = 8
    tiles, crop = _blockify(x * 255.0 - 128.0, block)  # JPEG works on level-shifted [-128, 127]
    coeffs = dct2(tiles)
    q = _jpeg_quant_table(quality, x.device, x.dtype)
    scaled = coeffs / q
    if differentiable:
        rounded = scaled + (torch.round(scaled) - scaled).detach()  # straight-through estimator
    else:
        rounded = torch.round(scaled)
    recovered = idct2(rounded * q)
    out = _unblockify(recovered, x.shape, block, crop)
    return ((out + 128.0) / 255.0).clamp(0.0, 1.0)


def _gaussian_kernel(sigma: float, device, dtype, radius: int = 2) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1 = torch.exp(-(coords**2) / (2 * sigma**2))
    k1 = k1 / k1.sum()
    return torch.outer(k1, k1)


def gaussian_denoise_onestep(
    x: torch.Tensor, sigma: float = 0.25, eot: int = 1, generator: torch.Generator | None = None
) -> torch.Tensor:
    """EOT mean of blur(x + sigma*eta). Synthetic one-step denoiser; a local linear manifold map."""
    kernel = _gaussian_kernel(max(sigma, 1e-3), x.device, x.dtype)
    k = kernel.shape[-1]
    weight = kernel.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
    acc = torch.zeros_like(x)
    for _ in range(max(1, eot)):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
        blurred = F.conv2d(x + noise, weight, padding=k // 2, groups=x.shape[1])
        acc = acc + blurred
    return (acc / max(1, eot)).clamp(0.0, 1.0)


def _psi_by_kind(kind: str, cfg, differentiable: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the pixel-space Psi named by `kind`, reading its parameters from `cfg`."""
    if kind == "none":
        return lambda x: x
    if kind == "dct":
        return lambda x: dct_lowpass(x, cfg.purify_cutoff)
    if kind == "jpeg":
        return lambda x: jpeg_compress(x, cfg.jpeg_quality, differentiable=differentiable)
    if kind == "gaussian":
        return lambda x: gaussian_denoise_onestep(x, cfg.purify_sigma, cfg.purify_eot)
    raise ValueError(f"unknown purifier: {kind}")


def build_purifier(cfg, differentiable: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the Psi selected by `cfg.purify`, honouring its parameters. `none` is the identity."""
    return _psi_by_kind(getattr(cfg, "purify", "none"), cfg, differentiable)


def named_purifiers(cfg, differentiable: bool = False) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """The three Psi used by the E0 diagnostic, each built from `cfg`'s parameters."""
    return {name: _psi_by_kind(name, cfg, differentiable) for name in ("dct", "jpeg", "gaussian")}


def purifier_normalized(
    psi: Callable[[torch.Tensor], torch.Tensor], dm: torch.Tensor, ds: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Wrap a pixel-space Psi so it acts in the normalized space the models see.

    x_norm -> clamp to [0,1] in pixel space -> Psi -> back to normalized. Both conversions are affine,
    so the wrapper is differentiable and the brewing gradient flows straight through Psi.
    """

    def wrapped(x: torch.Tensor) -> torch.Tensor:
        pixel = (x * ds + dm).clamp(0.0, 1.0)
        return (psi(pixel) - dm) / ds

    return wrapped


def brew_purifier_normalized(cfg, dm: torch.Tensor, ds: torch.Tensor):
    """E2 brewing-time Psi in normalized space (differentiable / BPDA), or None when `brew_purify` off."""
    kind = getattr(cfg, "brew_purify", "none")
    if kind == "none":
        return None
    return purifier_normalized(_psi_by_kind(kind, cfg, differentiable=True), dm, ds)


def victim_purifier_normalized(cfg, dm: torch.Tensor, ds: torch.Tensor):
    """Victim-side defense Psi in normalized space (non-differentiable), or None when `purify` off."""
    kind = getattr(cfg, "purify", "none")
    if kind == "none":
        return None
    return purifier_normalized(_psi_by_kind(kind, cfg, differentiable=False), dm, ds)
