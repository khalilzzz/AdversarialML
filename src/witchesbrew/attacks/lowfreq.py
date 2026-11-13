"""WB-LF: constrain the perturbation to the low-frequency subspace (E1, research extension).

The linear Proposition of piste 1 says the budget an attacker spends outside the preserved subspace
S is wasted, so the optimal attacker solves over `delta in S cap {||delta||_inf <= eps}`. That
intersection is reached by alternating projections (Dykstra): project onto S (a low-pass DCT
truncation), then onto the l_inf box and valid pixel range, and repeat. A few sweeps suffice because
both sets are convex.

This is a pure projection -- it replaces `PoisonBounds.project` inside the brewing loop when
`cfg.delta_param == "lowfreq"`, and leaves every other part of Algorithm 1 untouched.
"""

from __future__ import annotations

import torch

from ..defenses.purification import dct_lowpass


def lowpass_normalized(delta_norm: torch.Tensor, ds: torch.Tensor, cutoff: int) -> torch.Tensor:
    """Low-pass a normalized perturbation: scale to pixel space, DCT-truncate, scale back."""
    return dct_lowpass(delta_norm * ds, cutoff) / ds


def subspace_project(
    delta_norm: torch.Tensor,
    clean_norm: torch.Tensor,
    bounds,
    cutoff: int,
    iters: int = 3,
) -> torch.Tensor:
    """Project onto S (low-frequency) intersect {||delta||_inf <= eps} intersect valid pixel range."""
    out = delta_norm
    for _ in range(max(1, iters)):
        out = lowpass_normalized(out, bounds.ds, cutoff)
        out = bounds.project(out, clean_norm)
    return out
