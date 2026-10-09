"""Differential privacy as a defense (section 5.3, Fig. 4b, appendix F.3).

The paper's finding: DP is "the only existing defense that can partially mitigate the effects of the
attack", and it works through the *gradient noise*, following Hong et al. (2020). Appendix F.3 fixes
gradient clipping at 1 and sweeps the noise, applied on the mini-batch level. The cost is severe:
"to push the Poison Success below 15%, one has to sacrifice over 20% validation accuracy".

Two pieces live here:

* `dp_sweep_configs` -- the noise values of the Fig. 4b x-axis.
* `make_gradient_hook` -- the attacker's counter-measure. Appendix F.3: "If the level of gradient
  noise is known to the attacker, then the attacker can brew poisoned data by the approach shown in
  algorithm 1 but also add gradient noise and gradient clipping to the poison gradient. We use a
  naive strategy of redrawing the added noise every time the matching objective is evaluated."
  The defense itself is applied during victim training by `train.apply_privacy_defense`.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch

# Read off the Fig. 4b x-axis (10u, 100u, 0.001, 0.01), plus the undefended point.
DP_NOISE_LEVELS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
DP_CLIP = 1.0  # appendix F.3 keeps this fixed


def dp_sweep_configs(cfg, attacker_aware: bool = False) -> list:
    """One configuration per point of the Fig. 4b sweep."""
    out = []
    for noise in DP_NOISE_LEVELS:
        clone = copy.deepcopy(cfg)
        clone.dp_clip = DP_CLIP
        clone.dp_noise = noise if noise > 0 else None
        clone.attacker_aware_dp = attacker_aware
        clone.name = f"{cfg.name}_noise{noise:g}" + ("_aware" if attacker_aware else "")
        out.append(clone)
    return out


def make_gradient_hook(clip: float | None, noise: float | None) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
    """Clip and re-noise the poison gradient inside the brewing objective.

    Kept differentiable: clipping rescales by a factor computed from the (detached) norm, so the
    perturbation still receives a gradient, and the noise is a constant offset redrawn per call.
    """

    def hook(grads: list[torch.Tensor]) -> list[torch.Tensor]:
        out = grads
        if clip is not None:
            total = torch.sqrt(sum(g.detach().pow(2).sum() for g in out))
            factor = (clip / (total + 1e-6)).clamp(max=1.0)
            out = [g * factor for g in out]
        if noise:
            scale = (clip if clip is not None else 1.0) * noise
            out = [g + torch.randn_like(g) * scale for g in out]
        return out

    return hook
