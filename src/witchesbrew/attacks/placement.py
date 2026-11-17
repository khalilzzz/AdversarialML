"""The attacker's side of piste 2: where to put the poisons, and how to put them there.

Two results of docs/theory/theorie_pistes.tex sec. 4 are implemented:

* **Placement lemma.** DPA's hash depends only on the stored sample. With the pixel-sum hash the
  attacker moves a poison to any partition by adding +1 (in 0..255 units) to `t < k` unsaturated
  pixels; with a *public* weighted hash, a +-1 search on single pixels reaches the wanted residue in
  about `k` trials. Both cost 1/255 in l_inf, which is reserved by brewing at `eps - margin`. Under a
  *secret* key the partition of a modified image is uniform and unknown to the attacker: no placement
  is possible, and this module refuses to try.
* **Allocation theorem.** The attacker chooses `m >= m_0` partitions and spreads the budget evenly
  over them (Jensen, concave regime). Which `m`: `random`, or `weakest` -- the partitions whose clean
  base model holds the target with the smallest logit margin, so fewer poisons flip them.

Everything here edits the perturbation *after* brewing; the gradient-matching loop is untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.datasets import denormalization_tensors
from ..defenses.dpa import _HASH_PRIME, hash_values, hash_weights


def choose_partitions(
    k: int,
    m: int,
    pick: str,
    rng: np.random.Generator,
    clean_margins: list[float] | None = None,
) -> list[int]:
    """The `m` partitions to attack.

    `clean_margins[i]` is logit[y_0] - logit[y_adv] of clean base model `i` on the target: the
    smallest margins are the cheapest to flip (`weakest`).
    """
    if m <= 0:
        return []
    if m > k:
        raise ValueError(f"dpa_attack_partitions={m} exceeds dpa_k={k}")
    if pick == "random":
        return sorted(int(i) for i in rng.choice(k, size=m, replace=False))
    if pick == "weakest":
        if clean_margins is None:
            raise ValueError("dpa_pick=weakest needs the clean base models' target margins")
        order = np.argsort(np.asarray(clean_margins, dtype=np.float64), kind="stable")
        return sorted(int(i) for i in order[:m])
    raise ValueError(f"unknown dpa_pick: {pick}")


def allocate(poison_ids: list[int], targets: list[int]) -> dict[int, int]:
    """Equal split B/m of the poisons over the chosen partitions (remainder to the first ones)."""
    if not targets:
        return {}
    out: dict[int, int] = {}
    for row, pid in enumerate(poison_ids):
        out[int(pid)] = targets[row % len(targets)]
    return out


def _pixel_views(delta_norm: torch.Tensor, clean_norm: torch.Tensor, dataset: str):
    dm, ds = denormalization_tensors(dataset, delta_norm.device)
    clean_pixel = (clean_norm * ds + dm).clamp(0.0, 1.0)
    poisoned_pixel = (clean_pixel + delta_norm * ds).clamp(0.0, 1.0)
    return clean_pixel, poisoned_pixel, dm, ds


@torch.no_grad()
def place_in_partition(
    delta_norm: torch.Tensor,
    clean_norm: torch.Tensor,
    dataset: str,
    assignment: dict[int, int],
    poison_ids: list[int],
    k: int,
    mode: str,
    key: int | None,
    eps: float,
    rng: np.random.Generator,
    max_trials: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Edit the brewed perturbation so each poison's stored image hashes into its assigned partition.

    Works on the 8-bit grid: the returned perturbation is `quantize(x + delta') - x`, so the victim
    trains on exactly the image whose hash was checked. Returns the new normalized delta and a
    report (shift per poison, l_inf of the final delta in 0..255 units, trials used).
    """
    if mode == "secret":
        raise ValueError(
            "dpa_hash=secret: the attacker cannot steer the partition (placement lemma (c)); "
            "set dpa_attack_partitions=0"
        )
    if not assignment:
        return delta_norm, {"placed": 0}
    clean_pixel, poisoned_pixel, dm, ds = _pixel_views(delta_norm, clean_norm, dataset)
    clean_u8 = torch.round(clean_pixel * 255.0).to(torch.int64)
    u8 = torch.round(poisoned_pixel * 255.0).to(torch.int64)  # the stored image, before placement
    eps_u8 = int(round(eps))
    numel = u8[0].numel()
    weights = hash_weights(numel, mode, key, u8.device)
    max_trials = max_trials or 50 * k

    shifts: list[int] = []
    trials_used: list[int] = []
    for row, pid in enumerate(poison_ids):
        want = assignment.get(int(pid))
        if want is None:
            shifts.append(0)
            trials_used.append(0)
            continue
        img = u8[row].reshape(-1).clone()
        base = clean_u8[row].reshape(-1)
        low = torch.clamp(base - eps_u8, min=0)
        high = torch.clamp(base + eps_u8, max=255)
        h = int(hash_values(img.view(1, -1), weights)[0])
        if h % k == want:
            shifts.append(0)
            trials_used.append(0)
        elif weights is None:
            # Pixel sum: add +1 to t distinct pixels that still have room (lemma (a)).
            t = (want - h) % k
            room = torch.nonzero(img < high).reshape(-1)
            if len(room) < t:
                raise RuntimeError(f"poison {pid}: only {len(room)} pixels with room, need {t}")
            chosen = room[torch.as_tensor(rng.choice(len(room), size=t, replace=False), device=img.device)]
            img[chosen] += 1
            shifts.append(t)
            trials_used.append(1)
        else:
            # Public weighted hash: +-1 on a random pixel, updated incrementally (lemma (b)).
            done = False
            for trial in range(1, max_trials + 1):
                j = int(rng.integers(numel))
                step = 1 if rng.random() < 0.5 else -1
                v = int(img[j]) + step
                if v < int(low[j]) or v > int(high[j]):
                    continue
                h_new = (h + step * int(weights[j])) % _HASH_PRIME
                if h_new % k == want:
                    img[j] = v
                    h = h_new
                    shifts.append(1)
                    trials_used.append(trial)
                    done = True
                    break
            if not done:
                raise RuntimeError(f"poison {pid}: no partition hit in {max_trials} trials")
        u8[row] = img.view_as(u8[row])

    # Verify on the final stored images, then return the perturbation that reproduces them exactly.
    final = hash_values(u8.reshape(u8.shape[0], -1), weights) % k
    for row, pid in enumerate(poison_ids):
        want = assignment.get(int(pid))
        if want is not None and int(final[row]) != want:
            raise AssertionError(f"poison {pid} hashed to {int(final[row])}, wanted {want}")
    new_delta_pixel = u8.to(clean_pixel.dtype) / 255.0 - clean_pixel
    linf_u8 = float(new_delta_pixel.abs().max()) * 255.0
    if linf_u8 > eps + 1e-3:
        raise AssertionError(f"placement broke the eps bound: {linf_u8:.2f} > {eps}")
    return new_delta_pixel / ds, {
        "placed": sum(1 for s in shifts if s > 0),
        "mean_shift_pixels": float(np.mean(shifts)) if shifts else 0.0,
        "mean_trials": float(np.mean(trials_used)) if trials_used else 0.0,
        "linf_final_255": linf_u8,
    }
