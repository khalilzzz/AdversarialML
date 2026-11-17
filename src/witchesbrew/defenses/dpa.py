"""Deep Partition Aggregation (DPA) as a defense -- piste 2, research extension, NOT the paper.

Levine & Feizi (ICLR 2021): hash every training sample into one of `k` partitions with a
deterministic function of the sample itself, train one base classifier per partition, and predict by
majority vote. The certificate (their Thm. 1): with vote margin `g`, any `s < g/2` insertions or
deletions cannot change the prediction, because one sample sits in exactly one partition and can
therefore change at most one vote. Wang & Feizi (2023) show the empirical robustness is far larger
than the certificate and attribute it to *poisoning overfitting*, under the implicit assumption that
the poisons are spread uniformly over the partitions (H1 in docs/theory/theorie_pistes.tex, sec. 4).

This module is the defender's side:

* `partition_hash`     -- the three hash families of the placement lemma: pixel sum (the original
                          DPA implementation), a *public* weighted sum (Wang & Feizi Def. 4.1), and
                          the same weighted sum under a *secret* key (the only one the attacker cannot
                          steer).
* `partition_dataset`  -- partition of every training index, poisons hashed *after* their
                          perturbation is applied: the defender only ever sees the stored image.
* `train_dpa`          -- the `k` base models, each trained with the victim recipe on its partition.
* `aggregate_votes`    -- majority vote with ties broken toward the smaller class index, the vote
                          margin `g`, and the certified radius / `m_0` derived from it.

Hashes act on 8-bit pixel values (what a stored image is), never on the normalized float tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Subset

from ..data.datasets import denormalization_tensors, make_loader
from ..models.factory import build_model
from ..train import train_model
from ..utils import available_workers, set_seed

HASH_MODES = ("sum", "weighted", "secret")
# Large prime for the weighted hash (Wang & Feizi Def. 4.1 reduce modulo a prime before modulo k).
_HASH_PRIME = 2_147_483_647  # 2^31 - 1


def to_uint8(normalized: torch.Tensor, dataset: str) -> torch.Tensor:
    """Normalized float batch -> the 8-bit image the defender actually stores, as int64 [N, C, H, W]."""
    dm, ds = denormalization_tensors(dataset, normalized.device)
    pixel = (normalized * ds + dm).clamp(0.0, 1.0)
    return torch.round(pixel * 255.0).to(torch.int64)


def hash_weights(numel: int, mode: str, key: int | None, device) -> torch.Tensor | None:
    """The weight vector `W` of the hash, or None for the plain pixel sum."""
    if mode == "sum":
        return None
    if mode not in HASH_MODES:
        raise ValueError(f"unknown dpa_hash: {mode}")
    if key is None:
        raise ValueError(f"dpa_hash={mode} needs a key (dpa_hash_key or dpa_secret_key)")
    gen = torch.Generator(device="cpu").manual_seed(int(key))
    w = torch.randint(1, _HASH_PRIME, (numel,), generator=gen, dtype=torch.int64)
    return w.to(device)


def hash_values(images_uint8: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Raw hash value per image (before `mod k`): pixel sum, or weighted sum modulo the prime."""
    flat = images_uint8.reshape(images_uint8.shape[0], -1)
    if weights is None:
        return flat.sum(dim=1)
    # Products are < 2^39 and the sum over <= 2^18 pixels stays < 2^57: no int64 overflow.
    return (flat * weights.view(1, -1)).sum(dim=1) % _HASH_PRIME


def partition_hash(
    images_uint8: torch.Tensor, k: int, mode: str = "sum", key: int | None = None
) -> torch.Tensor:
    """Partition index in [0, k) of each 8-bit image, for the chosen hash family."""
    weights = hash_weights(images_uint8[0].numel(), mode, key, images_uint8.device)
    return hash_values(images_uint8, weights) % k


def hash_key_for(cfg) -> int | None:
    """Which key the *defender* uses: the public one for `weighted`, the secret one for `secret`."""
    if cfg.dpa_hash == "sum":
        return None
    if cfg.dpa_hash == "weighted":
        return int(cfg.dpa_hash_key)
    if cfg.dpa_secret_key is None:
        raise ValueError("dpa_hash=secret requires dpa_secret_key")
    return int(cfg.dpa_secret_key)


@torch.no_grad()
def partition_dataset(
    trainset,
    indices: list[int],
    k: int,
    mode: str,
    key: int | None,
    dataset: str,
    device: torch.device,
    delta: torch.Tensor | None = None,
    poison_ids: list[int] | None = None,
    workers: int = 0,
    batch_size: int = 256,
) -> list[list[int]]:
    """Assign every index in `indices` to a partition; poisoned images are hashed with delta applied.

    Returns `k` lists of dataset indices. The hash is computed on the quantized stored image, so a
    poison lands exactly where the attacker's placement (attacks/placement.py) computed it would.
    """
    lookup = {int(pid): row for row, pid in enumerate(poison_ids or [])}
    delta = delta.to(device) if delta is not None else None
    loader = make_loader(Subset(trainset, list(indices)), batch_size, shuffle=False, workers=workers)
    parts: list[list[int]] = [[] for _ in range(k)]
    weights = None
    for images, _, idx in loader:
        images = images.to(device)
        rows = [lookup.get(int(i), -1) for i in idx]
        if delta is not None and any(r >= 0 for r in rows):
            mask = torch.tensor([r >= 0 for r in rows], device=device)
            sel = torch.tensor([r for r in rows if r >= 0], device=device)
            images = images.clone()
            images[mask] = images[mask] + delta[sel]
        u8 = to_uint8(images, dataset)
        if weights is None and mode != "sum":
            weights = hash_weights(u8[0].numel(), mode, key, device)
        h = hash_values(u8, weights) % k
        for i, p in zip(idx, h.tolist()):
            parts[p].append(int(i))
    return parts


@dataclass
class BaseModelResult:
    partition: int
    size: int
    poisons: int
    target_pred: int
    target_logits: list[float]
    target_adv_loss: float
    valid_acc: float
    train_acc: float
    history: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DPAResult:
    base: list[BaseModelResult]
    votes: list[int]
    prediction: int
    margin: int
    certified_radius: int
    m0: int
    valid_acc: float  # accuracy of the aggregate (majority vote) on the validation set

    def to_dict(self) -> dict:
        return {
            "base": [b.to_dict() for b in self.base],
            "votes": self.votes,
            "prediction": self.prediction,
            "margin": self.margin,
            "certified_radius": self.certified_radius,
            "m0": self.m0,
            "valid_acc": self.valid_acc,
        }


def aggregate_votes(predictions: list[int], num_classes: int) -> tuple[int, list[int], int]:
    """Majority vote (ties -> smaller class index), and the margin `g` of Thm. 2.1 / Thm. 1.

    g = V_{y0} - max_{y != y0} (V_y + 1[y < y0]): the number of votes that would have to move for
    another class to win under the same tie rule.
    """
    votes = [0] * num_classes
    for p in predictions:
        votes[int(p)] += 1
    winner = max(range(num_classes), key=lambda y: (votes[y], -y))
    rival = max(votes[y] + (1 if y < winner else 0) for y in range(num_classes) if y != winner)
    return winner, votes, votes[winner] - rival


def certified_radius(margin: int) -> int:
    """Largest `s` with `s < g/2`: the number of poisons the certificate tolerates."""
    return max(0, math.ceil(margin / 2) - 1)


def flips_needed(margin: int) -> int:
    """m_0 = floor(g/2) + 1 base models must switch to y_adv for the vote to turn."""
    return max(1, margin // 2 + 1)


@torch.no_grad()
def _predict(model: torch.nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, labels = [], []
    for inputs, targets, _ in loader:
        preds.append(model(inputs.to(device)).argmax(dim=1).cpu())
        labels.append(targets)
    return torch.cat(preds), torch.cat(labels)


def train_dpa(
    cfg,
    trainset,
    validset,
    partitions: list[list[int]],
    device: torch.device,
    poison_delta: torch.Tensor | None,
    poison_ids: list[int] | None,
    target_inputs: torch.Tensor,
    adv_labels: torch.Tensor,
    true_labels: torch.Tensor,
    target_ids: list[int],
    augment,
    seed: int,
    num_classes: int,
    keep_history: bool = False,
) -> DPAResult:
    """Train one base model per partition with the victim recipe and vote on the target.

    Base model `i` is seeded with `seed + i`, so a clean reference run and a poisoned run that share
    `seed` differ only through the poisons each partition received.
    """
    # The partition *is* the training set: no further ablation on top of it.
    part_cfg = type(cfg)(**{**cfg.to_dict(), "data_ablation": 1.0})
    poison_set = set(int(p) for p in (poison_ids or []))
    workers = available_workers(cfg.workers)
    valid_loader = make_loader(validset, 256, shuffle=False, workers=workers)
    exclude = {int(i) for i in target_ids}

    base: list[BaseModelResult] = []
    vote_counts = None
    valid_labels = None
    for i, ids in enumerate(partitions):
        set_seed(seed + i)
        model = build_model(cfg.victim_net, cfg.dataset, num_classes, cfg.width).to(device)
        history = train_model(
            part_cfg,
            model,
            Subset(trainset, ids),
            validset,
            device,
            poison_delta=poison_delta,
            poison_ids=poison_ids,
            target_inputs=target_inputs,
            adv_labels=adv_labels,
            true_labels=true_labels,
            target_ids=target_ids,
            augment=augment,
            seed=seed + i,
        )
        model.eval()
        with torch.no_grad():
            logits = model(target_inputs)[0]
        preds, labels = _predict(model, valid_loader, device)
        if vote_counts is None:
            vote_counts = torch.zeros(len(labels), num_classes, dtype=torch.int64)
            valid_labels = labels
        vote_counts[torch.arange(len(preds)), preds] += 1
        base.append(
            BaseModelResult(
                partition=i,
                size=len(ids),
                poisons=sum(1 for j in ids if j in poison_set),
                target_pred=int(logits.argmax()),
                target_logits=[float(v) for v in logits],
                target_adv_loss=history.target_adv_loss[-1] if history.target_adv_loss else float("nan"),
                valid_acc=history.valid_acc[-1],
                train_acc=history.train_acc[-1],
                history=history.to_dict() if keep_history else {},
            )
        )

    prediction, votes, margin = aggregate_votes([b.target_pred for b in base], num_classes)
    # Aggregate validation accuracy, target excluded as in appendix F.9. Tie rule: smaller class.
    assert vote_counts is not None and valid_labels is not None
    agg_pred = torch.flip(vote_counts, dims=[1]).argmax(dim=1)
    agg_pred = num_classes - 1 - agg_pred
    keep = torch.tensor([int(i) not in exclude for i in range(len(valid_labels))])
    valid_acc = float((agg_pred[keep] == valid_labels[keep]).float().mean())
    return DPAResult(
        base=base,
        votes=votes,
        prediction=prediction,
        margin=margin,
        certified_radius=certified_radius(margin),
        m0=flips_needed(margin),
        valid_acc=valid_acc,
    )


def partition_sizes(partitions: list[list[int]]) -> list[int]:
    return [len(p) for p in partitions]


def natural_partition_of(poison_ids: list[int], partitions: list[list[int]]) -> dict[int, int]:
    """Map each poison id to the partition its (final) image hashes into."""
    where: dict[int, int] = {}
    wanted = set(int(p) for p in poison_ids)
    for p, ids in enumerate(partitions):
        for i in ids:
            if i in wanted:
                where[int(i)] = p
    return where


def as_numpy(partitions: list[list[int]]) -> list[np.ndarray]:
    return [np.asarray(p, dtype=np.int64) for p in partitions]
