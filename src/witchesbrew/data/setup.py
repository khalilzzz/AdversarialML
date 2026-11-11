"""Poison/target case construction.

A *case* is one row of the paper's Table 4 / Table 5: a target class, a poison class (= the
adversarial label `y_adv`), one target image from the validation set, and the set of training images
to perturb. Appendix B explains why this matters: poisoning is volatile with respect to the
class pair and the specific target, so "average" performance only means something over a fixed,
published collection of cases.

What the paper pins down and what it does not:

* Target class, poison class, target ID and the seed are published (Tables 4 and 5), so they are
  hard-coded in `consts.py` and used verbatim.
* The *poison image IDs* are not published. They are drawn from the poison class with the case seed.
  The reference implementation samples them without replacement from that class, which is what is
  done here. The exact subset therefore cannot match the paper's, and that is noted in DEVIATIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..consts import CIFAR10_CASES, IMAGENET_CASES
from .datasets import IndexedDataset


@dataclass
class PoisonCase:
    """One poison-target setting."""

    case_id: int
    seed: int
    target_class: int
    poison_class: int  # y_adv -- also the class the poisoned images are drawn from
    target_ids: list[int]  # indices into the validation set
    poison_ids: list[int]  # indices into the training set
    target_class_name: str = ""
    poison_class_name: str = ""

    @property
    def budget_images(self) -> int:
        return len(self.poison_ids)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "target_class": self.target_class,
            "target_class_name": self.target_class_name,
            "poison_class": self.poison_class,
            "poison_class_name": self.poison_class_name,
            "target_ids": self.target_ids,
            "poison_ids": self.poison_ids,
        }


def num_poisons(budget: float, train_size: int) -> int:
    """`P` in eq. 1. The paper gives the budget as a fraction; ceil follows the reference code."""
    return int(np.ceil(budget * train_size))


def _class_index(dataset: IndexedDataset, name: str) -> int | None:
    classes = [c.lower() for c in dataset.classes]
    needle = name.lower()
    if needle in classes:
        return classes.index(needle)
    # ImageNet class names in Table 5 are the human-readable synset names; ImageFolder exposes
    # wnids instead, so a name match is not generally possible. The caller falls back to the seed.
    for i, c in enumerate(classes):
        if needle in c:
            return i
    return None


def _published_cases(dataset: str):
    return CIFAR10_CASES if dataset.upper() == "CIFAR10" else IMAGENET_CASES


def build_case(
    case_id: int,
    cfg,
    trainset: IndexedDataset,
    validset: IndexedDataset,
) -> PoisonCase:
    """Build case `case_id` for the configured dataset.

    Uses the published class pair and target ID when the class names can be resolved against the
    dataset (always true for CIFAR-10). Otherwise everything is drawn from the published seed, which
    is the reference implementation's `random_construction`.
    """
    # Path 1: an explicit target handed in by the config -- total control, one precise case, no
    # published table consulted. This is what makes the attack usable on an arbitrary dataset.
    if cfg.target_index is not None and cfg.adv_class is not None:
        return _explicit_case(case_id, cfg, trainset, validset)

    # Path 2: a custom dataset with no explicit target -- draw the class pair and the target from the
    # seed, exactly as the reference implementation's `random_construction`, with no table.
    if cfg.dataset.lower() == "custom":
        return _random_case(case_id, cfg, trainset, validset)

    table = _published_cases(cfg.dataset)
    if case_id >= len(table):
        raise IndexError(f"case {case_id} out of range; the paper publishes {len(table)} cases")
    target_name, poison_name, target_id, seed = table[case_id]

    rng = np.random.default_rng(seed % (2**32 - 1))
    train_labels = np.asarray(trainset.targets)
    valid_labels = np.asarray(validset.targets)

    target_class = _class_index(validset, target_name)
    poison_class = _class_index(trainset, poison_name)
    resolved = target_class is not None and poison_class is not None
    if not resolved:
        # Fall back to drawing the class pair from the seed (reference `random_construction`).
        classes = len(set(train_labels.tolist()))
        target_class, poison_class = rng.choice(classes, size=2, replace=False).tolist()

    # Target images: the published ID when it is usable, otherwise sampled from the target class.
    pool = np.flatnonzero(valid_labels == target_class)
    if resolved and target_id < len(validset) and valid_labels[target_id] == target_class:
        target_ids = [int(target_id)]
    else:
        target_ids = [int(rng.choice(pool))]
    if cfg.targets > 1:
        # Appendix F.8 poisons several targets at once; the extra ones are drawn from the same
        # class, excluding the one already chosen so the count is always exactly `targets`.
        remaining = np.setdiff1d(pool, np.asarray(target_ids))
        wanted_targets = min(cfg.targets - 1, len(remaining))
        if wanted_targets < cfg.targets - 1:
            raise ValueError(
                f"target class has only {len(pool)} validation images, cannot pick {cfg.targets} targets"
            )
        extra = rng.choice(remaining, size=wanted_targets, replace=False)
        target_ids = target_ids + sorted(int(i) for i in extra)

    # Poison images: drawn from the adversarial class (Alg. 1 line 3).
    candidates = np.flatnonzero(train_labels == poison_class)
    wanted = num_poisons(cfg.budget, len(trainset))
    if wanted > len(candidates):
        wanted = len(candidates)
    poison_ids = sorted(int(i) for i in rng.choice(candidates, size=wanted, replace=False))

    return PoisonCase(
        case_id=case_id,
        seed=int(seed),
        target_class=int(target_class),
        poison_class=int(poison_class),
        target_ids=target_ids,
        poison_ids=poison_ids,
        target_class_name=target_name,
        poison_class_name=poison_name,
    )


def _draw_poisons(rng, train_labels: np.ndarray, poison_class: int, budget: float, train_size: int) -> list[int]:
    """Poison IDs: `ceil(budget*N)` images of the adversarial class (Alg. 1 line 3)."""
    candidates = np.flatnonzero(train_labels == poison_class)
    wanted = min(num_poisons(budget, train_size), len(candidates))
    if wanted == 0:
        raise ValueError(f"class {poison_class} has no training images to poison")
    return sorted(int(i) for i in rng.choice(candidates, size=wanted, replace=False))


def _class_name(dataset: IndexedDataset, idx: int) -> str:
    names = getattr(dataset, "classes", [])
    return names[idx] if 0 <= idx < len(names) else str(idx)


def _explicit_case(case_id, cfg, trainset, validset) -> PoisonCase:
    """Build exactly the case the config asks for: target image, y_adv, poison class."""
    valid_labels = np.asarray(validset.targets)
    train_labels = np.asarray(trainset.targets)
    target_index = int(cfg.target_index)
    if not 0 <= target_index < len(validset):
        raise IndexError(f"target_index {target_index} out of range for {len(validset)} validation images")
    poison_class = int(cfg.poison_class if cfg.poison_class is not None else cfg.adv_class)
    seed = int(cfg.target_index) * 1_000_003 + int(cfg.adv_class)
    rng = np.random.default_rng(seed % (2**32 - 1))

    target_ids = [target_index]
    if cfg.targets > 1:
        # Extra targets share the original's true class, as in appendix F.8.
        pool = np.flatnonzero(valid_labels == valid_labels[target_index])
        remaining = np.setdiff1d(pool, np.asarray(target_ids))
        if len(remaining) < cfg.targets - 1:
            raise ValueError(f"target class has too few validation images for {cfg.targets} targets")
        extra = rng.choice(remaining, size=cfg.targets - 1, replace=False)
        target_ids = target_ids + sorted(int(i) for i in extra)

    poison_ids = _draw_poisons(rng, train_labels, poison_class, cfg.budget, len(trainset))
    return PoisonCase(
        case_id=case_id,
        seed=seed,
        target_class=int(valid_labels[target_index]),
        poison_class=poison_class,
        target_ids=target_ids,
        poison_ids=poison_ids,
        target_class_name=_class_name(validset, int(valid_labels[target_index])),
        poison_class_name=_class_name(trainset, poison_class),
    )


def _random_case(case_id, cfg, trainset, validset) -> PoisonCase:
    """Draw a class pair and a target from the case seed (reference `random_construction`)."""
    seed = (abs(hash(cfg.name)) % (2**31)) + case_id
    rng = np.random.default_rng(seed % (2**32 - 1))
    valid_labels = np.asarray(validset.targets)
    train_labels = np.asarray(trainset.targets)
    classes = len(set(train_labels.tolist()))
    if classes < 2:
        raise ValueError("need at least two classes to pick a target/poison pair")
    target_class, poison_class = (int(c) for c in rng.choice(classes, size=2, replace=False))

    pool = np.flatnonzero(valid_labels == target_class)
    if len(pool) < cfg.targets:
        raise ValueError(f"target class {target_class} has only {len(pool)} validation images")
    target_ids = sorted(int(i) for i in rng.choice(pool, size=cfg.targets, replace=False))
    poison_ids = _draw_poisons(rng, train_labels, poison_class, cfg.budget, len(trainset))
    return PoisonCase(
        case_id=case_id,
        seed=int(seed),
        target_class=target_class,
        poison_class=poison_class,
        target_ids=target_ids,
        poison_ids=poison_ids,
        target_class_name=_class_name(validset, target_class),
        poison_class_name=_class_name(trainset, poison_class),
    )


def published_case_count(dataset: str) -> int:
    return len(_published_cases(dataset))
