"""The custom-dataset path: an arbitrary ImageFolder, an explicit target, a seeded multi-case mode.

These exercise the generalization that lifts the attack off the paper's CIFAR-10 / ImageNet rails,
without needing a GPU or a real dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from witchesbrew.config import Config
from witchesbrew.data.augment import build_augmentation
from witchesbrew.data.datasets import build_datasets, compute_mean_std, num_classes
from witchesbrew.data.setup import build_case

CLASSES = ["alpha", "beta", "gamma", "delta"]
PER_CLASS = 6


def _toy_imagefolder(root, size=16):
    """Write a tiny ImageFolder with train/ and val/ splits and return its path."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for label, name in enumerate(CLASSES):
            d = root / split / name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(PER_CLASS):
                # Each class gets a distinct base colour so a model could, in principle, learn it.
                base = np.full((size, size, 3), label * 50, dtype=np.uint8)
                noise = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
                Image.fromarray((base + noise).astype(np.uint8)).save(d / f"{i}.png")
    return root


def _cfg(root, **kw):
    params = dict(
        name="toy", dataset="custom", data_path=str(root), image_size=16,
        budget=0.2, eps=16.0, net="ResNet18",
    )
    params.update(kw)
    return Config(**params)


def test_custom_imagefolder_loads_and_counts_classes(tmp_path):
    root = _toy_imagefolder(tmp_path / "ds")
    cfg = _cfg(root)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    assert len(trainset) == len(CLASSES) * PER_CLASS
    assert num_classes("custom") == len(CLASSES)
    image, label, index = trainset[0]
    assert image.shape == (3, 16, 16)
    assert isinstance(int(label), int) and index == 0


def test_computed_mean_std_are_in_range(tmp_path):
    root = _toy_imagefolder(tmp_path / "ds")
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
    mean, std = compute_mean_std(datasets.ImageFolder(str(root / "train"), transform=tf), channels=3)
    assert len(mean) == 3 and len(std) == 3
    assert all(0.0 <= m <= 1.0 for m in mean)
    assert all(s > 0.0 for s in std)


def test_explicit_case_uses_the_requested_target_and_class(tmp_path):
    root = _toy_imagefolder(tmp_path / "ds")
    cfg = _cfg(root, target_index=1, adv_class=2)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    case = build_case(0, cfg, trainset, validset)

    assert case.target_ids == [1]
    assert case.poison_class == 2  # == adv_class
    assert case.target_class == int(validset[1][1])  # the target's own true label
    # Clean-label: every poison really is an image of the adversarial class.
    train_labels = np.asarray(trainset.targets)
    assert all(train_labels[i] == 2 for i in case.poison_ids)
    assert len(case.poison_ids) >= 1


def test_poison_class_can_differ_from_adv_class(tmp_path):
    root = _toy_imagefolder(tmp_path / "ds")
    cfg = _cfg(root, target_index=0, adv_class=1, poison_class=3)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    case = build_case(0, cfg, trainset, validset)
    assert case.poison_class == 3
    train_labels = np.asarray(trainset.targets)
    assert all(train_labels[i] == 3 for i in case.poison_ids)


def test_seeded_random_case_is_deterministic(tmp_path):
    root = _toy_imagefolder(tmp_path / "ds")
    cfg = _cfg(root)  # no explicit target -> seeded random construction
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    a = build_case(0, cfg, trainset, validset)
    b = build_case(0, cfg, trainset, validset)
    assert (a.target_class, a.poison_class, a.target_ids, a.poison_ids) == (
        b.target_class, b.poison_class, b.target_ids, b.poison_ids,
    )
    assert a.target_class != a.poison_class


def test_augmentation_scales_with_resolution():
    aug = build_augmentation("custom", size=20, shift=3)
    assert aug.size == 20 and aug.shift == 3
    x = torch.rand(2, 3, 20, 20)
    zero = torch.zeros(2, 3)  # randgen=0.5 -> no shift, but 0 -> -shift; use identity check instead
    # At the grid centre (randgen=0.5) the transform is the identity.
    randgen = torch.full((2, 3), 0.5)
    out = aug(x, randgen=randgen)
    assert torch.allclose(out, x, atol=1e-5)


def test_shift_defaults_to_an_eighth_of_the_size():
    assert build_augmentation("custom", size=32).shift == 4
    assert build_augmentation("custom", size=224).shift == 28
