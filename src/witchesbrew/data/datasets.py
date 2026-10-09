"""Datasets and loaders.

Two things this module has to provide that a plain torchvision dataset does not:

1. Every sample carries its **index**, so a perturbation can be looked up by training-set position
   (the paper's threat model perturbs specific images, and Appendix B indexes targets by their
   canonical dataset position).
2. Augmentation is applied *after* the perturbation is added, because the victim augments the
   poisoned images it receives (paper appendix B.2, and the reason the attack has to survive
   augmentation at all, section 3.4).

So the tensor pipeline is: ToTensor + Normalize in the dataset, and random crop/flip applied on
batches inside the training loop.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..consts import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD


class IndexedDataset(Dataset):
    """Wrap a dataset so it yields `(image, label, index)`."""

    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, label, index

    @property
    def targets(self) -> list[int]:
        inner = self.base
        if hasattr(inner, "targets"):
            return [int(t) for t in inner.targets]
        if hasattr(inner, "samples"):
            return [int(s[1]) for s in inner.samples]
        raise AttributeError("wrapped dataset exposes no labels")

    @property
    def classes(self) -> list[str]:
        return list(getattr(self.base, "classes"))


def normalization(dataset: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset.upper() == "CIFAR10":
        return CIFAR10_MEAN, CIFAR10_STD
    return IMAGENET_MEAN, IMAGENET_STD


def denormalization_tensors(dataset: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std shaped [1, C, 1, 1] for converting between pixel and normalized space."""
    mean, std = normalization(dataset)
    dm = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    ds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return dm, ds


def image_size(dataset: str) -> int:
    return 32 if dataset.upper() == "CIFAR10" else 224


def build_datasets(dataset: str, data_path: str) -> tuple[IndexedDataset, IndexedDataset]:
    """Return `(trainset, validset)` with normalization only -- no random augmentation.

    CIFAR-10 is downloaded automatically. ImageNet (ILSVRC2012) must be laid out by hand as
    `<data_path>/imagenet/{train,val}/<wnid>/*.JPEG`; see data/README.md. Appendix B.2 fixes the
    deterministic part of the ImageNet pipeline as resize 256 then central crop 224.
    """
    root = Path(data_path).expanduser()
    name = dataset.upper()

    if name == "CIFAR10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train = datasets.CIFAR10(str(root), train=True, download=True, transform=tf)
        valid = datasets.CIFAR10(str(root), train=False, download=True, transform=tf)
        return IndexedDataset(train), IndexedDataset(valid)

    if name in {"IMAGENET", "IMAGENET1K", "ILSVRC2012"}:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        tf = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        base = root / "imagenet"
        train = datasets.ImageFolder(str(base / "train"), transform=tf)
        valid = datasets.ImageFolder(str(base / "val"), transform=tf)
        return IndexedDataset(train), IndexedDataset(valid)

    raise ValueError(f"unsupported dataset: {dataset}")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=workers > 0,
    )


def num_classes(dataset: str) -> int:
    return 10 if dataset.upper() == "CIFAR10" else 1000
