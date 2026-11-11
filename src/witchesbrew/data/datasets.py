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

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ..consts import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD

# Resolved description of the active custom dataset, filled by `build_datasets` when
# `dataset == "custom"`. It lets the string-keyed helpers below (normalization, image_size, ...)
# serve the custom dataset without every call site having to thread the whole Config through -- the
# same way CIFAR10_MEAN etc. are looked up by name. There is one custom dataset per run.
_CUSTOM: dict = {}


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
    name = dataset.upper()
    if name == "CIFAR10":
        return CIFAR10_MEAN, CIFAR10_STD
    if name == "CUSTOM":
        return _require_custom("mean", "std")
    return IMAGENET_MEAN, IMAGENET_STD


def denormalization_tensors(dataset: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std shaped [1, C, 1, 1] for converting between pixel and normalized space."""
    mean, std = normalization(dataset)
    dm = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    ds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return dm, ds


def image_size(dataset: str) -> int:
    name = dataset.upper()
    if name == "CIFAR10":
        return 32
    if name == "CUSTOM":
        return int(_require_custom("size")[0])
    return 224


def _require_custom(*keys: str) -> tuple:
    """Fetch resolved custom-dataset fields, or explain that they are not resolved yet."""
    if not _CUSTOM:
        raise RuntimeError(
            "custom dataset not resolved; build_datasets(cfg.dataset, cfg.data_path, cfg) must run "
            "before normalization / image_size / num_classes are queried for a custom dataset"
        )
    return tuple(_CUSTOM[k] for k in keys)


def compute_mean_std(folder: datasets.ImageFolder, channels: int = 3) -> tuple[list[float], list[float]]:
    """Per-channel mean and std over an ImageFolder (ToTensor only, values in [0, 1])."""
    loader = DataLoader(folder, batch_size=64, shuffle=False, num_workers=0)
    total = 0
    mean = torch.zeros(channels)
    sq = torch.zeros(channels)
    for images, _ in loader:
        n = images.shape[0]
        flat = images.view(n, channels, -1)
        mean += flat.mean(dim=2).sum(dim=0)
        sq += flat.pow(2).mean(dim=2).sum(dim=0)
        total += n
    mean /= total
    std = (sq / total - mean.pow(2)).clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()


def _resolve_custom_stats(cfg, train_root: Path, size: int) -> tuple[list[float], list[float]]:
    """Mean/std from the config, else from a cache file, else computed once and cached."""
    if cfg.data_mean is not None and cfg.data_std is not None:
        return list(cfg.data_mean), list(cfg.data_std)
    cache = Path(cfg.data_path).expanduser() / "custom_stats.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["mean"], d["std"]
    tf = transforms.Compose(
        [transforms.Resize((size, size)), transforms.CenterCrop(size), transforms.ToTensor()]
    )
    mean, std = compute_mean_std(datasets.ImageFolder(str(train_root), transform=tf), cfg.channels)
    try:
        cache.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only data dir is fine; stats are just recomputed next run
    return mean, std


def build_datasets(dataset: str, data_path: str, cfg=None) -> tuple[IndexedDataset, IndexedDataset]:
    """Return `(trainset, validset)` with normalization only -- no random augmentation.

    CIFAR-10 is downloaded automatically. ImageNet (ILSVRC2012) must be laid out by hand as
    `<data_path>/imagenet/{train,val}/<wnid>/*.JPEG`; see data/README.md. Appendix B.2 fixes the
    deterministic part of the ImageNet pipeline as resize 256 then central crop 224.

    `dataset == "custom"` reads an arbitrary ImageFolder at `<data_path>/{train,val}`; `cfg` then
    supplies the resolution and (optionally) the normalization, and the resolved description is
    registered so the string-keyed helpers can serve it.
    """
    root = Path(data_path).expanduser()
    name = dataset.upper()

    if name == "CUSTOM":
        if cfg is None:
            raise ValueError("build_datasets(..., cfg) is required for a custom dataset")
        size = cfg.image_size or 32
        train_root, val_root = root / "train", root / "val"
        mean, std = _resolve_custom_stats(cfg, train_root, size)
        tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train = datasets.ImageFolder(str(train_root), transform=tf)
        valid = datasets.ImageFolder(str(val_root), transform=tf)
        _CUSTOM.clear()
        _CUSTOM.update(
            mean=mean,
            std=std,
            size=size,
            channels=cfg.channels,
            classes=cfg.num_classes_override or len(train.classes),
        )
        return IndexedDataset(train), IndexedDataset(valid)

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
    name = dataset.upper()
    if name == "CIFAR10":
        return 10
    if name == "CUSTOM":
        return int(_require_custom("classes")[0])
    return 1000
