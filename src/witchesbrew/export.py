"""Writing poisoned data out.

Appendix D of the paper describes exactly this: the authors ship "only the subset of poisoned images
and not the full dataset", noting that the separation "also aids in the development of defensive
strategies", and that the IDs refer to the dataset ordering discussed in appendix B.

Three formats:

* `packed`      -- `(delta, poison_ids)` as a `.pth`, the compact form appendix D describes.
* `png`         -- the poisoned images as 8-bit PNGs, which is what a victim would actually ingest;
                   going through uint8 also verifies that the perturbation survives quantization.
* `imagefolder` -- a torchvision `ImageFolder` tree of the poisoned subset, for feeding an external
                   training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image

from .data.datasets import denormalization_tensors


def to_pixel_space(
    normalized: torch.Tensor, dataset: str, device: torch.device | None = None
) -> torch.Tensor:
    """Undo normalization and clamp into [0, 1]."""
    device = device or normalized.device
    dm, ds = denormalization_tensors(dataset, device)
    return torch.clamp(normalized.to(device) * ds + dm, 0, 1)


def quantize(images: torch.Tensor) -> torch.Tensor:
    """Round to the 8-bit grid an exported image would land on."""
    return torch.round(images * 255) / 255


def export_packed(delta: torch.Tensor, poison_ids: list[int], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta": delta.cpu(), "poison_ids": list(poison_ids)}, out)
    return out


def load_packed(path: str | Path) -> tuple[torch.Tensor, list[int]]:
    blob = torch.load(Path(path), map_location="cpu")
    return blob["delta"], blob["poison_ids"]


def export_png(
    clean: torch.Tensor,
    delta: torch.Tensor,
    poison_ids: list[int],
    dataset: str,
    directory: str | Path,
    also_clean: bool = True,
) -> Path:
    """Write poisoned (and optionally clean) images, named by their dataset index."""
    out = Path(directory)
    (out / "poisoned").mkdir(parents=True, exist_ok=True)
    if also_clean:
        (out / "clean").mkdir(parents=True, exist_ok=True)
    poisoned = to_pixel_space(clean + delta.to(clean.device), dataset)
    originals = to_pixel_space(clean, dataset)
    for row, pid in enumerate(poison_ids):
        save_image(poisoned[row], out / "poisoned" / f"{pid:06d}.png")
        if also_clean:
            save_image(originals[row], out / "clean" / f"{pid:06d}.png")
    return out


def export_imagefolder(
    clean: torch.Tensor,
    delta: torch.Tensor,
    labels: torch.Tensor,
    poison_ids: list[int],
    class_names: list[str],
    dataset: str,
    directory: str | Path,
) -> Path:
    out = Path(directory)
    poisoned = to_pixel_space(clean + delta.to(clean.device), dataset)
    for row, pid in enumerate(poison_ids):
        class_dir = out / class_names[int(labels[row])]
        class_dir.mkdir(parents=True, exist_ok=True)
        save_image(poisoned[row], class_dir / f"{pid:06d}.png")
    return out


def quantization_error(
    clean: torch.Tensor, delta: torch.Tensor, dataset: str
) -> dict[str, float]:
    """How much the l_inf perturbation changes when written as 8-bit pixels.

    Worth checking, because the attack's entire premise is that the perturbation stays within the
    stated bound in *pixel* units (section 3.1).
    """
    exact = to_pixel_space(clean + delta.to(clean.device), dataset)
    stored = quantize(exact)
    base = to_pixel_space(clean, dataset)
    return {
        "max_abs_delta_pixels": float((exact - base).abs().max() * 255),
        "max_abs_delta_pixels_quantized": float((stored - base).abs().max() * 255),
        "max_quantization_shift_pixels": float((stored - exact).abs().max() * 255),
    }
