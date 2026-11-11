"""Augmentation.

Two users, one implementation:

* **Brewing** (paper section 3.4, Alg. 1 line 7). "At each step minimizing eq. (3), we randomly draw
  a translation, crop, and possibly a horizontal flip for each poisoned image, then use bilinear
  interpolation to resample to the original resolution. When updating Delta, we differentiate through
  this grid sampling operation as in Jaderberg et al. (2015)." That rules out the usual PIL-based
  transforms: the operation has to be differentiable w.r.t. the image, which `grid_sample` is.
* **Victim training** (appendix B.2): "horizontal flipping with probability 0.5 and random crops of
  size 32 x 32 with zero-padding of 4" (ImageNet: 224 x 224 with zero-padding of 28). Padding an
  image and then taking a crop of the original size is exactly a random translation with zeros
  shifted in at the border, which is what this module does -- so the victim's augmentation and the
  distribution the attack was optimized against are the same thing, and any gap between them would
  measure the implementation rather than the attack.

The translation budget therefore comes straight from the paper: +-4 pixels on CIFAR-10 and +-28 on
ImageNet. The `grid_sample` flags (`align_corners`, zero padding) are not specified in the paper and
follow the reference implementation; see DESIGN_DECISIONS.md section 4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RandomTransform(torch.nn.Module):
    """Differentiable random translation + optional horizontal flip, resampled bilinearly.

    Args:
        size: spatial resolution of the input (assumed square).
        shift: maximum translation in pixels, in either direction. 4 for CIFAR-10 and 28 for
            ImageNet, matching the zero-padding of the victim's random crop (appendix B.2).
        flip: allow horizontal flips (probability 0.5, as in appendix B.2).
        mode: interpolation mode for `grid_sample`; the paper specifies bilinear.
    """

    def __init__(self, size: int, shift: int = 4, flip: bool = True, mode: str = "bilinear") -> None:
        super().__init__()
        self.size = size
        self.shift = shift
        self.flip = flip
        self.mode = mode
        self.register_buffer("base_grid", self._identity_grid(size), persistent=False)
        # One pixel is 2/(size - 1) wide on the align_corners=True coordinate grid.
        self.delta = (2.0 * shift / (size - 1)) if size > 1 else 0.0

    @staticmethod
    def _identity_grid(size: int) -> torch.Tensor:
        """Sampling grid that reproduces the input exactly, shaped [1, size, size, 2]."""
        coords = torch.linspace(-1, 1, size)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    def random_grid(self, x: torch.Tensor, randgen: torch.Tensor | None = None) -> torch.Tensor:
        batch = x.shape[0]
        grid = self.base_grid.to(device=x.device, dtype=x.dtype).repeat(batch, 1, 1, 1).clone()
        if randgen is None:
            randgen = torch.rand(batch, 3, device=x.device, dtype=x.dtype)

        # Independent horizontal and vertical translation, each uniform in [-shift, +shift] pixels.
        offset_x = (randgen[:, 0] * 2 - 1) * self.delta
        offset_y = (randgen[:, 1] * 2 - 1) * self.delta
        grid[..., 0] = grid[..., 0] + offset_x.view(-1, 1, 1)
        grid[..., 1] = grid[..., 1] + offset_y.view(-1, 1, 1)

        if self.flip:
            flipped = randgen[:, 2] > 0.5
            grid[flipped, ..., 0] = -grid[flipped, ..., 0]
        return grid

    def forward(self, x: torch.Tensor, randgen: torch.Tensor | None = None) -> torch.Tensor:
        grid = self.random_grid(x, randgen)
        # padding_mode='zeros' is grid_sample's default and is what makes this a zero-padded crop.
        return F.grid_sample(x, grid, mode=self.mode, align_corners=True)


def build_augmentation(
    dataset: str,
    size: int | None = None,
    shift: int | None = None,
    flip: bool = True,
) -> RandomTransform:
    """Augmentation module.

    With no overrides this is the paper's recipe (appendix B.2): 32 px / +-4 on CIFAR-10, 224 / +-28
    otherwise. A custom dataset passes its own `size`; `shift` defaults to `round(size/8)`, which
    reproduces 4 at 32 and 28 at 224, so the translation budget scales with the resolution.
    """
    if size is None:
        size = 32 if dataset.upper() == "CIFAR10" else 224
    if shift is None:
        shift = max(1, round(size / 8))
    return RandomTransform(size=size, shift=shift, flip=flip)


def augmentation_for(cfg) -> RandomTransform:
    """Augmentation for a Config, honouring the custom-dataset resolution/shift fields."""
    return build_augmentation(
        cfg.dataset,
        size=getattr(cfg, "image_size", None),
        shift=getattr(cfg, "augment_shift", None),
        flip=getattr(cfg, "augment_flip", True),
    )
