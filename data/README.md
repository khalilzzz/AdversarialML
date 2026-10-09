# Data

## CIFAR-10

Downloaded automatically into `data/raw/` on first use (`torchvision.datasets.CIFAR10`), ~170 MB.

Appendix B indexes targets by "the canonical order of all images in the dataset (as downloaded from
`https://www.cs.toronto.edu/~kriz/cifar.html`)", which is the order torchvision preserves, so the
published target IDs in `consts.CIFAR10_CASES` line up without any reordering. The normalization
constants used are in `src/witchesbrew/consts.py`.

## ImageNet (ILSVRC2012)

Not downloadable without an account, so lay it out by hand:

```
data/
  imagenet/
    train/
      n01440764/*.JPEG
      n01443537/*.JPEG
      ...
    val/
      n01440764/*.JPEG
      ...
```

~150 GB. The validation set must be split into per-synset folders (the standard
`valprep.sh` reorganization) so that `torchvision.datasets.ImageFolder` can read it.

Appendix B indexes ImageNet targets by the validation set "where the syn-sets are ordered by their
increasing numerical value (as is the default in torchvision)". `ImageFolder` sorts directory names
alphabetically, which for `nXXXXXXXX` names is the same order, so the Table 5 IDs apply — but the
Table 5 class names are human-readable synset names rather than wnids, so `data/setup.py` cannot
match them against the folder names and falls back to drawing the class pair from the published
seed. That is recorded in `docs/DEVIATIONS.md`. To use the published class pairs instead, add a
`wnid -> name` mapping and resolve the names before `build_case` runs.

The deterministic part of the ImageNet pipeline (resize 256, central crop 224) is in
`data/datasets.py`; the random part (flip, zero-padded 224 crop) is the differentiable augmentation
module, so it is identical for the attack and the victim.

## What gets written

`poisons/`, `results/` and `checkpoints/` are created on demand and git-ignored:

* `checkpoints/` — cached clean surrogate models, one per `(architecture, width, pretrain epochs, seed)`.
* `poisons/` — exported perturbations. Appendix D ships "only the subset of poisoned images and not
  the full dataset"; `scripts/brew_case.py` does the same, in `packed`, `png` or `imagefolder` form.
* `results/<experiment>/caseNN.json` — one file per case, which is what makes long sweeps resumable.
