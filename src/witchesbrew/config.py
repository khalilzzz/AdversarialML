"""Experiment configuration.

One flat dataclass, loaded from YAML, with `base.yaml` as the default layer. Every field's default
is the paper's value where the paper gives one; see docs/DESIGN_DECISIONS.md for the per-field
source. Keeping the configuration flat and serializable means each result file can embed the exact
configuration that produced it.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- identification -------------------------------------------------
    name: str = "unnamed"
    notes: str = ""

    # --- data -----------------------------------------------------------
    dataset: str = "CIFAR10"  # CIFAR10 | ImageNet | custom
    data_path: str = "data/raw"

    # --- custom dataset (used only when dataset == "custom") ------------
    # An arbitrary ImageFolder with <data_path>/{train,val}/<class>/*. Everything the paper datasets
    # hard-code (resolution, normalization, class count) is declared here instead, or resolved from
    # the data when left as None, so the attack is not tied to CIFAR-10 / ImageNet.
    image_size: int | None = None  # square side; None -> 32 (CIFAR) / 224 (ImageNet)
    channels: int = 3
    data_mean: list[float] | None = None  # None -> computed once on train/ and cached
    data_std: list[float] | None = None
    augment_shift: int | None = None  # None -> round(image_size/8) (= 4 at 32, 28 at 224)
    augment_flip: bool = True
    num_classes_override: int | None = None  # None -> counted from the ImageFolder

    # --- explicit single target (an alternative to the published case tables) ---
    # When both are set, build_case ignores the paper tables and builds exactly this case.
    target_index: int | None = None  # index into the validation set of the image to misclassify
    adv_class: int | None = None  # y_adv: the label the target should be pushed to
    poison_class: int | None = None  # class the poisons are drawn from; None -> adv_class

    # --- threat model (paper section 3.1) -------------------------------
    budget: float = 0.01  # fraction of the training set that is poisoned
    eps: float = 16.0  # l_inf bound in 0..255 pixel units
    targets: int = 1  # number of target images, T in eq. 1

    # --- evaluation protocol (paper section 5, appendix B) --------------
    cases: int = 10  # random poison-target settings
    victim_runs: int = 8  # m: fresh victim trainings per case (1 for ImageNet)
    case_ids: list[int] | None = None  # subset of case indices, None = all

    # --- brewing (paper section 5: tau=0.1, R=8, M=250) ----------------
    recipe: str = "gradient-matching"  # gradient-matching | poison-frogs | bullseye
    objective: str = "cosine"  # cosine (eq. 3) | euclidean (eq. 2)
    cosine_mode: str = "global"  # global | per_layer  (see DESIGN_DECISIONS 2)
    target_criterion: str = "cross-entropy"  # cross-entropy | carlini-wagner
    tau: float = 0.1
    restarts: int = 8  # R
    attack_steps: int = 250  # M
    attack_optimizer: str = "signAdam"
    attack_scheduling: bool = True
    init: str = "randn"  # randn | rand | zero
    poison_batch_size: int = 512
    poison_shuffle: bool = False
    differentiable_augment: bool = True
    ensemble: int = 1  # K
    ensemble_nets: list[str] | None = None  # heterogeneous ensemble, e.g. Table 3 last row

    # --- models ---------------------------------------------------------
    net: str = "ResNet18"
    victim_net: str | None = None  # defaults to `net`; set for transfer experiments
    width: int = 64  # ResNet base width (Fig. 12 left)
    pretrained: bool = False  # torchvision weights, ImageNet only (appendix B.2)

    # --- victim / surrogate training (paper appendix B.2) ---------------
    epochs: int = 40
    lr: float | None = None  # None -> 0.1 residual, 0.01 otherwise
    batch_size: int = 128
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    lr_milestones: list[int] = field(default_factory=lambda: [14, 24, 35])
    lr_gamma: float = 0.1
    victim_augment: bool = True
    pretrain_epochs: int | None = None  # None -> `epochs`; Fig. 12 right sweeps this

    # --- defenses -------------------------------------------------------
    dp_clip: float | None = None  # appendix F.3 fixes this at 1 when the defense is on
    dp_noise: float | None = None
    attacker_aware_dp: bool = False
    data_ablation: float = 1.0  # Table 7: fraction of the victim's data kept
    # Table 7 has two regimes. With `ablate_surrogate = False` only the victim's training set is
    # reduced ("victim ablation"); with it True the brewing network is trained on a reduced set as
    # well ("pretrained + victim ablation").
    ablate_surrogate: bool = False

    # --- runtime --------------------------------------------------------
    device: str = "auto"
    workers: int = 2
    results_dir: str = "results"
    poison_dir: str = "poisons"
    checkpoint_dir: str = "checkpoints"
    save_poisons: str | None = None  # None | packed | png | imagefolder
    diagnostics: bool = False
    dryrun: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.dataset.lower() == "custom" and not self.data_path:
            raise ValueError("dataset: custom requires data_path pointing at an ImageFolder root")
        if self.poison_class is None:
            self.poison_class = self.adv_class
        if self.victim_net is None:
            self.victim_net = self.net
        if self.lr is None:
            # Appendix B.2: 0.1 for residual architectures, 0.01 for ConvNet/MobileNet-v2/VGG.
            self.lr = 0.1 if self.net.lower().startswith("resnet") else 0.01
        if self.pretrain_epochs is None:
            self.pretrain_epochs = self.epochs

    @property
    def is_paper_setting(self) -> bool:
        """False when a CLI override moved the run away from the published protocol."""
        return not self.overrides

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


_FIELDS = {f.name: f for f in dataclasses.fields(Config)}


def _coerce(name: str, value: Any) -> Any:
    """Cast a string (from the CLI) to the field's declared type."""
    declared = _FIELDS[name].type
    if not isinstance(value, str):
        return value
    text = declared if isinstance(declared, str) else str(declared)
    if value.lower() in {"none", "null"}:
        return None
    if "list" in text:
        return [int(v) if v.strip().lstrip("-").isdigit() else v.strip() for v in value.split(",")]
    if "bool" in text:
        return value.lower() in {"1", "true", "yes", "on"}
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load `configs/base.yaml`, then `path`, then `overrides` (highest priority)."""
    data: dict[str, Any] = {}
    base = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
    for layer in (base, Path(path) if path else None):
        if layer is None or not layer.exists():
            continue
        loaded = yaml.safe_load(layer.read_text(encoding="utf-8")) or {}
        data.update(loaded)

    applied: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        if key not in _FIELDS:
            raise KeyError(f"unknown config field: {key}")
        data[key] = _coerce(key, value)
        applied[key] = data[key]

    unknown = set(data) - set(_FIELDS)
    if unknown:
        raise KeyError(f"unknown config fields in {path}: {sorted(unknown)}")

    data["overrides"] = applied
    return Config(**data)


#: Fields whose default is computed from another field, and the field they follow.
_DERIVED = {"lr": "net", "victim_net": "net", "pretrain_epochs": "epochs"}


def _rederive(clone: Config, changed: str, explicit: set[str]) -> None:
    """Recompute defaults that follow `changed`, unless the user pinned them.

    Appendix B.2 ties the learning rate to the architecture (0.1 for residual nets, 0.01
    otherwise), so a sweep over `net` that kept the previous architecture's learning rate would
    train the new one with the wrong recipe.
    """
    for field_name, source in _DERIVED.items():
        if source == changed and field_name not in explicit and field_name != changed:
            setattr(clone, field_name, None)
    # Re-running the dataclass's own post-init fills every field that is now None.
    clone.__post_init__()


def variants(cfg: Config, sweep: dict[str, list[Any]]) -> list[Config]:
    """Cartesian product of `sweep` applied on top of `cfg`, for the paper's sweep figures."""
    explicit = set(cfg.overrides) | set(sweep)
    out = [cfg]
    for key, values in sweep.items():
        expanded = []
        for base_cfg in out:
            for value in values:
                clone = copy.deepcopy(base_cfg)
                setattr(clone, key, value)
                _rederive(clone, key, explicit)
                clone.name = f"{base_cfg.name}_{key}{value}"
                expanded.append(clone)
        out = expanded
    return out
