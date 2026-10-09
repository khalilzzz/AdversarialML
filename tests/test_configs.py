"""Every shipped config must load, and the ones that name a paper result must match it.

A typo in a YAML file would otherwise surface only after hours of GPU time, as a number that
quietly fails to reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from witchesbrew.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))


def test_configs_are_present():
    assert len(CONFIGS) >= 29


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads(path: Path):
    cfg = load_config(path)
    assert cfg.name
    assert cfg.budget >= 0 and cfg.eps > 0
    assert cfg.restarts >= 1 and cfg.attack_steps >= 1
    assert cfg.recipe in {"gradient-matching", "poison-frogs", "bullseye"}
    assert cfg.objective in {"cosine", "euclidean"}
    assert cfg.cosine_mode in {"global", "per_layer"}
    assert cfg.target_criterion in {"cross-entropy", "carlini-wagner"}
    assert cfg.init in {"randn", "rand", "zero"}
    assert cfg.dataset.upper() in {"CIFAR10", "IMAGENET"}
    assert cfg.lr in {0.1, 0.01}  # appendix B.2 allows only these two


def test_base_config_is_the_published_protocol():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    assert (cfg.tau, cfg.restarts, cfg.attack_steps) == (0.1, 8, 250)  # section 5
    assert (cfg.cases, cfg.victim_runs) == (10, 8)  # appendix B
    assert (cfg.budget, cfg.eps, cfg.targets) == (0.01, 16.0, 1)
    assert cfg.epochs == 40 and cfg.lr_milestones == [14, 24, 35]
    assert cfg.ensemble == 1 and cfg.differentiable_augment and cfg.victim_augment


@pytest.mark.parametrize(
    "name,field,expected",
    [
        # Table 1: the four rows of the augmentation / ensemble ablation.
        ("table1_K1_noaug_victim_noaug", "victim_augment", False),
        ("table1_K1_noaug_victim_noaug", "differentiable_augment", False),
        ("table1_K1_noaug_victim_aug", "victim_augment", True),
        ("table1_K8_noaug_victim_noaug", "ensemble", 8),
        ("table1_K1_diffaug_victim_aug", "differentiable_augment", True),
        # Table 2: the caption's eps per architecture, and the swapped objectives.
        ("table2_resnet18_gm", "eps", 16.0),
        ("table2_convnet_gm", "eps", 32.0),
        ("table2_convnet_gm", "net", "ConvNet"),
        ("table2_resnet18_frogs", "recipe", "poison-frogs"),
        ("table2_resnet18_bullseye", "recipe", "bullseye"),
        # Table 8 ablations.
        ("table8_cw", "target_criterion", "carlini-wagner"),
        ("table8_m50", "attack_steps", 50),
        ("table8_euclidean", "objective", "euclidean"),
        # Table 3 benchmark setting: 1% budget at eps = 8.
        ("transfer_benchmark_k1", "eps", 8.0),
        ("transfer_benchmark_k1", "budget", 0.01),
        ("transfer_benchmark_k6_heterogeneous", "ensemble", 6),
        # Defenses: appendix F.2 uses a 4% budget, F.3 fixes clipping at 1.
        ("filtering", "budget", 0.04),
        ("dp_sweep", "dp_clip", 1.0),
        ("dp_sweep_aware", "attacker_aware_dp", True),
        # ImageNet: no ensembles (section 3.4), pretrained weights (appendix B.2).
        ("imagenet_fig3_budget_eps", "ensemble", 1),
        ("imagenet_fig3_budget_eps", "pretrained", True),
        ("imagenet_fig3_budget_eps", "victim_runs", 1),
    ],
)
def test_config_matches_the_paper_setting(name, field, expected):
    matches = [p for p in CONFIGS if load_config(p).name == name]
    assert matches, f"no config named {name}"
    assert getattr(load_config(matches[0]), field) == expected


def test_heterogeneous_ensemble_matches_table3():
    """Table 3's last row: "2 ResNet-18, 2 MobileNet-V2 and 2 VGG11 models"."""
    path = ROOT / "configs" / "cifar10" / "transfer_benchmark_k6_heterogeneous.yaml"
    nets = load_config(path).ensemble_nets
    assert len(nets) == 6
    assert sum(n == "ResNet18" for n in nets) == 2
    assert sum(n == "MobileNetV2" for n in nets) == 2
    assert sum(n == "VGG11" for n in nets) == 2


def test_smoke_config_is_marked_as_a_non_paper_setting():
    cfg = load_config(ROOT / "configs" / "smoke_cpu.yaml")
    assert cfg.attack_steps < 250 and cfg.victim_runs < 8 and cfg.epochs < 40
    assert cfg.data_ablation < 1.0


# --- sweeps ---------------------------------------------------------------
def test_sweeping_the_architecture_rederives_its_learning_rate():
    """Appendix B.2 ties the learning rate to the architecture, so a sweep must follow it."""
    from witchesbrew.config import variants

    base = load_config(ROOT / "configs" / "base.yaml")
    assert base.net == "ResNet18" and base.lr == 0.1

    points = {cfg.net: cfg for cfg in variants(base, {"net": ["ResNet18", "ConvNet", "VGG11"]})}
    assert points["ResNet18"].lr == 0.1
    assert points["ConvNet"].lr == 0.01
    assert points["VGG11"].lr == 0.01
    # The victim defaults to the brewing architecture unless the sweep says otherwise.
    assert points["ConvNet"].victim_net == "ConvNet"


def test_an_explicit_learning_rate_survives_an_architecture_sweep():
    from witchesbrew.config import variants

    base = load_config(None, {"lr": "0.05"})
    for cfg in variants(base, {"net": ["ResNet18", "ConvNet"]}):
        assert cfg.lr == 0.05


def test_transfer_sweep_keeps_the_brewing_and_victim_networks_independent():
    """Fig. 13 brews on one architecture and evaluates on another."""
    from witchesbrew.config import variants

    base = load_config(ROOT / "configs" / "cifar10" / "transfer_benchmark_k1.yaml")
    points = variants(base, {"victim_net": ["ResNet18", "MobileNetV2", "VGG11"]})
    assert {cfg.victim_net for cfg in points} == {"ResNet18", "MobileNetV2", "VGG11"}
    assert all(cfg.net == "ResNet18" for cfg in points)
    assert all(cfg.lr == 0.1 for cfg in points)  # the brewing net is a ResNet


def test_sweep_points_get_distinct_names():
    from witchesbrew.config import variants

    base = load_config(ROOT / "configs" / "cifar10" / "inset_RK_sweep.yaml")
    points = variants(base, {"restarts": [1, 8], "ensemble": [1, 2, 4]})
    assert len({cfg.name for cfg in points}) == 6
