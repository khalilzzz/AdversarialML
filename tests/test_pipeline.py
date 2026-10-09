"""End-to-end pipeline tests on a synthetic dataset.

These cover the plumbing between the attack and the evaluation protocol — case construction, the
poison budget, injecting the perturbation into the victim's mini-batches by dataset index, the
success criterion of appendix B, and the resumable result files — without downloading CIFAR-10 or
training anything real. The numbers produced are meaningless; only the mechanics are checked.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch.utils.data import Dataset

from witchesbrew import evaluate as evaluate_module
from witchesbrew.config import load_config
from witchesbrew.consts import CIFAR10_CLASSES
from witchesbrew.data.datasets import IndexedDataset
from witchesbrew.data.setup import build_case, num_poisons
from witchesbrew.evaluate import run_case, run_experiment
from witchesbrew.train import train_model
from witchesbrew.utils import mean_and_stderr


class FakeCIFAR(Dataset):
    """Deterministic stand-in with CIFAR-10's shape, label set and class names."""

    def __init__(self, count: int, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(count, 3, 32, 32, generator=generator)
        self.targets = [index % 10 for index in range(count)]
        self.classes = list(CIFAR10_CLASSES)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


@pytest.fixture
def fake_data(monkeypatch):
    train = IndexedDataset(FakeCIFAR(200, seed=1))
    valid = IndexedDataset(FakeCIFAR(100, seed=2))
    monkeypatch.setattr(evaluate_module, "build_datasets", lambda *a, **k: (train, valid))
    return train, valid


def tiny_cfg(tmp_path, **overrides):
    base = {
        "name": "pipeline_test",
        "width": "4",
        "restarts": "1",
        "attack_steps": "2",
        "poison_batch_size": "16",
        "epochs": "1",
        "pretrain_epochs": "1",
        "lr_milestones": "1",
        "victim_runs": "1",
        "cases": "1",
        "case_ids": "0",
        "workers": "0",
        "device": "cpu",
        "results_dir": str(tmp_path / "results"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "poison_dir": str(tmp_path / "poisons"),
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return load_config(None, base)


# --- case construction ----------------------------------------------------
def test_case_uses_the_published_class_pair(fake_data):
    train, valid = fake_data
    cfg = load_config(None, {"budget": "0.05"})
    case = build_case(0, cfg, train, valid)

    # Paper Table 4, first row: target class dog, adversarial (poison) class frog.
    assert case.target_class_name == "dog" and case.poison_class_name == "frog"
    assert case.target_class == CIFAR10_CLASSES.index("dog")
    assert case.poison_class == CIFAR10_CLASSES.index("frog")
    assert case.seed == 2000000000


def test_poisons_are_drawn_from_the_adversarial_class(fake_data):
    """Alg. 1 line 3: "Select P training images with label y_adv"."""
    train, valid = fake_data
    cfg = load_config(None, {"budget": "0.05"})
    case = build_case(0, cfg, train, valid)
    labels = train.targets
    assert all(labels[i] == case.poison_class for i in case.poison_ids)
    assert len(set(case.poison_ids)) == len(case.poison_ids)


def test_budget_sets_the_number_of_poisons(fake_data):
    train, valid = fake_data
    for budget, expected in ((0.01, 2), (0.05, 10), (0.1, 20)):
        cfg = load_config(None, {"budget": str(budget)})
        assert len(build_case(0, cfg, train, valid).poison_ids) == expected
        assert num_poisons(budget, len(train)) == expected


def test_case_construction_is_deterministic(fake_data):
    train, valid = fake_data
    cfg = load_config(None, {"budget": "0.05"})
    first = build_case(0, cfg, train, valid)
    second = build_case(0, cfg, train, valid)
    assert first.poison_ids == second.poison_ids
    assert first.target_ids == second.target_ids
    # Different published cases are different settings.
    assert build_case(1, cfg, train, valid).poison_class != first.poison_class


def test_target_comes_from_the_target_class(fake_data):
    train, valid = fake_data
    cfg = load_config(None, {"budget": "0.05"})
    case = build_case(0, cfg, train, valid)
    assert all(valid.targets[i] == case.target_class for i in case.target_ids)
    assert len(case.target_ids) == 1  # T = 1 for the main results


def test_multiple_targets_are_requested_when_configured(fake_data):
    train, valid = fake_data
    cfg = load_config(None, {"budget": "0.05", "targets": "3"})
    case = build_case(0, cfg, train, valid)
    assert len(case.target_ids) == 3
    assert len(set(case.target_ids)) == 3


# --- victim training ------------------------------------------------------
def test_perturbation_reaches_the_victim_by_dataset_index(fake_data, tmp_path):
    """The victim must receive x_i + Delta_i for exactly the poisoned indices, nothing else."""
    train, valid = fake_data
    cfg = tiny_cfg(tmp_path, budget=0.05)
    poison_ids = [0, 10, 20]
    delta = torch.zeros(len(poison_ids), 3, 32, 32)
    delta[:] = 0.25

    seen: dict[int, torch.Tensor] = {}

    class Recorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3 * 32 * 32, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for row in range(x.shape[0]):
                seen.setdefault(len(seen), x[row].detach().clone())
            return self.linear(x.flatten(1))

    cfg.victim_augment = False
    train_model(
        cfg, Recorder(), train, valid, torch.device("cpu"),
        poison_delta=delta, poison_ids=poison_ids, seed=0,
    )
    stacked = torch.stack(list(seen.values()))
    # The recorder also sees the validation pass, so compare against every unperturbed image.
    originals = torch.stack(
        [train[i][0] for i in range(len(train))] + [valid[i][0] for i in range(len(valid))]
    )
    # Exactly the poisoned images must differ from every clean image; the rest pass through.
    shifted = 0
    for image in stacked:
        gaps = (image.unsqueeze(0) - originals).flatten(1).abs().max(dim=1).values
        if float(gaps.min()) > 1e-6:  # not equal to any clean image
            shifted += 1
    assert shifted == len(poison_ids)


def test_victim_training_runs_without_poisons(fake_data, tmp_path):
    train, valid = fake_data
    cfg = tiny_cfg(tmp_path)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 10))
    history = train_model(cfg, model, train, valid, torch.device("cpu"), seed=0)
    assert len(history.epochs) == 1
    assert 0.0 <= history.valid_acc[-1] <= 1.0


def test_data_ablation_shrinks_the_victim_training_set(fake_data, tmp_path):
    """Table 7 / appendix F.5: the victim trains on only part of the data."""
    train, valid = fake_data
    cfg = tiny_cfg(tmp_path, data_ablation=0.5)
    counted = {"batches": 0}

    class Counter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            counted["batches"] += x.shape[0]
            return self.linear(x.flatten(1))

    train_model(cfg, Counter(), train, valid, torch.device("cpu"), seed=0)
    # One training epoch plus one validation pass over 100 images.
    assert counted["batches"] == len(train) // 2 + len(valid)


# --- full case and experiment --------------------------------------------
def test_run_case_produces_the_expected_record(fake_data, tmp_path):
    cfg = tiny_cfg(tmp_path, budget=0.05)
    result = run_case(cfg, 0, torch.device("cpu"))

    assert result["poison_count"] == 10
    assert 0.0 <= result["case_success"] <= 1.0
    assert 0.0 <= result["mean_valid_acc"] <= 1.0
    assert len(result["victims"]) == 1
    assert result["brew_loss"] == min(result["brew_loss_per_restart"])
    assert result["case"]["target_class_name"] == "dog"
    assert result["paper_setting"] is False  # tiny_cfg uses overrides


def test_run_experiment_writes_resumable_per_case_files(fake_data, tmp_path):
    cfg = tiny_cfg(tmp_path, budget=0.05)
    summary = run_experiment(cfg)

    case_file = tmp_path / "results" / cfg.name / "case00.json"
    assert case_file.exists()
    payload = json.loads(case_file.read_text(encoding="utf-8"))
    assert payload["case"]["case_id"] == 0
    assert summary["cases"] == 1
    assert 0.0 <= summary["avg_poison_success"] <= 100.0

    # A second call must reuse the stored case rather than recomputing it.
    before = case_file.stat().st_mtime_ns
    run_experiment(cfg, resume=True)
    assert case_file.stat().st_mtime_ns == before


def test_surrogate_checkpoint_is_cached(fake_data, tmp_path):
    cfg = tiny_cfg(tmp_path, budget=0.05)
    run_case(cfg, 0, torch.device("cpu"))
    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1


# --- reported statistics --------------------------------------------------
def test_standard_error_matches_the_paper_definition():
    """Appendix B: error bars are the standard error over the per-case averages."""
    mean, stderr = mean_and_stderr([1.0, 1.0, 0.0, 1.0])
    assert mean == pytest.approx(0.75)
    assert stderr == pytest.approx(0.25)
    assert mean_and_stderr([0.5])[1] == 0.0


# --- surrogate reproducibility -------------------------------------------
def test_surrogate_initialization_depends_only_on_its_seed(fake_data, tmp_path):
    """A surrogate is cached by seed, so its weights must not depend on construction order."""
    from witchesbrew.evaluate import get_surrogate, surrogate_ablation, surrogate_path

    train, valid = fake_data
    cfg = tiny_cfg(tmp_path, epochs=0, pretrain_epochs=0)
    device = torch.device("cpu")

    first = get_surrogate(cfg, train, valid, device, seed=99)
    weights = {k: v.clone() for k, v in first.state_dict().items()}

    torch.randn(1000)  # disturb the global RNG between the two constructions
    second = get_surrogate(cfg, train, valid, device, seed=99)

    for key, value in second.state_dict().items():
        assert torch.equal(value, weights[key]), key
    assert surrogate_path(cfg, 99).exists()
    assert surrogate_ablation(cfg) == 1.0


def test_victim_ablation_does_not_reduce_the_surrogate_training_set(tmp_path):
    """Appendix F.5 regime 1: only the victim's data is reduced unless `ablate_surrogate` is set."""
    from witchesbrew.evaluate import surrogate_ablation, surrogate_path

    victim_only = tiny_cfg(tmp_path, data_ablation=0.3)
    both = tiny_cfg(tmp_path, data_ablation=0.3, ablate_surrogate=True)
    assert surrogate_ablation(victim_only) == 1.0
    assert surrogate_ablation(both) == 0.3
    # The two regimes must not share a cached surrogate.
    assert surrogate_path(victim_only, 0) != surrogate_path(both, 0)
