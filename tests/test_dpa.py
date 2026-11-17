"""Piste 2: DPA hash / partitions / vote, the placement lemma, and the end-to-end DPA case.

CPU-only, on the synthetic FakeCIFAR of test_pipeline.py. The placement tests encode the lemma: with
a public hash every poison lands in the partition the attacker chose, within the eps budget; with a
secret key the attacker is refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from witchesbrew import evaluate as evaluate_module
from witchesbrew.attacks.brew import PoisonBounds
from witchesbrew.attacks.placement import allocate, choose_partitions, place_in_partition
from witchesbrew.config import load_config
from witchesbrew.data.datasets import IndexedDataset
from witchesbrew.defenses.dpa import (
    aggregate_votes,
    certified_radius,
    flips_needed,
    hash_key_for,
    partition_dataset,
    partition_hash,
    to_uint8,
)
from witchesbrew.evaluate import run_dpa_case, run_dpa_experiment

from test_pipeline import FakeCIFAR, tiny_cfg


@pytest.fixture
def fake_data(monkeypatch):
    train = IndexedDataset(FakeCIFAR(200, seed=1))
    valid = IndexedDataset(FakeCIFAR(100, seed=2))
    monkeypatch.setattr(evaluate_module, "build_datasets", lambda *a, **k: (train, valid))
    return train, valid


# --- hash -------------------------------------------------------------------------------------


def test_pixel_sum_hash_matches_hand_computation():
    imgs = torch.randint(0, 256, (5, 3, 8, 8), dtype=torch.int64)
    got = partition_hash(imgs, k=7, mode="sum")
    want = imgs.reshape(5, -1).sum(dim=1) % 7
    assert torch.equal(got, want)


def test_weighted_hash_is_keyed_and_secret_differs():
    imgs = torch.randint(0, 256, (6, 3, 8, 8), dtype=torch.int64)
    a = partition_hash(imgs, 10, "weighted", key=1)
    b = partition_hash(imgs, 10, "weighted", key=1)
    c = partition_hash(imgs, 10, "secret", key=2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert (a >= 0).all() and (a < 10).all()


def test_hash_key_for_reads_the_defenders_key():
    assert hash_key_for(load_config(None, {"dpa_hash": "sum"})) is None
    assert hash_key_for(load_config(None, {"dpa_hash": "weighted", "dpa_hash_key": "5"})) == 5
    assert hash_key_for(load_config(None, {"dpa_hash": "secret", "dpa_secret_key": "9"})) == 9
    with pytest.raises(ValueError):
        hash_key_for(load_config(None, {"dpa_hash": "secret"}))


def test_to_uint8_is_the_stored_image():
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=torch.device("cpu"))
    pixel = torch.randint(0, 256, (2, 3, 32, 32)).float() / 255.0
    normalized = (pixel - bounds.dm) / bounds.ds
    assert torch.equal(to_uint8(normalized, "CIFAR10"), torch.round(pixel * 255).to(torch.int64))


# --- vote ---------------------------------------------------------------------------------------


def test_vote_margin_certificate_and_m0():
    preds = [0] * 30 + [1] * 20
    winner, votes, margin = aggregate_votes(preds, 10)
    assert winner == 0 and votes[0] == 30 and votes[1] == 20
    assert margin == 10  # 30 - (20 + 0): class 1 > class 0 so no tie bonus
    assert certified_radius(margin) == 4  # s < 5
    assert flips_needed(margin) == 6  # floor(10/2) + 1


def test_vote_ties_go_to_the_smaller_class():
    winner, _, margin = aggregate_votes([3] * 5 + [2] * 5, 10)
    assert winner == 2
    assert margin == 0  # V_2 - (V_3 + 0) = 0: one more vote for 3 would flip it
    winner2, _, margin2 = aggregate_votes([2] * 5 + [3] * 5 + [3], 10)
    assert winner2 == 3 and margin2 == 0  # 6 - (5 + 1[2 < 3])


# --- partitions ---------------------------------------------------------------------------------


def test_partition_dataset_is_a_partition_and_hashes_poisons_after_delta(fake_data):
    train, _ = fake_data
    ids = list(range(len(train)))
    parts = partition_dataset(train, ids, 4, "sum", None, "CIFAR10", torch.device("cpu"))
    flat = sorted(i for p in parts for i in p)
    assert flat == ids
    # Poison a single image strongly: it may change partition, nothing else moves.
    delta = torch.full((1, 3, 32, 32), 0.3)
    parts2 = partition_dataset(
        train, ids, 4, "sum", None, "CIFAR10", torch.device("cpu"), delta=delta, poison_ids=[7]
    )
    moved = {i for p in parts for i in p} ^ {i for p in parts2 for i in p}
    assert moved == set()
    where = {i: p for p, lst in enumerate(parts) for i in lst}
    where2 = {i: p for p, lst in enumerate(parts2) for i in lst}
    assert all(where[i] == where2[i] for i in ids if i != 7)


# --- placement lemma ----------------------------------------------------------------------------


def _poisons(n=6, seed=0):
    torch.manual_seed(seed)
    bounds = PoisonBounds(eps=16.0, dataset="CIFAR10", device=torch.device("cpu"))
    pixel = torch.randint(0, 256, (n, 3, 32, 32)).float() / 255.0
    clean = (pixel - bounds.dm) / bounds.ds
    # Brewed at eps - 2 (the placement margin), as run_dpa_case does.
    small = PoisonBounds(eps=14.0, dataset="CIFAR10", device=torch.device("cpu"))
    delta = small.project(torch.randn_like(clean) * small.eps_normalized, clean)
    return clean, delta, bounds


@pytest.mark.parametrize("mode,key", [("sum", None), ("weighted", 3)])
def test_placement_lands_every_poison_in_its_partition_within_eps(mode, key):
    clean, delta, bounds = _poisons()
    k = 10
    ids = list(range(6))
    assignment = allocate(ids, [2, 7])  # equal split over two partitions
    assert [assignment[i] for i in ids] == [2, 7, 2, 7, 2, 7]
    new_delta, report = place_in_partition(
        delta, clean, "CIFAR10", assignment, ids, k, mode, key, eps=16.0,
        rng=np.random.default_rng(0),
    )
    stored = to_uint8(clean + new_delta, "CIFAR10")
    got = partition_hash(stored, k, mode, key)
    assert [int(g) for g in got] == [assignment[i] for i in ids]
    assert report["linf_final_255"] <= 16.0 + 1e-6
    assert (new_delta.abs() <= bounds.eps_normalized + 1e-5).all()
    # The stored image is on the 8-bit grid and inside [0, 1].
    pixel = clean * bounds.ds + bounds.dm + new_delta * bounds.ds
    assert pixel.min() >= -1e-5 and pixel.max() <= 1 + 1e-5


def test_placement_cost_is_tiny_for_the_pixel_sum():
    clean, delta, _ = _poisons()
    assignment = allocate(list(range(6)), [0])
    new_delta, report = place_in_partition(
        delta, clean, "CIFAR10", assignment, list(range(6)), 10, "sum", None, 16.0,
        np.random.default_rng(1),
    )
    assert report["mean_shift_pixels"] < 10  # at most k-1 = 9 pixels moved by 1/255


def test_placement_is_refused_under_a_secret_hash():
    clean, delta, _ = _poisons()
    with pytest.raises(ValueError):
        place_in_partition(
            delta, clean, "CIFAR10", {0: 1}, [0], 10, "secret", 42, 16.0, np.random.default_rng(0)
        )


def test_choose_partitions_random_and_weakest():
    rng = np.random.default_rng(0)
    assert choose_partitions(10, 0, "random", rng) == []
    picked = choose_partitions(10, 3, "random", rng)
    assert len(picked) == 3 and len(set(picked)) == 3
    margins = [5.0, -1.0, 2.0, 0.5, 9.0]
    assert choose_partitions(5, 2, "weakest", rng, margins) == [1, 3]
    with pytest.raises(ValueError):
        choose_partitions(5, 6, "random", rng)


# --- end to end ---------------------------------------------------------------------------------


def _dpa_cfg(tmp_path, **overrides):
    base = {"dpa_k": "4", "budget": "0.05", "data_ablation": "1.0"}
    base.update(overrides)
    return tiny_cfg(tmp_path, **base)


def test_run_dpa_case_end_to_end_with_placement(fake_data, tmp_path):
    cfg = _dpa_cfg(tmp_path, dpa_attack_partitions="2", dpa_pick="weakest")
    result = run_dpa_case(cfg, 0, torch.device("cpu"))

    dpa = result["dpa"]
    assert dpa["k"] == 4 and len(dpa["attacked_partitions"]) == 2
    assert result["case_success"] in (0.0, 1.0)
    assert result["brew_eps"] == cfg.eps - cfg.dpa_placement_margin
    assert sum(dpa["partition_sizes"]) == 200
    assert len(dpa["poisoned_base_preds"]) == 4 and len(dpa["clean_base_preds"]) == 4
    # Every poison sits in an attacked partition, split evenly.
    counts = {r["partition"]: r["poisons"] for r in dpa["per_partition"]}
    assert set(counts) == set(dpa["attacked_partitions"])
    assert sum(counts.values()) == result["poison_count"]
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(isinstance(r["flipped"], bool) for r in dpa["per_partition"])
    assert 0.0 <= dpa["flip_rate"] <= 1.0
    assert dpa["placement"]["placed"] >= 0
    # The clean reference was cached as JSON.
    cached = list((tmp_path / "checkpoints").glob("dpa_clean_*.json"))
    assert len(cached) == 1 and "base" in json.loads(cached[0].read_text())
    # `weakest` picked the two partitions with the smallest clean margins.
    order = np.argsort(dpa["clean_margins"], kind="stable")[:2]
    assert sorted(int(i) for i in order) == dpa["attacked_partitions"]


def test_run_dpa_case_uniform_and_secret(fake_data, tmp_path):
    cfg = _dpa_cfg(tmp_path, dpa_hash="secret", dpa_secret_key="7")
    result = run_dpa_case(cfg, 0, torch.device("cpu"))
    assert result["dpa"]["attacked_partitions"] == []
    assert result["brew_eps"] == cfg.eps
    assert result["dpa"]["placement"] == {"placed": 0}
    with pytest.raises(ValueError):
        run_dpa_case(_dpa_cfg(tmp_path, dpa_hash="secret", dpa_secret_key="7",
                              dpa_attack_partitions="1"), 0, torch.device("cpu"))


def test_run_dpa_experiment_writes_summary_and_single_victim(fake_data, tmp_path):
    cfg = _dpa_cfg(tmp_path, dpa_single_victim="true", dpa_partition_surrogate="true",
                   dpa_attack_partitions="1")
    summary = run_dpa_experiment(cfg)
    assert summary["cases"] == 1
    assert 0 <= summary["avg_poison_success"] <= 100
    assert summary["avg_single_victim_success"] is not None
    path = tmp_path / "results" / cfg.name / "summary.json"
    assert path.exists()
    case = json.loads((tmp_path / "results" / cfg.name / "case00.json").read_text())
    assert case["single_victim"] is not None
    assert case["brew_loss_per_partition"]  # one group brewed on its partition surrogate
    assert list((tmp_path / "checkpoints").glob("dpa_part_*.pt"))


def test_run_dpa_case_under_data_ablation_checks_only_ingested_poisons(fake_data, tmp_path):
    """Table 7 ablation: poisons the victim never sees are in no partition and must not trip the check."""
    cfg = _dpa_cfg(tmp_path, data_ablation="0.5", dpa_attack_partitions="2")
    result = run_dpa_case(cfg, 0, torch.device("cpu"))
    assert sum(result["dpa"]["partition_sizes"]) == 100
    assert result["poisons_in_training"] <= result["poison_count"]
    assert sum(r["poisons"] for r in result["dpa"]["per_partition"]) == result["poisons_in_training"]


def test_dpa_off_leaves_run_case_untouched(fake_data, tmp_path):
    """Neutral defaults: the paper path never looks at the DPA fields."""
    cfg = tiny_cfg(tmp_path)
    assert cfg.dpa_k == 0 and cfg.dpa_attack_partitions == 0
    result = evaluate_module.run_case(cfg, 0, torch.device("cpu"))
    assert "dpa" not in result
