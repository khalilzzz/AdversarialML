"""End-to-end evaluation of one poison-target case, following the appendix B methodology.

For each case the paper does three things, in this order:

1. obtain a clean pretrained network, which supplies the fixed `theta` of eq. 3;
2. brew one poisoned subset (Algorithm 1);
3. retrain `m` freshly initialized victims on the poisoned dataset and record, for each, whether the
   target ends up classified as the adversarial class -- "we do not count mere misclassification".

The per-case success rate is the average over the `m` victims; the number reported in the paper is
the mean over the 10 cases, with the standard error of those 10 values as the error bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import numpy as np
from torch.utils.data import Subset

from .analysis.spectrum import delta_report
from .attacks.brew import Brewer
from .attacks.placement import allocate, choose_partitions, place_in_partition
from .data.augment import augmentation_for
from .data.datasets import build_datasets, make_loader, num_classes
from .data.setup import PoisonCase, build_case
from .defenses.dp import make_gradient_hook
from .defenses.dpa import DPAResult, hash_key_for, natural_partition_of, partition_dataset, train_dpa
from .diagnostics import make_diagnostics_fn
from .models.factory import build_model
from .train import ablated_subset, evaluate_accuracy, train_model
from .utils import ResultStore, Timer, available_workers, get_device, mean_and_stderr, set_seed


def _stack(dataset, indices: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices]).to(device)
    return images, labels


def surrogate_ablation(cfg) -> float:
    """How much of the training set the brewing network itself sees.

    Appendix F.5 separates "victim ablation" (only the victim's data is reduced) from
    "pretrained + victim ablation" (the brewing network is trained on a reduced set too), so the
    victim's `data_ablation` must not silently propagate to the surrogate.
    """
    return cfg.data_ablation if cfg.ablate_surrogate else 1.0


def surrogate_purifier(cfg) -> str:
    """Which Psi the surrogate trains under: the brewing Psi for an adaptive attacker, else none."""
    if getattr(cfg, "surrogate_purify", False) and getattr(cfg, "brew_purify", "none") != "none":
        return cfg.brew_purify
    return "none"


def surrogate_path(cfg, seed: int) -> Path:
    psi = surrogate_purifier(cfg)
    # A surrogate trained on purified data is a different model, so it gets its own checkpoint.
    psi_tag = "" if psi == "none" else f"_p{psi}{cfg.purify_cutoff if psi == 'dct' else ''}"
    tag = (
        f"{cfg.dataset}_{cfg.net}_w{cfg.width}_e{cfg.pretrain_epochs}"
        f"_a{surrogate_ablation(cfg):g}{psi_tag}_s{seed}.pt"
    )
    return Path(cfg.checkpoint_dir) / tag


def get_surrogate(cfg, trainset, validset, device: torch.device, seed: int) -> torch.nn.Module:
    """The clean network that provides `theta`.

    ImageNet uses torchvision's pretrained weights (appendix B.2). CIFAR-10 trains one with the
    victim recipe and caches it, since the same surrogate serves every restart and every case that
    shares its seed.
    """
    classes = num_classes(cfg.dataset)
    if cfg.dataset.upper() != "CIFAR10" and cfg.pretrained:
        model = build_model(cfg.net, cfg.dataset, classes, cfg.width, pretrained=True)
        return model.to(device)

    path = surrogate_path(cfg, seed)
    # Seed before constructing the model: a surrogate is identified by its seed, so its random
    # initialization has to depend on that seed alone and not on how many models were built before
    # it. Otherwise the cached checkpoint and a freshly trained one would not agree.
    set_seed(seed)
    model = build_model(cfg.net, cfg.dataset, classes, cfg.width).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    pretrain_cfg = type(cfg)(
        **{
            **cfg.to_dict(),
            "epochs": cfg.pretrain_epochs,
            "data_ablation": surrogate_ablation(cfg),
            # By default theta comes from a *clean* model: the purification defense is a victim-time
            # measure. With `surrogate_purify` the attacker is assumed to know Psi and trains its
            # surrogate the way the victim will, so the matched gradient belongs to the right model.
            "purify": surrogate_purifier(cfg),
        }
    )
    augment = augmentation_for(cfg).to(device)
    train_model(pretrain_cfg, model, trainset, validset, device, augment=augment, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def build_surrogate_ensemble(cfg, trainset, validset, device, seed) -> list[torch.nn.Module]:
    """`K` clean models from different initializations (section 3.4).

    `ensemble_nets` makes the ensemble heterogeneous, which is what Table 3's last row uses
    ("an ensemble of K = 6, consisting of 2 ResNet-18, 2 MobileNet-V2 and 2 VGG11 models").
    """
    if cfg.ensemble_nets:
        models = []
        for index, net in enumerate(cfg.ensemble_nets):
            clone = type(cfg)(**{**cfg.to_dict(), "net": net})
            models.append(get_surrogate(clone, trainset, validset, device, seed + index))
        return models
    return [
        get_surrogate(cfg, trainset, validset, device, seed + index) for index in range(cfg.ensemble)
    ]


def run_case(cfg, case_id: int, device: torch.device | None = None) -> dict:
    """Brew one case and evaluate it on `cfg.victim_runs` fresh victims."""
    device = device or get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    case: PoisonCase = build_case(case_id, cfg, trainset, validset)
    classes = num_classes(cfg.dataset)
    augment = augmentation_for(cfg).to(device)

    set_seed(case.seed)
    models = build_surrogate_ensemble(cfg, trainset, validset, device, case.seed)

    target_inputs, target_true = _stack(validset, case.target_ids, device)
    adv_labels = torch.full_like(target_true, case.poison_class)
    clean_poisons, poison_labels = _stack(trainset, case.poison_ids, device)

    # Appendix F.3: when the noise level is known to the attacker, the same clipping and freshly
    # drawn noise are applied to the poison gradient each time B is evaluated.
    gradient_hook = (
        make_gradient_hook(cfg.dp_clip, cfg.dp_noise) if cfg.attacker_aware_dp else None
    )

    timer = Timer()
    brewer = Brewer(
        cfg,
        models,
        clean_poisons.cpu(),
        poison_labels.cpu(),
        target_inputs,
        adv_labels,
        device,
        augment=augment if cfg.differentiable_augment else None,
        gradient_hook=gradient_hook,
    )
    delta, brew_loss, per_restart = brewer.brew()
    brew_minutes = timer.minutes()
    # E3: where the budget landed, for the success/detectability/visibility trade-off. Cheap, on the
    # brewed delta; the cutoff is the low-frequency subspace boundary (same one WB-LF projects onto).
    delta_metrics = delta_report(
        delta, clean_poisons, cfg.dataset, cfg.lowfreq_cutoff, purify_cutoff=cfg.purify_cutoff
    )

    victim_results = []
    runs = 1 if cfg.dryrun else cfg.victim_runs
    for run in range(runs):
        victim_seed = case.seed + 7919 * (run + 1)  # independent of the brewing seed
        set_seed(victim_seed)
        victim = build_model(cfg.victim_net, cfg.dataset, classes, cfg.width).to(device)
        # Fig. 2 / 5 / 11 measurements are expensive (one extra backward per mini-batch), so they
        # are recorded for the first victim only.
        diagnostics_fn = (
            make_diagnostics_fn(device, target_inputs, adv_labels, target_true, max_batches=50)
            if cfg.diagnostics and run == 0
            else None
        )
        history = train_model(
            cfg,
            victim,
            trainset,
            validset,
            device,
            poison_delta=delta,
            poison_ids=case.poison_ids,
            target_inputs=target_inputs,
            adv_labels=adv_labels,
            true_labels=target_true,
            target_ids=case.target_ids,
            augment=augment,
            diagnostics_fn=diagnostics_fn,
            seed=victim_seed,
        )
        victim_results.append(
            {
                "seed": victim_seed,
                "success": history.target_adv_acc[-1] if history.target_adv_acc else 0.0,
                "valid_acc": history.valid_acc[-1],
                "train_acc": history.train_acc[-1],
                "history": history.to_dict(),
            }
        )

    successes = [r["success"] for r in victim_results]
    return {
        "config": cfg.to_dict(),
        "case": case.to_dict(),
        "poison_count": len(case.poison_ids),
        "brew_loss": brew_loss,
        "brew_loss_per_restart": per_restart,
        "brew_minutes": brew_minutes,
        "victims": victim_results,
        "case_success": sum(successes) / max(1, len(successes)),
        "mean_valid_acc": sum(r["valid_acc"] for r in victim_results) / max(1, len(victim_results)),
        "delta_metrics": delta_metrics,
        "paper_setting": cfg.is_paper_setting,
    }


def run_experiment(cfg, resume: bool = True) -> dict:
    """Loop over the published cases and aggregate exactly as appendix B prescribes."""
    device = get_device(cfg.device)
    store = ResultStore(cfg.results_dir, cfg.name)
    case_ids = cfg.case_ids if cfg.case_ids is not None else list(range(cfg.cases))

    for case_id in case_ids:
        if resume and store.done(case_id):
            continue
        result = run_case(cfg, case_id, device)
        store.write(case_id, result)

    results = store.read_all()
    successes = [r["case_success"] for r in results]
    accuracies = [r["mean_valid_acc"] for r in results]
    mean, stderr = mean_and_stderr(successes)
    acc_mean, acc_stderr = mean_and_stderr(accuracies)
    summary = {
        "experiment": cfg.name,
        "cases": len(results),
        "avg_poison_success": 100 * mean,
        "stderr": 100 * stderr,
        "avg_valid_acc": 100 * acc_mean,
        "valid_acc_stderr": 100 * acc_stderr,
        "mean_brew_minutes": sum(r["brew_minutes"] for r in results) / max(1, len(results)),
        "paper_setting": all(r.get("paper_setting", False) for r in results),
    }
    (store.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------------------------
# Piste 2: Witches' Brew against Deep Partition Aggregation (research extension, NOT the paper).
# ----------------------------------------------------------------------------------------------


def _dpa_clean_tag(cfg, seed: int) -> str:
    key = hash_key_for(cfg)
    return (
        f"dpa_clean_{cfg.dataset}_{cfg.victim_net}_w{cfg.width}_e{cfg.epochs}_k{cfg.dpa_k}"
        f"_{cfg.dpa_hash}{'' if key is None else key}_a{cfg.data_ablation:g}_s{seed}.json"
    )


def _dpa_clean_reference(
    cfg, partitions, trainset, validset, device, target_inputs, adv_labels, target_true, case, augment,
    seed, classes,
) -> dict:
    """The k clean base models on the natural partitions: votes, margin g, m_0, per-model margins.

    Cached as JSON (predictions only): every attack cell of the same case shares this reference, and
    `dpa_pick=weakest` reads its per-partition logit margins.
    """
    path = Path(cfg.checkpoint_dir) / _dpa_clean_tag(cfg, seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    result = train_dpa(
        cfg, trainset, validset, partitions, device, None, None, target_inputs, adv_labels,
        target_true, case.target_ids, augment, seed, classes,
    )
    payload = result.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _partition_surrogate(cfg, trainset, validset, ids: list[int], device, seed: int, tag: str):
    """A surrogate trained on one clean partition only (the attacker simulates base model j)."""
    path = Path(cfg.checkpoint_dir) / f"dpa_part_{tag}.pt"
    classes = num_classes(cfg.dataset)
    set_seed(seed)
    model = build_model(cfg.net, cfg.dataset, classes, cfg.width).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        return model
    pre_cfg = type(cfg)(
        **{**cfg.to_dict(), "epochs": cfg.pretrain_epochs, "data_ablation": 1.0, "purify": "none"}
    )
    augment = augmentation_for(cfg).to(device)
    train_model(pre_cfg, model, Subset(trainset, ids), validset, device, augment=augment, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def run_dpa_case(cfg, case_id: int, device: torch.device | None = None) -> dict:
    """Brew one case, place the poisons (optionally), and evaluate against a DPA victim.

    Steps: clean reference (k clean base models, cached) -> choose m partitions and allocate the
    budget -> brew at `eps - margin` -> steer each poison's hash into its partition -> partition the
    stored dataset -> train the k base models -> vote. Reports, per attacked partition, whether its
    base model flipped to y_adv: the fraction of flips at poison fraction rho is the p(rho) of the
    allocation theorem.
    """
    if cfg.dpa_k <= 0:
        raise ValueError("run_dpa_case needs dpa_k > 0")
    device = device or get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    case: PoisonCase = build_case(case_id, cfg, trainset, validset)
    classes = num_classes(cfg.dataset)
    augment = augmentation_for(cfg).to(device)
    workers = available_workers(cfg.workers)
    k = cfg.dpa_k
    key = hash_key_for(cfg)
    victim_seed = case.seed + 7919  # same base seed as the first victim of `run_case`

    # The victim's data: the (possibly ablated) training set, partitioned by the hash.
    kept = ablated_subset(trainset, cfg.data_ablation, victim_seed)
    kept_ids = list(kept.indices) if isinstance(kept, Subset) else list(range(len(trainset)))

    target_inputs, target_true = _stack(validset, case.target_ids, device)
    adv_labels = torch.full_like(target_true, case.poison_class)
    clean_poisons, poison_labels = _stack(trainset, case.poison_ids, device)

    # 1. Clean reference on the natural partitions.
    clean_parts = partition_dataset(
        trainset, kept_ids, k, cfg.dpa_hash, key, cfg.dataset, device, workers=workers
    )
    clean_ref = _dpa_clean_reference(
        cfg, clean_parts, trainset, validset, device, target_inputs, adv_labels, target_true, case,
        augment, victim_seed, classes,
    )
    # Logit margin each clean base model holds between its own prediction and y_adv.
    clean_margins = [
        b["target_logits"][b["target_pred"]] - b["target_logits"][case.poison_class]
        for b in clean_ref["base"]
    ]

    # 2. Allocation: which partitions, how many poisons each.
    rng = np.random.default_rng((case.seed + 31) % (2**32 - 1))
    attacked = choose_partitions(k, cfg.dpa_attack_partitions, cfg.dpa_pick, rng, clean_margins)
    assignment = allocate(case.poison_ids, attacked)
    if assignment and cfg.dpa_hash == "secret":
        raise ValueError("dpa_hash=secret: placement impossible, set dpa_attack_partitions=0")

    # 3. Brewing, with eps reduced by the placement margin when the hash will be steered.
    brew_eps = cfg.eps - cfg.dpa_placement_margin if assignment else cfg.eps
    brew_cfg = type(cfg)(**{**cfg.to_dict(), "eps": brew_eps})
    timer = Timer()
    set_seed(case.seed)
    brew_losses: dict[str, float] = {}
    if cfg.dpa_partition_surrogate and assignment:
        delta = torch.zeros_like(clean_poisons.cpu())
        for j in attacked:
            rows = [r for r, pid in enumerate(case.poison_ids) if assignment[int(pid)] == j]
            tag = f"{cfg.dataset}_{cfg.net}_w{cfg.width}_e{cfg.pretrain_epochs}_k{k}_s{case.seed}_p{j}"
            model = _partition_surrogate(
                cfg, trainset, validset, clean_parts[j], device, victim_seed + 100_003 * (j + 1), tag
            )
            brewer = Brewer(
                brew_cfg, [model], clean_poisons[rows].cpu(), poison_labels[rows].cpu(),
                target_inputs, adv_labels, device, augment=augment if cfg.differentiable_augment else None,
            )
            group_delta, loss, _ = brewer.brew()
            delta[rows] = group_delta.cpu()
            brew_losses[str(j)] = loss
        delta = delta.to(device)
        brew_loss = float(np.mean(list(brew_losses.values())))
    else:
        models = build_surrogate_ensemble(cfg, trainset, validset, device, case.seed)
        brewer = Brewer(
            brew_cfg, models, clean_poisons.cpu(), poison_labels.cpu(), target_inputs, adv_labels,
            device, augment=augment if cfg.differentiable_augment else None,
        )
        delta, brew_loss, _ = brewer.brew()
    brew_minutes = timer.minutes()

    # 4. Placement: steer each poison's stored image into its partition (lemma (a)/(b)).
    placement = {"placed": 0}
    if assignment:
        delta, placement = place_in_partition(
            delta, clean_poisons, cfg.dataset, assignment, case.poison_ids, k, cfg.dpa_hash, key,
            cfg.eps, rng,
        )
    delta_metrics = delta_report(delta, clean_poisons, cfg.dataset, cfg.lowfreq_cutoff)

    # 5. The DPA victim on the stored (poisoned) dataset.
    parts = partition_dataset(
        trainset, kept_ids, k, cfg.dpa_hash, key, cfg.dataset, device,
        delta=delta, poison_ids=case.poison_ids, workers=workers,
    )
    where = natural_partition_of(case.poison_ids, parts)
    # Under `data_ablation` the victim keeps a random subset: poisons outside it are in no partition
    # (the paper's Table 7 behaviour), so only the ones the victim actually ingests are checked.
    kept_set = set(kept_ids)
    poisons_seen = [int(p) for p in case.poison_ids if int(p) in kept_set]
    if assignment:
        misplaced = [pid for pid in poisons_seen if where.get(pid) != assignment[pid]]
        assert not misplaced, f"{len(misplaced)} poisons did not land in their partition"
    dpa: DPAResult = train_dpa(
        cfg, trainset, validset, parts, device, delta, case.poison_ids, target_inputs, adv_labels,
        target_true, case.target_ids, augment, victim_seed, classes,
    )

    # Per-partition flips: p(rho) of the allocation theorem, measured.
    poisoned_parts = sorted({p for p in where.values()})
    per_partition = []
    for p in poisoned_parts:
        b = dpa.base[p]
        per_partition.append(
            {
                "partition": p,
                "size": b.size,
                "poisons": b.poisons,
                "rho": b.poisons / max(1, b.size),
                "flipped": b.target_pred == case.poison_class,
                "clean_pred": clean_ref["base"][p]["target_pred"],
                "clean_margin": clean_margins[p],
                "target_adv_loss": b.target_adv_loss,
            }
        )
    flips = [r["flipped"] for r in per_partition]

    # 6. Optional: the undefended single victim at the same budget, for the DPA-vs-none comparison.
    single = None
    if cfg.dpa_single_victim:
        set_seed(victim_seed)
        victim = build_model(cfg.victim_net, cfg.dataset, classes, cfg.width).to(device)
        history = train_model(
            cfg, victim, trainset, validset, device, poison_delta=delta, poison_ids=case.poison_ids,
            target_inputs=target_inputs, adv_labels=adv_labels, true_labels=target_true,
            target_ids=case.target_ids, augment=augment, seed=victim_seed,
        )
        single = {
            "success": history.target_adv_acc[-1],
            "valid_acc": history.valid_acc[-1],
            "target_adv_loss": history.target_adv_loss[-1],
        }

    success = float(dpa.prediction == case.poison_class)
    return {
        "config": cfg.to_dict(),
        "case": case.to_dict(),
        "poison_count": len(case.poison_ids),
        "poisons_in_training": len(poisons_seen),
        "dpa": {
            "k": k,
            "hash": cfg.dpa_hash,
            "attacked_partitions": attacked,
            "placement": placement,
            "clean": {kk: v for kk, v in clean_ref.items() if kk != "base"},
            "clean_base_preds": [b["target_pred"] for b in clean_ref["base"]],
            "clean_margins": clean_margins,
            "poisoned": {kk: v for kk, v in dpa.to_dict().items() if kk != "base"},
            "poisoned_base_preds": [b.target_pred for b in dpa.base],
            "partition_sizes": [b.size for b in dpa.base],
            "per_partition": per_partition,
            "flip_rate": float(np.mean(flips)) if flips else 0.0,
            "flipped": int(sum(flips)),
        },
        "single_victim": single,
        "brew_loss": brew_loss,
        "brew_loss_per_partition": brew_losses,
        "brew_eps": brew_eps,
        "brew_minutes": brew_minutes,
        "case_success": success,
        "mean_valid_acc": dpa.valid_acc,
        "delta_metrics": delta_metrics,
        "paper_setting": False,
    }


def run_dpa_experiment(cfg, resume: bool = True) -> dict:
    """Loop `run_dpa_case` over the cases and aggregate like `run_experiment`."""
    device = get_device(cfg.device)
    store = ResultStore(cfg.results_dir, cfg.name)
    case_ids = cfg.case_ids if cfg.case_ids is not None else list(range(cfg.cases))
    for case_id in case_ids:
        if resume and store.done(case_id):
            continue
        store.write(case_id, run_dpa_case(cfg, case_id, device))

    results = store.read_all()
    mean, stderr = mean_and_stderr([r["case_success"] for r in results])
    acc_mean, acc_stderr = mean_and_stderr([r["mean_valid_acc"] for r in results])
    flip_mean, flip_stderr = mean_and_stderr([r["dpa"]["flip_rate"] for r in results])
    singles = [r["single_victim"]["success"] for r in results if r.get("single_victim")]
    summary = {
        "experiment": cfg.name,
        "cases": len(results),
        "avg_poison_success": 100 * mean,
        "stderr": 100 * stderr,
        "avg_valid_acc": 100 * acc_mean,
        "valid_acc_stderr": 100 * acc_stderr,
        "avg_flip_rate": 100 * flip_mean,
        "flip_rate_stderr": 100 * flip_stderr,
        "avg_clean_margin": float(np.mean([r["dpa"]["clean"]["margin"] for r in results])),
        "avg_poisoned_margin": float(np.mean([r["dpa"]["poisoned"]["margin"] for r in results])),
        "avg_single_victim_success": 100 * float(np.mean(singles)) if singles else None,
        "mean_brew_minutes": sum(r["brew_minutes"] for r in results) / max(1, len(results)),
        "paper_setting": False,
    }
    (store.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def clean_baseline_accuracy(cfg, device: torch.device | None = None) -> float:
    """Unpoisoned validation accuracy, the left column of Table 10."""
    device = device or get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    set_seed(1234)
    model = build_model(cfg.victim_net, cfg.dataset, num_classes(cfg.dataset), cfg.width).to(device)
    augment = augmentation_for(cfg).to(device)
    train_model(cfg, model, trainset, validset, device, augment=augment, seed=1234)
    loader = make_loader(validset, 256, shuffle=False, workers=available_workers(cfg.workers))
    return evaluate_accuracy(model, loader, device)
