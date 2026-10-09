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

from .attacks.brew import Brewer
from .data.augment import build_augmentation
from .data.datasets import build_datasets, make_loader, num_classes
from .data.setup import PoisonCase, build_case
from .defenses.dp import make_gradient_hook
from .diagnostics import make_diagnostics_fn
from .models.factory import build_model
from .train import evaluate_accuracy, train_model
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


def surrogate_path(cfg, seed: int) -> Path:
    tag = (
        f"{cfg.dataset}_{cfg.net}_w{cfg.width}_e{cfg.pretrain_epochs}"
        f"_a{surrogate_ablation(cfg):g}_s{seed}.pt"
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
        }
    )
    augment = build_augmentation(cfg.dataset).to(device)
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
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path)
    case: PoisonCase = build_case(case_id, cfg, trainset, validset)
    classes = num_classes(cfg.dataset)
    augment = build_augmentation(cfg.dataset).to(device)

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


def clean_baseline_accuracy(cfg, device: torch.device | None = None) -> float:
    """Unpoisoned validation accuracy, the left column of Table 10."""
    device = device or get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path)
    set_seed(1234)
    model = build_model(cfg.victim_net, cfg.dataset, num_classes(cfg.dataset), cfg.width).to(device)
    augment = build_augmentation(cfg.dataset).to(device)
    train_model(cfg, model, trainset, validset, device, augment=augment, seed=1234)
    loader = make_loader(validset, 256, shuffle=False, workers=available_workers(cfg.workers))
    return evaluate_accuracy(model, loader, device)
