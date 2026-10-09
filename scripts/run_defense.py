"""Reproduce the two defense experiments of section 5.3.

Differential privacy (Fig. 4b, appendix F.3) -- sweep the gradient noise with clipping fixed at 1:

    python scripts/run_defense.py dp --config defenses/dp_sweep.yaml
    python scripts/run_defense.py dp --config defenses/dp_sweep_aware.yaml --attacker-aware

Outlier filtering (Fig. 4a, Table 6) -- brew at a 4% budget, train one victim, then measure how well
feature-space filtering separates poisons from clean data:

    python scripts/run_defense.py filtering --config defenses/filtering.yaml

The filtering experiment reports the Table 6 columns: expected poisons and clean images removed at
10% and 20% filtering, next to the random-guessing baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from _cli import announce, build_parser, config_from_args

from witchesbrew.attacks.brew import Brewer
from witchesbrew.data.augment import build_augmentation
from witchesbrew.data.datasets import build_datasets, num_classes
from witchesbrew.data.setup import build_case
from witchesbrew.defenses.dp import dp_sweep_configs
from witchesbrew.defenses.filtering import extract_features, feature_statistics, filter_outliers
from witchesbrew.evaluate import build_surrogate_ensemble, run_experiment
from witchesbrew.models.factory import build_model
from witchesbrew.train import train_model
from witchesbrew.utils import get_device, set_seed


def run_dp(args) -> None:
    base = config_from_args(args)
    rows = []
    for cfg in dp_sweep_configs(base, attacker_aware=args.attacker_aware):
        announce(cfg)
        summary = run_experiment(cfg, resume=not args.no_resume)
        summary["dp_noise"] = cfg.dp_noise or 0.0
        summary["dp_clip"] = cfg.dp_clip
        summary["attacker_aware"] = cfg.attacker_aware_dp
        rows.append(summary)
        print(
            f"-> noise {summary['dp_noise']:g}: success {summary['avg_poison_success']:.2f}%, "
            f"val acc {summary['avg_valid_acc']:.2f}%\n"
        )
    out = Path(base.results_dir) / f"{base.name}_dp_sweep.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", out)
    print("paper (Fig. 4b): success below 15% costs more than 20% validation accuracy")


def run_filtering(args) -> None:
    cfg = config_from_args(args)
    announce(cfg)
    device = get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path)
    augment = build_augmentation(cfg.dataset).to(device)

    all_reports = []
    case_ids = cfg.case_ids if cfg.case_ids is not None else list(range(cfg.cases))
    for case_id in case_ids:
        case = build_case(case_id, cfg, trainset, validset)
        set_seed(case.seed)
        models = build_surrogate_ensemble(cfg, trainset, validset, device, case.seed)

        target_inputs = torch.stack([validset[i][0] for i in case.target_ids]).to(device)
        adv_labels = torch.full((len(case.target_ids),), case.poison_class, device=device)
        clean = torch.stack([trainset[i][0] for i in case.poison_ids])
        labels = torch.tensor([trainset[i][1] for i in case.poison_ids])

        brewer = Brewer(
            cfg, models, clean, labels, target_inputs, adv_labels, device,
            augment=augment if cfg.differentiable_augment else None,
        )
        delta, _, _ = brewer.brew()

        # The defense runs on the victim's own features, so train a victim on the poisoned data.
        set_seed(case.seed + 1)
        victim = build_model(cfg.victim_net, cfg.dataset, num_classes(cfg.dataset), cfg.width).to(device)
        train_model(
            cfg, victim, trainset, validset, device,
            poison_delta=delta, poison_ids=case.poison_ids,
            target_inputs=target_inputs, adv_labels=adv_labels,
            true_labels=torch.tensor([validset[i][1] for i in case.target_ids], device=device),
            target_ids=case.target_ids, augment=augment, seed=case.seed + 1,
        )

        # Every training image of the poison class, poisoned ones with their perturbation applied.
        class_ids = [i for i, label in enumerate(trainset.targets) if label == case.poison_class]
        lookup = {pid: row for row, pid in enumerate(case.poison_ids)}
        images = []
        is_poison = []
        for index in class_ids:
            image = trainset[index][0]
            if index in lookup:
                image = image + delta[lookup[index]].cpu()
                is_poison.append(True)
            else:
                is_poison.append(False)
            images.append(image)
        images = torch.stack(images)

        features = extract_features(victim, images, device)
        target_feature = extract_features(victim, target_inputs.cpu(), device)[0]
        stats = feature_statistics(features, target_feature)
        flags = np.asarray(is_poison)

        case_reports = {
            "case_id": case_id,
            "class_size": len(class_ids),
            "poisons": int(flags.sum()),
            "distance_to_centroid_mean_poison": float(stats["distance_to_centroid"][flags].mean()),
            "distance_to_centroid_mean_clean": float(stats["distance_to_centroid"][~flags].mean()),
            "distance_to_target_mean_poison": float(stats["distance_to_target"][flags].mean()),
            "distance_to_target_mean_clean": float(stats["distance_to_target"][~flags].mean()),
            "filters": [
                filter_outliers(stats["distance_to_centroid"], flags, fraction).to_dict()
                for fraction in (0.10, 0.20)
            ],
        }
        all_reports.append(case_reports)
        print(f"case {case_id}: {case_reports['filters']}")

    out = Path(cfg.results_dir) / f"{cfg.name}_filtering.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print("wrote", out)
    print("paper (Table 6): 10% filtering removes 248 poisons / 252 clean; random guessing 200 / 300")


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("defense", choices=["dp", "filtering"])
    parser.add_argument("--attacker-aware", action="store_true", help="appendix F.3 counter-attack")
    args = parser.parse_args()
    if args.defense == "dp":
        run_dp(args)
    else:
        run_filtering(args)


if __name__ == "__main__":
    main()
