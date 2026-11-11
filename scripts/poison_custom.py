"""Poison an arbitrary dataset with an explicit target -- the attack, off the paper's rails.

This is the short path to "poison MY data, MY target, MY parameters". It does not read any published
case table: the target image and the adversarial class come from the command line (or the config),
the poisons are drawn from the adversarial class, and the whole thing runs on any `dataset: custom`
ImageFolder.

    # see which validation index is which image, to pick a target
    python scripts/poison_custom.py --config configs/custom/my_attack.yaml --list-targets

    # brew + evaluate: make validation image 1234 be classified as class 3
    python scripts/poison_custom.py --config configs/custom/my_attack.yaml \
           --target-index 1234 --adv-class 3 --export png

    # brew only, keep the poisons, skip victim training
    python scripts/poison_custom.py --config configs/custom/my_attack.yaml \
           --target-index 1234 --adv-class 3 --brew-only --export imagefolder

One target on a handful of victims gives a binary (0% or 100%) success signal -- that is the nature
of a single-target attack, not a bug. The reliable checks at this scale are `B < 1` (the gradients
aligned) and a clean accuracy close to an unpoisoned model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from _cli import announce, build_parser, config_from_args

from witchesbrew.attacks.brew import Brewer
from witchesbrew.data.augment import augmentation_for
from witchesbrew.data.datasets import build_datasets, num_classes
from witchesbrew.data.setup import build_case
from witchesbrew.evaluate import build_surrogate_ensemble
from witchesbrew.export import export_imagefolder, export_packed, export_png, quantization_error
from witchesbrew.models.factory import build_model
from witchesbrew.train import train_model
from witchesbrew.utils import Timer, get_device, set_seed


def list_targets(cfg) -> None:
    """Print `index | class | path` for the validation set, to choose a --target-index."""
    _, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    base = getattr(validset, "base", validset)
    samples = getattr(base, "samples", None)
    classes = getattr(base, "classes", [])
    if samples is None:
        raise SystemExit("--list-targets needs an ImageFolder dataset (dataset: custom)")
    for index, (path, label) in enumerate(samples):
        name = classes[label] if label < len(classes) else str(label)
        print(f"{index:6d} | {name:<20} | {path}")


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--list-targets", action="store_true", help="list validation images and exit")
    parser.add_argument("--target-index", type=int, default=None, help="validation index to misclassify")
    parser.add_argument("--adv-class", type=int, default=None, help="y_adv: the class to push it to")
    parser.add_argument("--poison-class", type=int, default=None, help="class poisons are drawn from (default: adv-class)")
    parser.add_argument("--export", choices=["packed", "png", "imagefolder"], default="packed")
    parser.add_argument("--brew-only", action="store_true", help="brew and export, skip victim training")
    args = parser.parse_args()

    # CLI target/class override the config, and are recorded as overrides like any --set.
    extra = {}
    if args.target_index is not None:
        extra["target_index"] = str(args.target_index)
    if args.adv_class is not None:
        extra["adv_class"] = str(args.adv_class)
    if args.poison_class is not None:
        extra["poison_class"] = str(args.poison_class)
    for item in args.set:
        key, value = item.split("=", 1)
        extra.setdefault(key.strip(), value.strip())
    args.set = [f"{k}={v}" for k, v in extra.items()]
    cfg = config_from_args(args)

    if args.list_targets:
        list_targets(cfg)
        return

    if cfg.target_index is None or cfg.adv_class is None:
        raise SystemExit("need --target-index and --adv-class (or target_index/adv_class in the config)")

    announce(cfg)
    device = get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    classes = num_classes(cfg.dataset)
    case = build_case(0, cfg, trainset, validset)
    print(
        f"target '{case.target_class_name}' (id {case.target_ids}) "
        f"-> adversarial class '{case.poison_class_name}', {len(case.poison_ids)} poisons"
    )

    set_seed(case.seed)
    models = build_surrogate_ensemble(cfg, trainset, validset, device, case.seed)

    target_inputs = torch.stack([validset[i][0] for i in case.target_ids]).to(device)
    target_true = torch.tensor([validset[i][1] for i in case.target_ids], device=device)
    adv_labels = torch.full((len(case.target_ids),), case.poison_class, device=device)
    clean = torch.stack([trainset[i][0] for i in case.poison_ids])
    labels = torch.tensor([trainset[i][1] for i in case.poison_ids])

    timer = Timer()
    brewer = Brewer(
        cfg,
        models,
        clean,
        labels,
        target_inputs,
        adv_labels,
        device,
        augment=augmentation_for(cfg).to(device) if cfg.differentiable_augment else None,
    )
    delta, brew_loss, per_restart = brewer.brew()
    print(f"B = {brew_loss:.4f} after {timer.minutes():.1f} min; per restart {[round(v, 4) for v in per_restart]}")

    # Export the poisons.
    out_dir = Path(cfg.poison_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.export == "packed":
        print("wrote", export_packed(delta, case.poison_ids, out_dir / "poisons.pth"))
    elif args.export == "png":
        print("wrote", export_png(clean.to(device), delta, case.poison_ids, cfg.dataset, out_dir))
    else:
        print("wrote", export_imagefolder(clean.to(device), delta, labels, case.poison_ids,
                                          trainset.classes, cfg.dataset, out_dir / "imagefolder"))
    report = quantization_error(clean.to(device), delta, cfg.dataset)
    print(f"l_inf after quantization: {report['max_abs_delta_pixels_quantized']:.2f} / {cfg.eps:g} pixel values")

    result = {
        "case": case.to_dict(),
        "brew_loss": brew_loss,
        "brew_minutes": timer.minutes(),
        "quantization": report,
    }

    if not args.brew_only:
        successes, accuracies = [], []
        runs = 1 if cfg.dryrun else cfg.victim_runs
        for run in range(runs):
            victim_seed = case.seed + 7919 * (run + 1)
            set_seed(victim_seed)
            victim = build_model(cfg.victim_net, cfg.dataset, classes, cfg.width).to(device)
            history = train_model(
                cfg, victim, trainset, validset, device,
                poison_delta=delta, poison_ids=case.poison_ids,
                target_inputs=target_inputs, adv_labels=adv_labels,
                true_labels=target_true, target_ids=case.target_ids,
                augment=augmentation_for(cfg).to(device), seed=victim_seed,
            )
            hit = history.target_adv_acc[-1] if history.target_adv_acc else 0.0
            successes.append(hit)
            accuracies.append(history.valid_acc[-1])
            print(f"  victim {run + 1}/{runs}: target hit {100 * hit:.0f}%, clean acc {100 * history.valid_acc[-1]:.2f}%")
        result["poison_success"] = 100 * sum(successes) / len(successes)
        result["clean_valid_acc"] = 100 * sum(accuracies) / len(accuracies)
        print(f"\npoison success : {result['poison_success']:.1f}%  (over {runs} victims)")
        print(f"clean val acc  : {result['clean_valid_acc']:.2f}%")

    results_dir = Path(cfg.results_dir) / cfg.name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "custom_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("report ->", results_dir / "custom_result.json")


if __name__ == "__main__":
    main()
