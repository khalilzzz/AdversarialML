"""Brew one case and write the perturbation out, without training any victim.

    python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export png

Useful for three things the paper itself does: inspecting the poisons visually (Figs. 6-8, where
"poisoned data is only barely distinguishable from clean data"), shipping just the poisoned subset
as appendix D describes, and checking that the perturbation stays inside the l_inf bound once it is
quantized to 8-bit pixels.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from _cli import announce, build_parser, config_from_args

from witchesbrew.attacks.brew import Brewer
from witchesbrew.data.augment import augmentation_for
from witchesbrew.data.datasets import build_datasets
from witchesbrew.data.setup import build_case
from witchesbrew.evaluate import build_surrogate_ensemble
from witchesbrew.export import export_imagefolder, export_packed, export_png, quantization_error
from witchesbrew.utils import Timer, get_device, set_seed


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--case", type=int, default=0, help="index into the paper's case table")
    parser.add_argument(
        "--export",
        choices=["packed", "png", "imagefolder"],
        default="packed",
        help="output format (appendix D)",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    announce(cfg)

    device = get_device(cfg.device)
    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    case = build_case(args.case, cfg, trainset, validset)
    print(
        f"case {case.case_id}: target '{case.target_class_name}' (id {case.target_ids}) "
        f"-> adversarial class '{case.poison_class_name}', {len(case.poison_ids)} poisons"
    )

    set_seed(case.seed)
    models = build_surrogate_ensemble(cfg, trainset, validset, device, case.seed)

    target_inputs = torch.stack([validset[i][0] for i in case.target_ids]).to(device)
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
    delta, loss, per_restart = brewer.brew()
    print(f"B = {loss:.4f} after {timer.minutes():.1f} min; per restart {[round(v, 4) for v in per_restart]}")

    out_dir = Path(cfg.poison_dir) / f"{cfg.name}_case{case.case_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.export == "packed":
        print("wrote", export_packed(delta, case.poison_ids, out_dir / "poisons.pth"))
    elif args.export == "png":
        print("wrote", export_png(clean.to(device), delta, case.poison_ids, cfg.dataset, out_dir))
    else:
        print(
            "wrote",
            export_imagefolder(
                clean.to(device),
                delta,
                labels,
                case.poison_ids,
                trainset.classes,
                cfg.dataset,
                out_dir / "imagefolder",
            ),
        )

    report = quantization_error(clean.to(device), delta, cfg.dataset)
    report.update({"eps": cfg.eps, "brew_loss": loss, "case": case.to_dict()})
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"l_inf after quantization: {report['max_abs_delta_pixels_quantized']:.2f} / {cfg.eps:g} pixel values"
    )


if __name__ == "__main__":
    main()
