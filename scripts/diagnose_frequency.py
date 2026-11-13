"""E0 -- measure how much of Witches' Brew's budget lives in a purifiable subspace.

Runs on poisons that have already been brewed and exported (so the brewing can happen on a GPU and
the analysis on a CPU). It loads the perturbation and its case, then for each synthetic purifier Psi
reports: the DCT energy profile of the perturbation, the fraction of its budget that survives Psi,
the matching objective B before and after Psi, and rho_S(x_t).

    # brew + export first (GPU):
    python scripts/brew_case.py --config cifar10/baseline.yaml --case 0 --export packed
    # then diagnose (CPU):
    python scripts/diagnose_frequency.py --config cifar10/baseline.yaml \
           --poisons poisons/cifar10_baseline_case00

Compare WB against WB-LF by brewing twice (--set delta_param=full, then delta_param=lowfreq),
exporting each, and running this on both directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from _cli import announce, build_parser, config_from_args

from witchesbrew.analysis.spectrum import (
    alignment_survival,
    budget_survival,
    delta_to_pixel,
    radial_dct_profile,
    rho_subspace,
)
from witchesbrew.data.datasets import build_datasets, denormalization_tensors
from witchesbrew.defenses.purification import named_purifiers
from witchesbrew.evaluate import build_surrogate_ensemble
from witchesbrew.export import load_packed
from witchesbrew.utils import get_device, set_seed


def _load_case(poison_dir: Path) -> dict:
    report = poison_dir / "report.json"
    if not report.exists():
        raise SystemExit(f"no report.json in {poison_dir}; export with brew_case.py --export packed")
    return json.loads(report.read_text(encoding="utf-8"))["case"]


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--poisons", required=True, help="directory with poisons.pth + report.json")
    args = parser.parse_args()
    cfg = config_from_args(args)
    announce(cfg)

    device = get_device(cfg.device)
    poison_dir = Path(args.poisons)
    delta, poison_ids = load_packed(poison_dir / "poisons.pth")
    delta = delta.to(device)
    case = _load_case(poison_dir)

    trainset, validset = build_datasets(cfg.dataset, cfg.data_path, cfg)
    set_seed(case["seed"])
    models = build_surrogate_ensemble(cfg, trainset, validset, device, case["seed"])
    model = models[0]

    clean = torch.stack([trainset[i][0] for i in case["poison_ids"]]).to(device)
    labels = torch.tensor([trainset[i][1] for i in case["poison_ids"]], device=device)
    target_inputs = torch.stack([validset[i][0] for i in case["target_ids"]]).to(device)
    adv_labels = torch.full((len(case["target_ids"]),), case["poison_class"], device=device)

    dm, ds = denormalization_tensors(cfg.dataset, device)
    clean_pixel = (clean * ds + dm).clamp(0.0, 1.0)
    delta_pixel = delta_to_pixel(delta, cfg.dataset)
    target_pixel = (target_inputs * ds + dm).clamp(0.0, 1.0)

    profile = radial_dct_profile(delta_pixel)
    psis = named_purifiers(cfg)
    per_psi = {}
    for name, psi in psis.items():
        per_psi[name] = {
            "budget_survival": budget_survival(clean_pixel, delta_pixel, psi),
            "rho_target": rho_subspace(target_pixel, psi),
            **alignment_survival(
                model, clean, delta, labels, target_inputs, adv_labels, psi,
                cfg.dataset, cfg.target_criterion,
            ),
        }

    result = {
        "experiment": cfg.name,
        "delta_param": cfg.delta_param,
        "poison_count": len(poison_ids),
        "radial_dct_profile": profile.tolist(),
        "high_freq_energy_fraction": float(profile[len(profile) // 2:].sum()),
        "per_psi": per_psi,
    }
    out_dir = Path(cfg.results_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "frequency_probe.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nperturbation: {cfg.delta_param}, {len(poison_ids)} poisons")
    print(f"high-frequency energy fraction: {result['high_freq_energy_fraction']:.3f}")
    print(f"{'Psi':<10} {'budget surv.':>12} {'B before':>10} {'B after':>10} {'rho_target':>11}")
    for name, d in per_psi.items():
        print(f"{name:<10} {d['budget_survival']:>12.3f} {d['B_before']:>10.3f} "
              f"{d['B_after']:>10.3f} {d['rho_target']:>11.3f}")
    print("report ->", out)


if __name__ == "__main__":
    main()
