"""E3 -- the success / detectability / visibility trade-off of the frequency probe.

Research extension, NOT the paper. For a grid of (eps, low-frequency cutoff) this brews and evaluates
one experiment per point and records three quantities per point:

* poison success               -- does the attack still work (from the victim runs);
* detectability (off-manifold) -- the high-frequency DCT energy fraction of the perturbation, a
                                  synthetic stand-in for the EBM energy a purifier flags;
* visibility                   -- 1 - SSIM between the clean and poisoned images.

The *curve* is the contribution (cf. "Pick your Poison"): as the cutoff tightens the perturbation
moves into the preserved subspace, so detectability and the budget destroyed by Psi fall, while
visibility rises and success trades off. Evaluate every point under a victim-side defence with
`--set purify=jpeg` (and brew Psi-aware with `--set brew_purify=jpeg`) to read success *after*
purification.

    python scripts/tradeoff.py --config cifar10/baseline.yaml \
           --eps 16,8 --cutoff 8,12,16 --include-full \
           --set purify=jpeg --set brew_purify=jpeg

Points land under results/<name>__epsE_cutC/, and the collected curve in
results/<name>/tradeoff.json. Plot it with `python scripts/plot.py tradeoff --file <that file>`.
Brewing is the cost, so this is a GPU/Colab driver; a tiny CPU smoke works with a reduced config.
"""

from __future__ import annotations

import json
from pathlib import Path

from _cli import build_parser, config_from_args

from witchesbrew.config import load_config
from witchesbrew.evaluate import run_experiment


def _mean_metric(cases: list[dict], key: str) -> float:
    vals = [c["delta_metrics"][key] for c in cases if "delta_metrics" in c]
    return sum(vals) / len(vals) if vals else float("nan")


def _run_point(base_args, eps: float, cutoff, base_name: str) -> dict:
    """Brew + evaluate one (eps, cutoff) point and read its success and delta metrics back."""
    overrides = dict(base_args_overrides(base_args))
    tag = f"eps{eps:g}_cut{'full' if cutoff is None else cutoff}"
    overrides["eps"] = str(eps)
    overrides["name"] = f"{base_name}__{tag}"
    if cutoff is None:
        overrides["delta_param"] = "full"
    else:
        overrides["delta_param"] = "lowfreq"
        overrides["lowfreq_cutoff"] = str(cutoff)

    cfg = load_config(base_args.config_resolved, overrides)
    print(f"\n=== point {tag}: eps={eps:g} cutoff={cutoff} ===")
    summary = run_experiment(cfg, resume=not base_args.no_resume)

    cases = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((Path(cfg.results_dir) / cfg.name).glob("case*.json"))
    ]
    return {
        "eps": eps,
        "cutoff": cutoff,
        "name": cfg.name,
        "success": summary["avg_poison_success"],
        "stderr": summary["stderr"],
        "valid_acc": summary["avg_valid_acc"],
        "brew_minutes": summary["mean_brew_minutes"],
        "ssim": _mean_metric(cases, "ssim"),
        "visibility": 1.0 - _mean_metric(cases, "ssim"),
        "offmanifold_hf": _mean_metric(cases, "offmanifold_hf"),
        "l2_pixel": _mean_metric(cases, "l2_pixel"),
        "linf_pixel": _mean_metric(cases, "linf_pixel"),
    }


def base_args_overrides(args) -> dict:
    """The --set overrides shared by every point (purify / brew_purify / any field)."""
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--eps", default="16", help="comma-separated l_inf budgets (0..255 units)")
    parser.add_argument("--cutoff", default="8,12,16", help="comma-separated low-frequency cutoffs")
    parser.add_argument(
        "--include-full", action="store_true", help="also run full-budget WB (delta_param=full)"
    )
    args = parser.parse_args()

    # Resolve the config path the same way _cli does, and stash it for per-point rebuilds.
    path = args.config
    if path and not Path(path).exists():
        candidate = Path(__file__).resolve().parents[1] / "configs" / path
        if candidate.exists():
            path = str(candidate)
    args.config_resolved = path

    base_cfg = config_from_args(args)  # only to read the base name
    base_name = base_cfg.name

    eps_list = [float(e) for e in str(args.eps).split(",")]
    cutoffs = [int(c) for c in str(args.cutoff).split(",")]
    if args.include_full:
        cutoffs = cutoffs + [None]  # None == full WB, no subspace constraint

    rows = []
    for eps in eps_list:
        for cutoff in cutoffs:
            rows.append(_run_point(args, eps, cutoff, base_name))

    out = Path(base_cfg.results_dir) / base_name / "tradeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote", out)
    print("plot: python scripts/plot.py tradeoff --file", out)


if __name__ == "__main__":
    main()
