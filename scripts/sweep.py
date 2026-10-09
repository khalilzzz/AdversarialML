"""Run a grid of experiments — the mechanism behind the paper's sweep figures and tables.

    # section 5.1 inset: restarts x ensemble size
    python scripts/sweep.py --config cifar10/inset_RK_sweep.yaml --grid restarts=1,8 ensemble=1,2,4,8

    # Fig. 12 left: thin ResNets, cosine vs Euclidean
    python scripts/sweep.py --config cifar10/fig12_width.yaml --grid width=4,8,16,32,48,64,96 objective=cosine,euclidean

    # Fig. 12 right: pretraining effort
    python scripts/sweep.py --config cifar10/fig12_pretrain.yaml --grid pretrain_epochs=0,5,10,20,30,40 eps=8,16

    # Table 9: multiple targets
    python scripts/sweep.py --config cifar10/table9_multitarget.yaml --grid targets=1,2,4,8,16 budget=0.01,0.04

Each point gets its own `results/<name>_<key><value>/` directory, so the grid is resumable and can
be extended later without recomputing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from _cli import announce, build_parser, config_from_args

from witchesbrew.config import variants
from witchesbrew.evaluate import run_experiment


def parse_grid(items: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--grid expects KEY=V1,V2,..., got {item!r}")
        key, raw = item.split("=", 1)
        values = []
        for piece in raw.split(","):
            piece = piece.strip()
            if piece.lower() in {"true", "false"}:
                values.append(piece.lower() == "true")
            else:
                try:
                    values.append(int(piece))
                except ValueError:
                    try:
                        values.append(float(piece))
                    except ValueError:
                        values.append(piece)
        grid[key.strip()] = values
    return grid


def main() -> None:
    parser = build_parser(__doc__ or "")
    parser.add_argument("--grid", nargs="+", required=True, metavar="KEY=V1,V2")
    args = parser.parse_args()
    base = config_from_args(args)
    grid = parse_grid(args.grid)

    points = variants(base, grid)
    print(f"{len(points)} grid points\n")
    summaries = []
    for cfg in points:
        announce(cfg)
        summary = run_experiment(cfg, resume=not args.no_resume)
        summary["point"] = {key: getattr(cfg, key) for key in grid}
        summaries.append(summary)
        print(
            f"-> {summary['avg_poison_success']:.2f}% (+-{summary['stderr']:.2f}), "
            f"{summary['mean_brew_minutes']:.1f} min brewing\n"
        )

    out = Path(base.results_dir) / f"{base.name}_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
