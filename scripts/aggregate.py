"""Turn the result JSONs into the paper's tables, side by side with the published numbers.

    python scripts/aggregate.py                 # every experiment found under results/
    python scripts/aggregate.py --table table2  # one table
    python scripts/aggregate.py --csv out.csv

The published values are hard-coded here from the paper so a run can be compared at a glance; the
`delta` column is ours minus theirs, in percentage points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from witchesbrew.utils import mean_and_stderr  # noqa: E402

# Published avg. poison success (%) keyed by experiment name, from the paper's tables.
PAPER_VALUES = {
    "cifar10_baseline": 91.25,
    "table1_K1_noaug_victim_noaug": 100.00,
    "table1_K1_noaug_victim_aug": 32.50,
    "table1_K8_noaug_victim_noaug": 78.75,
    "table1_K1_diffaug_victim_aug": 91.25,
    "table2_resnet18_gm": 90.00,
    "table2_resnet18_bullseye": 3.75,
    "table2_resnet18_frogs": 1.25,
    "table2_convnet_gm": 86.25,
    "table2_convnet_bullseye": 78.75,
    "table2_convnet_frogs": 52.50,
    "table8_cw": 77.50,
    "table8_m50": 40.00,
    "table8_euclidean": 61.25,
    "table9_multitarget": 36.25,  # the (4 targets, 1% budget) cell
}

# Published clean validation accuracy (%) for the Table 10 rows.
PAPER_VALID_ACC = {"cifar10_baseline": 92.20, "table1_K1_diffaug_victim_aug": 92.20}


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for experiment in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        cases = sorted(experiment.glob("case*.json"))
        if not cases:
            continue
        successes, accuracies, minutes, paper_setting = [], [], [], True
        for file in cases:
            data = json.loads(file.read_text(encoding="utf-8"))
            successes.append(data["case_success"])
            accuracies.append(data["mean_valid_acc"])
            minutes.append(data["brew_minutes"])
            paper_setting = paper_setting and data.get("paper_setting", False)
        mean, stderr = mean_and_stderr(successes)
        acc_mean, _ = mean_and_stderr(accuracies)
        name = experiment.name
        rows.append(
            {
                "experiment": name,
                "cases": len(cases),
                "ours_success_%": round(100 * mean, 2),
                "ours_stderr": round(100 * stderr, 2),
                "paper_success_%": PAPER_VALUES.get(name),
                "delta_pp": (
                    round(100 * mean - PAPER_VALUES[name], 2) if name in PAPER_VALUES else None
                ),
                "ours_valid_acc_%": round(100 * acc_mean, 2),
                "paper_valid_acc_%": PAPER_VALID_ACC.get(name),
                "brew_min_mean": round(sum(minutes) / len(minutes), 1),
                "paper_protocol": paper_setting,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--table", type=str, default=None, help="substring filter on the name")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    directory = Path(args.results)
    if not directory.exists():
        raise SystemExit(f"no results directory at {directory}")
    frame = collect(directory)
    if frame.empty:
        raise SystemExit("no completed cases found")
    if args.table:
        frame = frame[frame["experiment"].str.contains(args.table)]

    pd.set_option("display.width", 200)
    print(frame.to_string(index=False))
    incomplete = frame[frame["cases"] < 10]
    if not incomplete.empty:
        print(
            "\nnote: experiments with fewer than 10 cases are partial; the paper averages over 10 "
            "poison-target settings (appendix B)"
        )
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
