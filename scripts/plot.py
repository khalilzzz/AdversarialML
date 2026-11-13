"""Redraw the paper's figures from our results.

    python scripts/plot.py sweep --file results/inset_RK_sweep_sweep.json --x ensemble --series restarts
    python scripts/plot.py alignment --case results/cifar10_baseline/case00.json
    python scripts/plot.py dp --file results/dp_sweep_dp_sweep.json
    python scripts/plot.py filtering --file results/filtering_filtering.json

Figures land in `results/figures/`. Which paper figure each subcommand corresponds to:

* `sweep`     -- the section 5.1 inset (success vs brewing time / R / K) and Fig. 12's two panels.
* `alignment` -- Fig. 2 and Fig. 11 (per-epoch gradient alignment, poisoned vs clean) and Fig. 5
                 (the eq. 4 bound, with the `a_k L ~= 1` reference line).
* `target`    -- Fig. 9 (target loss and accuracy under the adversarial and the original label).
* `dp`        -- Fig. 4b (poison success and validation accuracy against gradient noise).
* `filtering` -- Fig. 4a (feature-distance histograms) as a bar comparison of what filtering catches.
* `frequency` -- the piste-1 probe: radial DCT profile and per-Psi budget survival, WB vs WB-LF
                 (research extension, not a paper figure; see docs/FREQUENCY_PROBE.md).
* `tradeoff`  -- the E3 curve: poison success vs. low-frequency cutoff, and success against
                 detectability / visibility (research extension; see docs/FREQUENCY_PROBE.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/figures")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_sweep(args) -> None:
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    series = {}
    for row in rows:
        key = row["point"].get(args.series, "all") if args.series else "all"
        series.setdefault(key, []).append(row)
    for key, group in sorted(series.items(), key=lambda kv: str(kv[0])):
        group = sorted(group, key=lambda r: r["point"][args.x])
        x = [r["point"][args.x] for r in group]
        y = [r["avg_poison_success"] for r in group]
        err = [r["stderr"] for r in group]
        label = f"{args.series}={key}" if args.series else None
        axes[0].errorbar(x, y, yerr=err, marker="o", capsize=3, label=label)
        axes[1].errorbar(
            [r["mean_brew_minutes"] for r in group], y, yerr=err, marker="o", capsize=3, label=label
        )
    axes[0].set_xlabel(args.x)
    axes[0].set_ylabel("avg. poison success (%)")
    axes[1].set_xlabel("brewing time (minutes)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(alpha=0.3)
        if args.series:
            ax.legend()
    _save(fig, f"sweep_{args.x}.png")


def plot_alignment(args) -> None:
    data = json.loads(Path(args.case).read_text(encoding="utf-8"))
    history = data["victims"][0]["history"]
    diagnostics = history.get("diagnostics") or []
    if not diagnostics:
        raise SystemExit("this case was run without diagnostics (set diagnostics: true)")
    epochs = [d["epoch"] for d in diagnostics]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [d["cosine_adv"] for d in diagnostics], label="adversarial label")
    if "cosine_orig" in diagnostics[0]:
        axes[0].plot(epochs, [d["cosine_orig"] for d in diagnostics], "--", label="original label")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("avg. batch cosine similarity")
    axes[0].set_title("Fig. 2 / 11: gradient alignment")
    axes[0].legend()

    axes[1].plot(epochs, [d["bound_eq4"] for d in diagnostics], "r:", label="bound value")
    axes[1].axhline(1.0, color="black", lw=1, label=r"$\alpha_k L \approx 1$")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("bound value")
    axes[1].set_title("Fig. 5: eq. 4 during training")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    _save(fig, "alignment.png")


def plot_target(args) -> None:
    data = json.loads(Path(args.case).read_text(encoding="utf-8"))
    history = data["victims"][0]["history"]
    epochs = history["epochs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(epochs, history["target_adv_loss"], "r:", label="adversarial label loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy loss")
    axes[1].plot(epochs, history["target_adv_acc"], "r:", label="adversarial label")
    axes[1].plot(epochs, history["target_orig_acc"], "b-", label="original label")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("target accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Fig. 9: target loss and accuracy during victim training")
    _save(fig, "target_curves.png")


def plot_dp(args) -> None:
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
    rows = sorted(rows, key=lambda r: r["dp_noise"])
    noise = [max(r["dp_noise"], 1e-6) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        noise,
        [r["avg_poison_success"] for r in rows],
        yerr=[r["stderr"] for r in rows],
        marker="o",
        color="firebrick",
        capsize=3,
        label="poison success",
    )
    ax.set_xscale("log")
    ax.set_xlabel("gradient noise")
    ax.set_ylabel("poison success (%)", color="firebrick")
    twin = ax.twinx()
    twin.plot(noise, [r["avg_valid_acc"] for r in rows], marker="s", color="navy", label="validation accuracy")
    twin.set_ylabel("validation accuracy (%)", color="navy")
    ax.axhline(15, color="gray", ls=":", lw=1)
    ax.set_title("Fig. 4b: differential privacy trade-off")
    ax.grid(alpha=0.3)
    _save(fig, "dp_tradeoff.png")


def plot_filtering(args) -> None:
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))
    fractions = [f["filtering"] for f in rows[0]["filters"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    for offset, key, label in (
        (-width / 2, "poisons_removed", "poisons removed (detector)"),
        (width / 2, "poisons_removed_random", "poisons removed (random)"),
    ):
        values = [
            sum(case["filters"][i][key] for case in rows) / len(rows) for i in range(len(fractions))
        ]
        ax.bar([i + offset for i in range(len(fractions))], values, width, label=label)
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{int(f * 100)}% filtering" for f in fractions])
    ax.set_ylabel("images removed")
    ax.set_title("Table 6: filtering is close to random guessing")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "filtering.png")


def plot_frequency(args) -> None:
    """Radial DCT profile and per-Psi survival; overlays WB-LF when a second file is given."""
    files = [("WB", args.file)] + ([("WB-LF", args.lowfreq)] if args.lowfreq else [])
    data = {label: json.loads(Path(path).read_text(encoding="utf-8")) for label, path in files}

    fig, (ax_spec, ax_surv) = plt.subplots(1, 2, figsize=(11, 4))
    for label, d in data.items():
        ax_spec.plot(d["radial_dct_profile"], label=label)
    ax_spec.set_xlabel("radial DCT frequency bin")
    ax_spec.set_ylabel("fraction of perturbation energy")
    ax_spec.set_title("Where the budget is spent")
    ax_spec.legend()
    ax_spec.grid(alpha=0.3)

    psis = list(next(iter(data.values()))["per_psi"].keys())
    width = 0.8 / len(data)
    for k, (label, d) in enumerate(data.items()):
        values = [d["per_psi"][p]["budget_survival"] for p in psis]
        ax_surv.bar([i + k * width for i in range(len(psis))], values, width, label=label)
    ax_surv.set_xticks([i + width * (len(data) - 1) / 2 for i in range(len(psis))])
    ax_surv.set_xticklabels(psis)
    ax_surv.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax_surv.set_ylabel("budget survival (fraction)")
    ax_surv.set_title("Budget surviving each purifier")
    ax_surv.legend()
    ax_surv.grid(alpha=0.3, axis="y")
    _save(fig, "frequency.png")


def plot_tradeoff(args) -> None:
    """E3: success vs low-frequency cutoff, and success against detectability and visibility."""
    rows = json.loads(Path(args.file).read_text(encoding="utf-8"))

    def cutoff_x(r):
        return r["cutoff"] if r["cutoff"] is not None else max(
            (x["cutoff"] for x in rows if x["cutoff"] is not None), default=0
        ) + 4  # draw the full-WB point just past the largest cutoff

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(11, 4))

    by_eps: dict = {}
    for r in rows:
        by_eps.setdefault(r["eps"], []).append(r)
    for eps, group in sorted(by_eps.items()):
        group = sorted(group, key=cutoff_x)
        ax_c.errorbar(
            [cutoff_x(r) for r in group],
            [r["success"] for r in group],
            yerr=[r.get("stderr", 0) for r in group],
            marker="o", capsize=3, label=f"eps={eps:g}",
        )
    ax_c.set_xlabel("low-frequency cutoff (full WB = rightmost)")
    ax_c.set_ylabel("poison success (%)")
    ax_c.set_title("Success vs frequency cutoff")
    ax_c.legend()
    ax_c.grid(alpha=0.3)

    # Success against the two costs: detectability (off-manifold energy) and visibility (1 - SSIM).
    ax_d.scatter(
        [r["offmanifold_hf"] for r in rows], [r["success"] for r in rows],
        c="firebrick", label="detectability (off-manifold)",
    )
    twin = ax_d.twiny()
    twin.scatter(
        [r["visibility"] for r in rows], [r["success"] for r in rows],
        c="navy", marker="s", label="visibility (1 - SSIM)",
    )
    ax_d.set_xlabel("off-manifold energy fraction", color="firebrick")
    twin.set_xlabel("visibility (1 - SSIM)", color="navy")
    ax_d.set_ylabel("poison success (%)")
    ax_d.set_title("Success vs detectability / visibility")
    ax_d.grid(alpha=0.3)
    _save(fig, "tradeoff.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sweep")
    p.add_argument("--file", required=True)
    p.add_argument("--x", required=True)
    p.add_argument("--series", default=None)
    p.set_defaults(func=plot_sweep)

    p = sub.add_parser("alignment")
    p.add_argument("--case", required=True)
    p.set_defaults(func=plot_alignment)

    p = sub.add_parser("target")
    p.add_argument("--case", required=True)
    p.set_defaults(func=plot_target)

    p = sub.add_parser("dp")
    p.add_argument("--file", required=True)
    p.set_defaults(func=plot_dp)

    p = sub.add_parser("filtering")
    p.add_argument("--file", required=True)
    p.set_defaults(func=plot_filtering)

    p = sub.add_parser("frequency")
    p.add_argument("--file", required=True, help="frequency_probe.json (WB)")
    p.add_argument("--lowfreq", default=None, help="a second frequency_probe.json (WB-LF) to overlay")
    p.set_defaults(func=plot_frequency)

    p = sub.add_parser("tradeoff")
    p.add_argument("--file", required=True, help="tradeoff.json from scripts/tradeoff.py")
    p.set_defaults(func=plot_tradeoff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
