"""Run one experiment: brew and evaluate every published case, then print the aggregate.

    python scripts/run_experiment.py --config cifar10/baseline.yaml

Results are written per case under `results/<name>/caseNN.json`, so an interrupted sweep resumes
where it stopped. The printed aggregate is the paper's `avg. poison success`: the mean over cases of
the per-case success rate, with the standard error of the case values (appendix B).
"""

from __future__ import annotations

from _cli import announce, build_parser, config_from_args

from witchesbrew.evaluate import run_experiment


def main() -> None:
    parser = build_parser(__doc__ or "")
    args = parser.parse_args()
    cfg = config_from_args(args)
    announce(cfg)

    summary = run_experiment(cfg, resume=not args.no_resume)

    print()
    print(f"cases evaluated     : {summary['cases']}")
    print(f"avg. poison success : {summary['avg_poison_success']:.2f}% (+-{summary['stderr']:.2f})")
    print(f"clean validation acc: {summary['avg_valid_acc']:.2f}% (+-{summary['valid_acc_stderr']:.2f})")
    print(f"mean brewing time   : {summary['mean_brew_minutes']:.1f} min")
    if not summary["paper_setting"]:
        print("note: at least one case used overrides, so this is not the published protocol")


if __name__ == "__main__":
    main()
