"""Piste 2: Witches' Brew against Deep Partition Aggregation (research extension, NOT the paper).

    # H1 reference: the hash spreads the poisons, no placement
    python scripts/run_dpa.py --config piste2/dpa_imagenette.yaml --set name=dpa_uniform
    # hash-aware attacker: budget concentrated on the 3 weakest partitions
    python scripts/run_dpa.py --config piste2/dpa_imagenette.yaml \
           --set dpa_attack_partitions=3 --set dpa_pick=weakest --set name=dpa_conc3
    # counter-measure: secret hash key, placement impossible
    python scripts/run_dpa.py --config piste2/dpa_imagenette.yaml \
           --set dpa_hash=secret --set dpa_secret_key=42 --set name=dpa_secret

Results are written per case under `results/<name>/caseNN.json` (resumable). The printed aggregate is
the success of the *aggregate* classifier on the target, the flip rate of the poisoned base models
(the measured p(rho) of the allocation theorem) and the clean / poisoned vote margins.
"""

from __future__ import annotations

from _cli import announce, build_parser, config_from_args

from witchesbrew.evaluate import run_dpa_experiment


def main() -> None:
    parser = build_parser(__doc__ or "")
    args = parser.parse_args()
    cfg = config_from_args(args)
    if cfg.dpa_k <= 0:
        raise SystemExit("set dpa_k > 0 (number of partitions)")
    announce(cfg)
    print(
        f"dpa        : k={cfg.dpa_k} hash={cfg.dpa_hash} attacked={cfg.dpa_attack_partitions} "
        f"pick={cfg.dpa_pick} partition_surrogate={cfg.dpa_partition_surrogate}"
    )
    summary = run_dpa_experiment(cfg, resume=not args.no_resume)
    print()
    print(f"cases evaluated       : {summary['cases']}")
    print(f"DPA poison success    : {summary['avg_poison_success']:.2f}% (+-{summary['stderr']:.2f})")
    print(f"base-model flip rate  : {summary['avg_flip_rate']:.2f}% (+-{summary['flip_rate_stderr']:.2f})")
    print(f"vote margin clean/pois: {summary['avg_clean_margin']:.1f} / {summary['avg_poisoned_margin']:.1f}")
    print(f"DPA validation acc    : {summary['avg_valid_acc']:.2f}% (+-{summary['valid_acc_stderr']:.2f})")
    if summary["avg_single_victim_success"] is not None:
        print(f"single victim success : {summary['avg_single_victim_success']:.2f}%")
    print(f"mean brewing time     : {summary['mean_brew_minutes']:.1f} min")


if __name__ == "__main__":
    main()
