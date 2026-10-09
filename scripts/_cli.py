"""Shared argument parsing for the scripts.

`--set key=value` is the only way to change a configuration field from the command line, and every
value passed that way is recorded in the result JSON under `overrides` and flips the run's
`paper_setting` flag to false. That keeps a cheap partial run from being mistaken later for a
reproduction of a published number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from witchesbrew.config import Config, load_config  # noqa: E402


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="YAML config, layered on base.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field (repeatable); marks the run as a non-paper setting",
    )
    parser.add_argument("--no-resume", action="store_true", help="recompute cases already on disk")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    path = args.config
    if path and not Path(path).exists():
        candidate = ROOT / "configs" / path
        if candidate.exists():
            path = candidate
    return load_config(path, overrides)


def announce(cfg: Config) -> None:
    print(f"experiment : {cfg.name}")
    print(f"dataset    : {cfg.dataset}   net {cfg.net} -> victim {cfg.victim_net}")
    print(f"threat     : budget {cfg.budget:.2%}  eps {cfg.eps:g}  targets {cfg.targets}")
    print(f"brewing    : {cfg.recipe}/{cfg.objective}  R={cfg.restarts} M={cfg.attack_steps} K={cfg.ensemble}")
    print(f"victims    : {cfg.victim_runs} x {cfg.epochs} epochs")
    if cfg.overrides:
        print(f"overrides  : {cfg.overrides}  (NOT the published protocol)")
