"""Seeding, device selection, result bookkeeping."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Appendix B fixes seeds so methods are compared on the same cases."""
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        # Reference implementation runs with cudnn.benchmark enabled (consts.py BENCHMARK = True).
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def mean_and_stderr(values: list[float]) -> tuple[float, float]:
    """Mean and standard error over the per-case averages, as appendix B defines the error bars."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


class ResultStore:
    """One JSON file per (experiment, case). Lets a long sweep resume after an interruption."""

    def __init__(self, root: str | Path, experiment: str) -> None:
        self.dir = Path(root) / experiment
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, case_id: int) -> Path:
        return self.dir / f"case{case_id:02d}.json"

    def done(self, case_id: int) -> bool:
        return self.path(case_id).exists()

    def write(self, case_id: int, payload: dict[str, Any]) -> Path:
        target = self.path(case_id)
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def read_all(self) -> list[dict[str, Any]]:
        out = []
        for file in sorted(self.dir.glob("case*.json")):
            out.append(json.loads(file.read_text(encoding="utf-8")))
        return out


class Timer:
    """Wall-clock timing; the paper reports brewing time in the section 5.1 inset figure."""

    def __init__(self) -> None:
        self.start = time.time()

    def minutes(self) -> float:
        return (time.time() - self.start) / 60.0


def available_workers(requested: int) -> int:
    cpu = os.cpu_count() or 1
    return max(0, min(requested, cpu - 1))
