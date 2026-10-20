"""Benchmark splits over topology family, workload profile, and seed range (never seeds alone)."""
from __future__ import annotations

import hashlib
import json

from ..scenarios import generate
from ..schemas import ScenarioSpec

SPLITS = {
    "train": {"families": ["hub", "tiered"], "workloads": ["day", "night"], "seeds": range(0, 10_000)},
    "validation": {"families": ["flat"], "workloads": ["day", "surge"], "seeds": range(10_000, 11_000)},
    "test": {"families": ["segmented"], "workloads": ["surge", "night"], "seeds": range(20_000, 21_000)},
}
SPLIT_VERSION = 1


def scenario_for(split: str, seed: int, size: int = 8, difficulty: float = 0.5, horizon: int = 40) -> ScenarioSpec:
    cfg = SPLITS[split]
    if seed not in cfg["seeds"]:
        raise ValueError(f"seed {seed} outside {split} seed range")
    fam = cfg["families"][seed % len(cfg["families"])]
    wl = cfg["workloads"][(seed // len(cfg["families"])) % len(cfg["workloads"])]
    return generate(fam, size, seed, difficulty, wl, horizon)


def split_hash() -> str:
    blob = json.dumps({"v": SPLIT_VERSION, **{k: {"f": v["families"], "w": v["workloads"],
                       "s": [v["seeds"].start, v["seeds"].stop]} for k, v in SPLITS.items()}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
