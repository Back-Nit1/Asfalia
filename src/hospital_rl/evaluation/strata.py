"""Versioned benchmark strata over every evaluation dimension, hashed as one definition.

A split is only a name; a *stratum* pins topology family, workload profile, attack schedule, vulnerability
difficulty, telemetry-loss level, visibility mode and seed range. Any change to any field changes ``benchmark_hash``.
The ``test`` stratum is report-only: ``require_trainable`` rejects it for training, curricula, opponent selection and
checkpoint selection.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .splits import SPLITS

STRATA_VERSION = 1
ROLE = {"train": "train", "validation": "select", "test": "report_only"}   # test is never used to choose anything
# What the engine can run today. Values outside this set are valid *definitions* but not runnable (docs/requests/agent-F.md).
ENGINE_SUPPORT = {"telemetry_loss": (0.1,), "visibility_mode": ("partial",),
                  "attack_schedule": ("scripted_v1", "learned_frozen")}
DIFFICULTIES = (0.5, 0.7)


class StratumError(ValueError):
    pass


class Stratum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["train", "validation", "test"]
    topology_families: tuple[str, ...]
    workload_profiles: tuple[str, ...]
    attack_schedule: str = "scripted_v1"
    vulnerability_difficulty: tuple[float, ...] = DIFFICULTIES
    telemetry_loss: float = 0.1            # engine.telemetry.release default drop probability
    visibility_mode: Literal["partial", "full_diagnostic"] = "partial"
    seed_start: int
    seed_stop: int

    @property
    def role(self) -> str:
        return ROLE[self.name]

    def seeds(self) -> range:
        return range(self.seed_start, self.seed_stop)

    def runnable(self) -> list[str]:
        """Dimensions the current engine cannot execute (empty = runnable)."""
        return [f"{k}={getattr(self, k)!r}" for k, ok in ENGINE_SUPPORT.items() if getattr(self, k) not in ok]


class BenchmarkDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: int = STRATA_VERSION
    strata: tuple[Stratum, ...]

    def get(self, name: str) -> Stratum:
        return next(s for s in self.strata if s.name == name)

    def stratum_hash(self, name: str) -> str:
        return _h(self.get(name).model_dump(mode="json"))

    def hash(self) -> str:
        return _h(self.model_dump(mode="json"))

    def validate(self) -> None:
        """Strata must not share seeds, families or workloads with test (no leakage by overlap)."""
        names = [s.name for s in self.strata]
        if sorted(names) != ["test", "train", "validation"]:
            raise StratumError(f"need exactly train/validation/test strata, got {names}")
        test = self.get("test")
        for s in self.strata:
            if s.name == "test":
                continue
            if set(s.seeds()) & set(test.seeds()):
                raise StratumError(f"{s.name} and test share seeds")
            if set(s.topology_families) & set(test.topology_families):
                raise StratumError(f"{s.name} and test share topology families")   # family is the held-out axis
            if s.attack_schedule != test.attack_schedule:
                raise StratumError(f"{s.name} and test use different attack schedules")


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()[:16]


def default_definition() -> BenchmarkDefinition:
    return BenchmarkDefinition(strata=tuple(
        Stratum(name=k, topology_families=tuple(v["families"]), workload_profiles=tuple(v["workloads"]),
                seed_start=v["seeds"].start, seed_stop=v["seeds"].stop) for k, v in SPLITS.items()))


DEFINITION = default_definition()


def benchmark_hash() -> str:
    return DEFINITION.hash()


def require_trainable(split: str, use: str = "training") -> str:
    """Raise unless ``split`` may influence learning (training, curriculum, opponent or checkpoint selection)."""
    if split not in SPLITS:
        raise StratumError(f"unknown split {split!r}")
    if ROLE[split] == "report_only":
        raise StratumError(f"the {split!r} stratum is report-only; it cannot be used for {use}")
    return split


def check_training_config(cfg: Any, where: str = "config") -> None:
    """Reject any `split`/`*_split`/`splits` entry naming the test stratum anywhere in a (nested) config."""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "split" or str(k).endswith("_split") or k == "splits":
                for s in (v if isinstance(v, (list, tuple)) else [v]):
                    if s == "test":
                        raise StratumError(f"{where}.{k}: test stratum named in a training-side config")
            check_training_config(v, f"{where}.{k}")
    elif isinstance(cfg, (list, tuple)):
        for i, v in enumerate(cfg):
            check_training_config(v, f"{where}[{i}]")


def check_world(split: str, difficulty: float) -> None:
    if difficulty not in DEFINITION.get(split).vulnerability_difficulty:
        raise StratumError(f"world difficulty {difficulty} not in the {split} stratum "
                           f"{DEFINITION.get(split).vulnerability_difficulty}")
