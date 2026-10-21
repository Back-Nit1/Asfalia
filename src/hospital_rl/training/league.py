"""Bounded opponent league: a training pool (sampled for updates) and a disjoint evaluation pool
(never sampled for updates, never used for checkpoint selection), plus cross-play matrices."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class Entry:
    id: str
    kind: Literal["scripted", "checkpoint"]
    path: str | None = None  # checkpoint path; scripted entries are resolved by id by the caller


class League:
    """Scripted entries are permanent. Checkpoint entries are bounded FIFO (oldest evicted).

    Sampling: p_scripted to a uniform scripted entry; otherwise p_latest to the newest checkpoint,
    else uniform over older checkpoints. Missing classes fall back to what exists."""

    def __init__(self, max_checkpoints: int = 8, p_scripted: float = 0.3, p_latest: float = 0.5):
        self.max_checkpoints, self.p_scripted, self.p_latest = max_checkpoints, p_scripted, p_latest
        self.scripted: list[Entry] = []
        self.checkpoints: list[Entry] = []
        self._eval: list[Entry] = []
        self.evictions: list[dict] = []   # {"id", "after_adds"}: FIFO history, persisted in the manifest
        self.adds = 0                      # checkpoint entries ever added to the train pool
        self.provenance: dict = {}         # selection provenance, set by the caller

    @property
    def eval_pool(self) -> list[Entry]:
        return list(self._eval)

    @property
    def train_pool(self) -> list[Entry]:
        return self.scripted + self.checkpoints

    def add_train(self, e: Entry) -> None:
        if any(x.id == e.id for x in self._eval):
            raise ValueError(f"{e.id} is in the evaluation pool; it may not be used for updates")
        pool = self.scripted if e.kind == "scripted" else self.checkpoints
        if all(x.id != e.id for x in pool):
            pool.append(e)
            self.adds += e.kind == "checkpoint"
        while len(self.checkpoints) > self.max_checkpoints:
            self.evictions.append({"id": self.checkpoints.pop(0).id, "after_adds": self.adds})

    def add_eval(self, e: Entry) -> None:
        if any(x.id == e.id for x in self.train_pool):
            raise ValueError(f"{e.id} is in the training pool; evaluation entries must be disjoint")
        if all(x.id != e.id for x in self._eval):
            self._eval.append(e)

    def to_dict(self) -> dict:
        return {"version": 1, "max_checkpoints": self.max_checkpoints, "p_scripted": self.p_scripted,
                "p_latest": self.p_latest, "adds": self.adds, "train": [asdict(e) for e in self.train_pool],
                "eval": [asdict(e) for e in self._eval], "evictions": self.evictions, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: dict) -> "League":
        lg = cls(d["max_checkpoints"], d["p_scripted"], d["p_latest"])
        for e in d["train"]:
            lg.add_train(Entry(**e))
        for e in d["eval"]:
            lg.add_eval(Entry(**e))
        lg.evictions, lg.adds, lg.provenance = d["evictions"], d["adds"], d["provenance"]
        return lg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "League":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def sample(self, rng: random.Random) -> Entry:
        if not self.train_pool:
            raise LookupError("empty league")
        if self.scripted and (not self.checkpoints or rng.random() < self.p_scripted):
            return rng.choice(self.scripted)
        if len(self.checkpoints) == 1 or rng.random() < self.p_latest:
            return self.checkpoints[-1]
        return rng.choice(self.checkpoints[:-1])


def cross_play(evaluate: Callable[[str, str], dict[str, float]], attackers: list[str], defenders: list[str],
               metric: str) -> tuple[np.ndarray, dict[tuple[str, str], dict[str, float]]]:
    """matrix[i, j] = metric of attackers[i] vs defenders[j]; also returns every metric per pair.
    Rows/cols with cyclic strengths show up as non-transitive structure; read the whole matrix."""
    full = {(a, d): evaluate(a, d) for a in attackers for d in defenders}
    return np.array([[full[a, d][metric] for d in defenders] for a in attackers]), full
