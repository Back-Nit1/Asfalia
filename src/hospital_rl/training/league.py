"""Bounded opponent league: a training pool (sampled for updates) and a disjoint evaluation pool
(never sampled for updates, never used for checkpoint selection), plus cross-play matrices."""
from __future__ import annotations

import random
from dataclasses import dataclass
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
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def add_eval(self, e: Entry) -> None:
        if any(x.id == e.id for x in self.train_pool):
            raise ValueError(f"{e.id} is in the training pool; evaluation entries must be disjoint")
        if all(x.id != e.id for x in self._eval):
            self._eval.append(e)

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
