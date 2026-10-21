"""Durable run states and the legal transition table. The controller is the only writer; nothing else may set a state."""
from __future__ import annotations

from enum import Enum


class RunState(str, Enum):
    queued = "queued"
    starting = "starting"
    running = "running"
    draining = "draining"
    checkpointing = "checkpointing"
    stopped = "stopped"
    failed = "failed"
    cancelled = "cancelled"
    completed = "completed"


S = RunState
ACTIVE = {S.starting, S.running, S.draining, S.checkpointing}
TERMINAL = {S.stopped, S.failed, S.cancelled, S.completed}
RESUMABLE = {S.stopped, S.failed, S.cancelled}      # completed runs have nothing left to resume

LEGAL: dict[RunState, set[RunState]] = {
    S.queued: {S.starting, S.cancelled},
    S.starting: {S.running, S.draining, S.checkpointing, S.failed, S.cancelled, S.stopped, S.completed},
    S.running: {S.draining, S.checkpointing, S.stopped, S.failed, S.cancelled, S.completed},
    S.draining: {S.stopped, S.failed, S.cancelled, S.completed},
    S.checkpointing: {S.stopped, S.failed, S.cancelled, S.completed},
    S.stopped: {S.queued}, S.failed: {S.queued}, S.cancelled: {S.queued}, S.completed: set(),
}


class IllegalRunTransition(ValueError):
    pass


def check(src: RunState, dst: RunState) -> None:
    if dst not in LEGAL[src]:
        raise IllegalRunTransition(f"run cannot go {src.value} -> {dst.value}")
