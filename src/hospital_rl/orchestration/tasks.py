"""Task record and state machine (prompt section 14): queued -> ready -> running -> review -> passed -> integrated,
with failed / blocked / cancelled explicit."""
from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath

import yaml
from pydantic import BaseModel, Field

from .contracts import Evidence, PlanConfig, TaskSpec


class Status(str, Enum):
    queued = "queued"
    ready = "ready"
    running = "running"
    review = "review"
    passed = "passed"
    integrated = "integrated"
    failed = "failed"
    blocked = "blocked"
    cancelled = "cancelled"


S = Status
TRANSITIONS: dict[Status, set[Status]] = {
    S.queued: {S.ready, S.blocked, S.cancelled},
    S.ready: {S.running, S.blocked, S.cancelled},
    S.running: {S.review, S.failed, S.blocked, S.cancelled},
    S.review: {S.passed, S.failed, S.cancelled},
    S.passed: {S.integrated, S.failed, S.cancelled},   # failed: combined suite rejected the merge
    S.failed: {S.ready, S.blocked, S.cancelled},        # ready = bounded retry
    S.blocked: {S.ready, S.cancelled},                  # ready = explicit operator unblock
    S.integrated: set(),
    S.cancelled: set(),
}
ACTIVE = {S.running, S.review, S.passed}   # statuses that hold path ownership (passed until integrated)


class IllegalTransition(ValueError):
    pass


class Task(BaseModel):
    id: str
    role: str
    description: str
    owner: str = "agent"
    dependencies: list[str] = Field(default_factory=list)
    acceptance_commands: list[str] = Field(default_factory=list)
    assigned_paths: list[str] = Field(default_factory=list)
    prompt: str = ""
    base_commit: str = ""
    branch: str = ""
    worktree: str = ""
    status: Status = Status.queued
    heartbeat: float | None = None
    lease_expiry: float | None = None
    started_at: float | None = None
    attempt: int = 0            # attempts started so far
    session_id: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    outcome: str = ""           # concise, human-readable
    evidence: Evidence = Field(default_factory=Evidence)

    def move(self, to: Status) -> None:
        if to not in TRANSITIONS[self.status]:
            raise IllegalTransition(f"{self.id}: {self.status.value} -> {to.value}")
        self.status = to


def load_plan(path: str) -> tuple[PlanConfig, list[Task]]:
    raw = yaml.safe_load(open(path))
    cfg = PlanConfig(**raw.get("config", {}))
    tasks = [Task(**TaskSpec(**t).model_dump()) for t in raw["tasks"]]
    ids = {t.id for t in tasks}
    if len(ids) != len(tasks):
        raise ValueError("duplicate task ids")
    for t in tasks:
        if missing := set(t.dependencies) - ids:
            raise ValueError(f"{t.id}: unknown dependencies {sorted(missing)}")
    _assert_acyclic(tasks)
    return cfg, tasks


def _assert_acyclic(tasks: list[Task]) -> None:
    deps, seen, stack = {t.id: t.dependencies for t in tasks}, set(), set()

    def visit(n: str) -> None:
        if n in stack:
            raise ValueError(f"dependency cycle through {n}")
        if n in seen:
            return
        stack.add(n)
        for d in deps[n]:
            visit(d)
        stack.discard(n)
        seen.add(n)

    for n in deps:
        visit(n)


def _norm(p: str) -> str:
    """'src/x/**' and 'src/x/' and 'src/x' all own the directory prefix; plain files own themselves."""
    p = p.replace("\\", "/").removesuffix("/**").removesuffix("/*").rstrip("/")
    return str(PurePosixPath(p))


def paths_overlap(a: list[str], b: list[str]) -> bool:
    for x in map(_norm, a):
        for y in map(_norm, b):
            if x == y or x.startswith(y + "/") or y.startswith(x + "/"):
                return True
    return False


def owns(task_paths: list[str], file: str) -> bool:
    return any(file == p or file.startswith(p + "/") for p in map(_norm, task_paths))
