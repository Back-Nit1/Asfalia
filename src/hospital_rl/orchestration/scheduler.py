"""Dependency-, lease-, ownership- and concurrency-aware task scheduler.

`tick()` is the only place that advances time-driven state; everything else is an explicit operator action.
It runs under one lock, so acceptance commands block API mutations while they execute.
# ponytail: synchronous acceptance runs; move them to a worker thread if suites get long.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from ..api.storage import Store
from . import integration as git_ops
from .adapters import NOT_CONFIGURED, CodingAgentAdapter
from .contracts import PlanConfig
from .events import Emitter, tail
from .tasks import ACTIVE, Status, Task, paths_overlap

DEP_DEAD = {Status.blocked, Status.cancelled}
TERMINAL = {Status.integrated, Status.cancelled}


class Scheduler:
    def __init__(self, store: Store, cfg: PlanConfig, tasks: list[Task] | None = None,
                 adapter: CodingAgentAdapter | None = None, clock: Callable[[], float] = time.time):
        self.store, self.cfg, self.adapter, self.clock = store, cfg, adapter, clock
        self.emit = Emitter(store)
        self.lock = threading.RLock()
        for t in tasks or []:    # a restart keeps persisted state; only unseen tasks are inserted
            if store.get_task(t.id) is None:
                self._save(t)

    # ---- persistence helpers ----
    def _save(self, t: Task) -> None:
        self.store.put_task(t.id, t.status.value, t.model_dump(mode="json"))

    def get(self, task_id: str) -> Task:
        raw = self.store.get_task(task_id)
        if raw is None:
            raise KeyError(task_id)
        return Task.model_validate(raw)

    def all(self) -> list[Task]:
        return [Task.model_validate(r) for r in self.store.list_tasks()]

    def _set(self, t: Task, to: Status, outcome: str | None = None) -> None:
        frm = t.status
        t.move(to)
        if outcome is not None:
            t.outcome = outcome
        self._save(t)
        self.emit.emit("task.state_changed", t.id, **{"from": frm.value, "to": to.value, "outcome": t.outcome,
                                                       "attempt": t.attempt})

    # ---- queries ----
    def runtime_status(self) -> dict:
        if self.adapter is None:
            return {"configured": False, "name": None, "detail": NOT_CONFIGURED}
        return self.adapter.capabilities().model_dump()

    def path_conflicts(self, t: Task, tasks: list[Task] | None = None) -> list[str]:
        return [o.id for o in (tasks or self.all()) if o.id != t.id and o.status in ACTIVE
                and paths_overlap(t.assigned_paths, o.assigned_paths)]

    # ---- lifecycle ----
    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            self._expire(now)
            self._poll_agents(now)
            for t in self.all():
                if t.status is Status.review:
                    self._review(t)
            self._promote()
            self._start_ready(now)

    def _expire(self, now: float) -> None:
        for t in self.all():
            if t.status is not Status.running:
                continue
            if t.lease_expiry is not None and now > t.lease_expiry:
                self._stop_session(t)
                self._fail(t, "lease expired (no heartbeat)")
            elif t.started_at is not None and now - t.started_at > self.cfg.task_timeout_seconds:
                self._stop_session(t)
                self._fail(t, f"timed out after {self.cfg.task_timeout_seconds:.0f}s")

    def _poll_agents(self, now: float) -> None:
        for t in self.all():
            if t.status is not Status.running or t.session_id is None or self.adapter is None:
                continue
            try:
                events = self.adapter.drain_events(t.session_id)
                res = self.adapter.result(t.session_id)
            except KeyError:    # session unknown to this process (backend restarted): lease will expire
                continue
            for e in events:
                self.emit.emit("task.agent_event", t.id, kind=e.kind, summary=e.summary)
            if events:
                self._beat(t, now)
            if res is not None:
                t.evidence.agent_summary = res.summary
                if not res.ok:
                    self._fail(t, f"agent exited {res.exit_code}: {tail(res.stderr_tail, 300)}")
                else:
                    self._to_review(t)

    def _to_review(self, t: Task) -> None:
        if t.worktree:
            files = git_ops.changed_files(t)
            t.evidence.changed_files = files
            t.evidence.out_of_scope_files = git_ops.out_of_scope(t, files)
            patch = Path(t.worktree).parent / "_artifacts" / t.id / f"attempt{t.attempt}.patch"
            patch.parent.mkdir(parents=True, exist_ok=True)
            patch.write_text(git_ops.diff(t))
            ref = f"{t.id}-attempt{t.attempt}-patch"
            self.store.add_artifact({"id": ref, "kind": "diff", "task_id": t.id, "path": str(patch),
                                     "created_at": _iso(self.clock())})
            t.artifact_refs.append(ref)
        self._set(t, Status.review, "awaiting orchestrator-run acceptance")

    def _review(self, t: Task) -> None:
        """passed only if the orchestrator itself ran every acceptance command and all exited 0."""
        if t.evidence.out_of_scope_files:
            return self._fail(t, f"changed files outside assigned paths: {t.evidence.out_of_scope_files}")
        if not t.acceptance_commands:
            return self._fail(t, "no acceptance commands defined; refusing to pass on the agent's word")
        results = [git_ops.run_command(c, t.worktree or ".", self.cfg.acceptance_timeout_seconds)
                   for c in t.acceptance_commands]
        t.evidence.test_results = results
        self.emit.emit("task.evidence", t.id, tests=[{"command": r.command, "exit_code": r.exit_code} for r in results])
        bad = [r.command for r in results if r.exit_code]
        if bad:
            self._fail(t, f"acceptance failed: {bad}")
        else:
            self._set(t, Status.passed, "acceptance commands passed")

    def _fail(self, t: Task, reason: str) -> None:
        self._set(t, Status.failed, reason)
        if t.attempt <= self.cfg.retry_cap:   # attempt 1 is the original; retries are attempts 2..cap+1
            self._set(t, Status.ready, f"retrying after: {reason}")
        else:
            self._set(t, Status.blocked, f"retry cap ({self.cfg.retry_cap}) exhausted; last failure: {reason}")

    def _promote(self) -> None:
        tasks = {t.id: t for t in self.all()}
        for t in tasks.values():
            if t.status is not Status.queued:
                continue
            deps = [tasks[d] for d in t.dependencies]
            if any(d.status in DEP_DEAD for d in deps):
                dead = [d.id for d in deps if d.status in DEP_DEAD]
                self._set(t, Status.blocked, f"dependency {dead} {'/'.join(sorted({tasks[d].status.value for d in dead}))}")
            elif all(d.status is Status.integrated for d in deps):
                self._set(t, Status.ready)

    def _start_ready(self, now: float) -> None:
        tasks = self.all()
        running = sum(t.status is Status.running for t in tasks)
        for t in tasks:
            if running >= self.cfg.max_concurrency:
                break
            if t.status is not Status.ready or t.owner != "agent" or self.adapter is None:
                continue
            if self.path_conflicts(t, self.all()):
                continue
            if not self._begin(t, now):
                continue
            try:
                t.session_id = self.adapter.submit(t, Path(t.worktree))
            except Exception as e:   # runtime refused to start: counts as a failed attempt
                self._fail(t, f"submit failed: {tail(str(e), 300)}")
                continue
            self._save(t)
            running += 1

    def _begin(self, t: Task, now: float) -> bool:
        try:
            git_ops.create_worktree(self.cfg.repo, self.cfg.worktree_root, t, self.cfg.base_commit)
        except RuntimeError as e:
            self._set(t, Status.blocked, f"worktree setup failed: {tail(str(e), 300)}")
            return False
        t.attempt += 1
        t.started_at, t.heartbeat, t.lease_expiry = now, now, now + self.cfg.lease_seconds
        t.session_id = None
        self._set(t, Status.running, f"attempt {t.attempt}")
        return True

    def _beat(self, t: Task, now: float) -> None:
        t.heartbeat, t.lease_expiry = now, now + self.cfg.lease_seconds
        self._save(t)
        self.emit.emit("task.heartbeat", t.id)

    def _stop_session(self, t: Task) -> None:
        if self.adapter and t.session_id:
            try:
                self.adapter.cancel(t.session_id)
            except KeyError:
                pass

    # ---- operator / human actions ----
    def heartbeat(self, task_id: str) -> Task:
        with self.lock:
            t = self.get(task_id)
            if t.status is not Status.running:
                raise ValueError(f"{task_id} is {t.status.value}, not running")
            self._beat(t, self.clock())
            return t

    def claim(self, task_id: str) -> Task:
        """A human starts a human-owned task (no agent runtime involved)."""
        with self.lock:
            t = self.get(task_id)
            if t.owner != "human":
                raise ValueError("only human-owned tasks can be claimed")
            if t.status is not Status.ready:
                raise ValueError(f"{task_id} is {t.status.value}, not ready")
            if conflicts := self.path_conflicts(t):
                raise ValueError(f"path ownership conflict with {conflicts}")
            if not self._begin(t, self.clock()):
                raise ValueError(self.get(task_id).outcome)
            return t

    def submit_for_review(self, task_id: str) -> Task:
        with self.lock:
            t = self.get(task_id)
            if t.status is not Status.running or t.owner != "human":
                raise ValueError("only running human-owned tasks can be submitted manually")
            self._to_review(t)
            return t

    def cancel(self, task_id: str) -> Task:
        with self.lock:
            t = self.get(task_id)
            self._stop_session(t)
            self._set(t, Status.cancelled, "cancelled by operator")
            self._promote()    # dependents become blocked
            return t

    def retry(self, task_id: str) -> Task:
        """Operator retry of a failed/blocked task; resets the attempt budget (an explicit, audited human decision)."""
        with self.lock:
            t = self.get(task_id)
            t.attempt = 0
            self._set(t, Status.ready, "retry requested by operator")
            return t

    def integrate(self, task_id: str) -> Task:
        """Merge a passed task into the integration branch and run the combined suite; reverts on failure."""
        with self.lock:
            t = self.get(task_id)
            if t.status is not Status.passed:
                raise ValueError(f"{task_id} is {t.status.value}; only passed tasks can be integrated")
            if not self.cfg.integration_repo:
                raise ValueError("integration_repo not configured")
            results = git_ops.integrate(self.cfg.integration_repo, t, self.cfg.integration_branch, self.cfg.combined_suite,
                                        self.cfg.acceptance_timeout_seconds)
            if any(r.exit_code for r in results):
                t.evidence.test_results += results
                self._fail(t, "combined suite failed after merge; merge reverted")
            else:
                self._set(t, Status.integrated, "merged and combined suite passed")
            return t


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()
