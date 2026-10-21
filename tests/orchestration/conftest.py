import subprocess
import sys
from pathlib import Path

import pytest

from hospital_rl.api.storage import Store
from hospital_rl.orchestration.adapters import CodingAgentAdapter
from hospital_rl.orchestration.contracts import AdapterCapabilities, AgentEvent, AgentResult, PlanConfig
from hospital_rl.orchestration.scheduler import Scheduler
from hospital_rl.orchestration.tasks import Task


class FakeAdapter(CodingAgentAdapter):
    """TEST ONLY. Writes scripted files into the worktree; finishes when `finish()` is called (or immediately)."""

    def __init__(self, writes: dict[str, dict[str, str]] | None = None, auto_finish: bool = True, ok: bool = True):
        self.writes, self.auto_finish, self.ok = writes or {}, auto_finish, ok
        self.sessions: dict[str, Task] = {}
        self.done: set[str] = set()
        self.cancelled: list[str] = []
        self.submitted: list[str] = []

    def capabilities(self):
        return AdapterCapabilities(name="FAKE-TEST-ADAPTER", configured=True)

    def submit(self, task, cwd):
        for rel, body in self.writes.get(task.id, {}).items():
            (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
            (cwd / rel).write_text(body)
        sid = f"fake-{task.id}-{len(self.submitted)}"
        self.submitted.append(task.id)
        self.sessions[sid] = task
        if self.auto_finish:
            self.done.add(sid)
        return sid

    def finish(self, task_id):
        self.done.update(s for s, t in self.sessions.items() if t.id == task_id)

    def drain_events(self, sid):
        return [AgentEvent(kind="message", summary="did a thing")] if sid in self.done else []

    def result(self, sid):
        return AgentResult(ok=self.ok, summary="agent says done", exit_code=0 if self.ok else 1) if sid in self.done else None

    def cancel(self, sid):
        self.cancelled.append(sid)


def sh(*a, cwd):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=r)
    sh("git", "config", "user.email", "t@t", cwd=r)
    sh("git", "config", "user.name", "t", cwd=r)
    (r / "README").write_text("x")
    sh("git", "add", ".", cwd=r)
    sh("git", "commit", "-qm", "base", cwd=r)
    sh("git", "tag", "base-v0", cwd=r)
    return r


@pytest.fixture
def clock():
    class C:
        t = 1000.0

        def __call__(self):
            return self.t
    return C()


def exists_cmd(path: str) -> str:
    return f"{sys.executable} -c \"import pathlib,sys; sys.exit(0 if pathlib.Path('{path}').exists() else 1)\""


def mk(id, deps=(), paths=("src/a",), cmds=(), owner="agent"):
    return Task(id=id, role="r", description=id, dependencies=list(deps), assigned_paths=list(paths),
                acceptance_commands=list(cmds), owner=owner)


@pytest.fixture
def make(repo, tmp_path, clock):
    def _make(tasks, adapter=None, **cfg):
        c = PlanConfig(repo=str(repo), worktree_root=str(tmp_path / "wt"), base_commit="base-v0", **cfg)
        return Scheduler(Store(), c, tasks, adapter, clock)
    return _make
