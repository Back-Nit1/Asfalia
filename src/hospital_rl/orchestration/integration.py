"""Git worktree per task, evidence collection, acceptance runs, and combined-suite integration."""
from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from .contracts import TestResult
from .events import clean_env, tail
from .tasks import Task, owns


def git(repo: str | Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {tail(p.stderr)}")
    return p.stdout.strip()


def create_worktree(repo: str, root: str, task: Task, base: str) -> Path:
    """Pinned-commit worktree on branch task/<id>. Reuses an existing one (retries continue on the same branch)."""
    path = Path(root).resolve() / task.id
    if not path.exists():
        git(repo, "worktree", "add", "-b", f"task/{task.id}", str(path), base)
    task.base_commit = git(repo, "rev-parse", base)
    task.branch, task.worktree = f"task/{task.id}", str(path)
    return path


def remove_worktree(repo: str, task: Task) -> None:
    if task.worktree:
        git(repo, "worktree", "remove", "--force", task.worktree, check=False)


def changed_files(task: Task) -> list[str]:
    """Committed + uncommitted + untracked changes relative to the pinned base."""
    wt = task.worktree
    tracked = git(wt, "diff", "--name-only", task.base_commit).splitlines()
    untracked = git(wt, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({*tracked, *untracked})


def out_of_scope(task: Task, files: list[str]) -> list[str]:
    return [f for f in files if not owns(task.assigned_paths, f)]


def diff(task: Task) -> str:
    git(task.worktree, "add", "-N", ".", check=False)   # make untracked files visible in the diff
    return tail(git(task.worktree, "diff", task.base_commit), 200_000)


def run_command(command: str, cwd: str | Path, timeout: float) -> TestResult:
    t0 = time.monotonic()
    try:
        p = subprocess.run(shlex.split(command), cwd=cwd, env=clean_env(), capture_output=True, text=True, timeout=timeout)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 124, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), f"timeout after {timeout}s"
    except OSError as e:
        rc, out, err = 127, "", str(e)
    return TestResult(command=command, exit_code=rc, duration_s=round(time.monotonic() - t0, 3),
                      stdout_tail=tail(out), stderr_tail=tail(err))


def integrate(repo: str, task: Task, branch: str, suite: list[str], timeout: float) -> list[TestResult]:
    """Merge task branch into `branch` (checked out in `repo`), run the combined suite, and undo the merge if it fails.
    The caller must pass a dedicated integration checkout, never a developer's working copy."""
    git(repo, "checkout", branch)
    before = git(repo, "rev-parse", "HEAD")
    try:
        git(repo, "merge", "--no-ff", "-m", f"integrate {task.id}", task.branch)
    except RuntimeError:
        git(repo, "merge", "--abort", check=False)
        raise
    results = [run_command(c, repo, timeout) for c in suite]
    if any(r.exit_code for r in results):
        git(repo, "reset", "--hard", before)
    return results
