"""One job lifecycle for every compute provider.

``GatewayBackend`` owns all semantics (bundle validation, caps, idempotent submit, durable attempt records,
state mapping, drain / checkpoint-and-stop / force-cancel, checkpoint freshness). A ``Gateway`` is only the thin
transport: spawn a remote unit, poll it, cancel it, move control words and read files under the store root.
``LocalGateway`` (subprocess) and ``modal_app.RealModalGateway`` implement it; tests use fakes.

Durable state lives outside process memory: ``Registry`` holds one JSON record per attempt (remote id, state,
lineage, labels). A restarted controller re-attaches with the same registry directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from hospital_rl.infra import bundle
from hospital_rl.infra.backend import JobBackend, JobSpec, JobState, JobStatus
from hospital_rl.infra.job import PROGRESS, control_path
from hospital_rl.policies.checkpoint import atomic_write


class ResourceCaps(BaseModel):
    cpu: float = 1.0
    memory_mb: int = 2048
    timeout_s: int = 600
    grace_s: int = 30
    max_containers: int = 1
    max_retries: int = 0
    max_steps: int = 100_000


class Gateway(Protocol):
    def spawn(self, spec: JobSpec) -> str: ...

    def poll(self, remote_id: str) -> tuple[str, dict | str]:
        """("running", {}) | ("done", result) | ("timed_out", msg) | ("failed", msg) | ("lost", msg)"""

    def cancel(self, remote_id: str) -> None: ...

    def write_control(self, run_id: str, attempt_id: str, word: str) -> None: ...

    def read_file(self, run_id: str, attempt_id: str, name: str) -> bytes | None: ...

    def list_attempts(self) -> list[tuple[str, str]]: ...

    def delete_attempt(self, run_id: str, attempt_id: str) -> None: ...


class Registry:
    """Durable attempt records: <dir>/<run>__<attempt>.json, written atomically."""

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _file(self, run_id: str, attempt_id: str) -> Path:
        return self.dir / f"{run_id}__{attempt_id}.json"

    def get(self, run_id: str, attempt_id: str) -> dict | None:
        f = self._file(run_id, attempt_id)
        return json.loads(f.read_text()) if f.exists() else None

    def put(self, rec: dict) -> None:
        with self._lock:
            atomic_write(self._file(rec["run_id"], rec["attempt_id"]), json.dumps(rec, indent=1).encode())

    def update(self, run_id: str, attempt_id: str, **kw) -> dict:
        with self._lock:
            rec = self.get(run_id, attempt_id)
            rec.update(kw)
            atomic_write(self._file(run_id, attempt_id), json.dumps(rec, indent=1).encode())
            return rec

    def all(self) -> list[dict]:
        return [json.loads(f.read_text()) for f in sorted(self.dir.glob("*__*.json"))]


class GatewayBackend(JobBackend):
    def __init__(self, name: str, gateway: Gateway, registry: Registry, caps: ResourceCaps | None = None):
        self.name, self.gw, self.reg, self.caps = name, gateway, registry, caps
        self._remote: dict[str, str] = {}

    # --- ids -------------------------------------------------------------------------------------------------
    def _rec(self, job_id: str) -> dict:
        _, run, attempt = job_id.split(":", 2)
        rec = self.reg.get(run, attempt)
        if rec is None:
            raise KeyError(f"unknown job {job_id}: no attempt record in the registry")
        return rec

    # --- submit ----------------------------------------------------------------------------------------------
    def check_caps(self, spec: JobSpec) -> None:
        if self.caps is None:
            return
        c = self.caps
        over = [f"{n}={v} > cap {cap}" for n, v, cap in (
            ("timeout_s", spec.timeout_s, c.timeout_s), ("grace_s", spec.grace_s, c.grace_s), ("cpu", spec.cpu, c.cpu),
            ("memory_mb", spec.memory_mb, c.memory_mb), ("max_containers", spec.max_containers, c.max_containers),
            ("max_retries", spec.max_retries, c.max_retries), ("total_steps", bundle.total_steps(spec), c.max_steps)) if v > cap]
        if over:
            raise ValueError("job exceeds configured resource caps: " + "; ".join(over))

    def submit(self, spec: JobSpec) -> str:
        job_id = f"{self.name}:{spec.run_id}:{spec.attempt_id}"
        if self.reg.get(spec.run_id, spec.attempt_id):
            return job_id  # idempotent across process restarts: an attempt id is launched at most once, ever
        spec = bundle.resolve(spec)
        bundle.validate(spec)
        self.check_caps(spec)
        if spec.resume_from_attempt:
            src = self.reg.get(spec.run_id, spec.resume_from_attempt)
            if src and not JobState(src["state"]).terminal:
                raise ValueError(f"cannot resume from {spec.resume_from_attempt}: still {src['state']}; stop it first")
        # intent is recorded before the spawn so a crash in between can never lead to a second launch
        self.reg.put({"job_id": job_id, "backend": self.name, "run_id": spec.run_id, "attempt_id": spec.attempt_id,
                      "remote_id": None, "state": JobState.PENDING.value, "submitted_at": time.time(),
                      "finished_at": None, "resume_from_attempt": spec.resume_from_attempt, "labels": spec.labels,
                      "spec": json.loads(spec.model_dump_json()), "detail": "", "result": {}})
        remote = self.gw.spawn(spec)
        self.reg.update(spec.run_id, spec.attempt_id, remote_id=remote)
        return job_id

    # --- status ----------------------------------------------------------------------------------------------
    def _checkpoint(self, rec: dict) -> dict:
        raw = self.gw.read_file(rec["run_id"], rec["attempt_id"], PROGRESS)
        if not raw:
            return {}
        p = json.loads(raw)
        return {**p, "age_s": round(time.time() - p["written_at"], 1)}

    def status(self, job_id: str) -> JobStatus:
        rec = self._rec(job_id)
        state = JobState(rec["state"])
        if state.terminal:
            return JobStatus(job_id=job_id, state=state, detail=rec["detail"], result=rec["result"], checkpoint=self._checkpoint(rec))
        if rec["remote_id"] is None:
            detail = "spawn outcome unknown (controller died between intent and spawn); this attempt id is burned, resubmit with a new attempt id"
            self.reg.update(rec["run_id"], rec["attempt_id"], state=JobState.FAILED.value, detail=detail, finished_at=time.time())
            return JobStatus(job_id=job_id, state=JobState.FAILED, detail=detail)
        kind, payload = self.gw.poll(rec["remote_id"])
        if kind == "running":
            return JobStatus(job_id=job_id, state=JobState.RUNNING, checkpoint=self._checkpoint(rec))
        result: dict = {}
        if kind == "done":
            state, result = JobState(payload["state"]), payload
            detail = ""
        else:
            state = JobState.TIMED_OUT if kind == "timed_out" else JobState.FAILED
            detail = str(payload)
        cp = self._checkpoint(rec)
        if state in (JobState.FAILED, JobState.TIMED_OUT):
            m = self.gw.read_file(rec["run_id"], rec["attempt_id"], "manifest.json")
            detail += (f" | partial artifacts: no manifest; last durable checkpoint {cp.get('env_steps')} steps"
                       f" (resume via a new attempt with resume_from_attempt={rec['attempt_id']!r})") if not m and cp else ""
        self.reg.update(rec["run_id"], rec["attempt_id"], state=state.value, detail=detail, result=result, finished_at=time.time())
        return JobStatus(job_id=job_id, state=state, detail=detail, result=result, checkpoint=cp)

    # --- control ---------------------------------------------------------------------------------------------
    def _control(self, job_id: str, word: str) -> None:
        rec = self._rec(job_id)
        if not JobState(rec["state"]).terminal:
            self.gw.write_control(rec["run_id"], rec["attempt_id"], word)

    def drain(self, job_id: str) -> None:
        self._control(job_id, "drain")

    def checkpoint_and_stop(self, job_id: str) -> None:
        self._control(job_id, "stop")

    def force_cancel(self, job_id: str) -> None:
        rec = self._rec(job_id)
        if JobState(rec["state"]).terminal or rec["remote_id"] is None:
            return
        self.reg.update(rec["run_id"], rec["attempt_id"], state=JobState.CANCELLED.value,
                        detail="force-cancelled; work since last checkpoint lost", finished_at=time.time())
        self.gw.cancel(rec["remote_id"])

    def wait(self, job_id: str, timeout: float = 120.0, poll_s: float = 0.2) -> JobStatus:
        end = time.time() + timeout
        while time.time() < end:
            if (st := self.status(job_id)).state.terminal:
                return st
            time.sleep(poll_s)
        return self.status(job_id)


# ------------------------------------------------------------------------------------------------------------------
def _pid_alive(pid: int, started: str) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True).stdout.strip() == started


class LocalGateway:
    """Subprocess transport. remote_id = "<pid>:<lstart>" so a restarted controller can re-attach (liveness only: the
    exit status of a process we did not start is unknowable, so its outcome is read from result.json)."""

    def __init__(self, root: str | Path, python: str = sys.executable, env: dict[str, str] | None = None):
        self.root, self.python, self.env = Path(root), python, env or {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._timed_out: set[str] = set()
        self._specs: dict[str, JobSpec] = {}

    def _launch(self, spec: JobSpec) -> subprocess.Popen:
        spec_file = control_path(self.root, spec.run_id, spec.attempt_id).with_name(spec.attempt_id + ".spec.json")
        atomic_write(spec_file, spec.model_dump_json().encode())
        threads = str(max(1, round(spec.cpu)))
        env = {**os.environ, **self.env, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        return subprocess.Popen([self.python, "-m", "hospital_rl.infra.job", "--spec", str(spec_file), "--root", str(self.root)],
                                env=env, cwd=bundle.repo_root(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def spawn(self, spec: JobSpec) -> str:
        ctl = control_path(self.root, spec.run_id, spec.attempt_id)
        ctl.parent.mkdir(parents=True, exist_ok=True)
        ctl.unlink(missing_ok=True)
        proc = self._launch(spec)
        started = subprocess.run(["ps", "-p", str(proc.pid), "-o", "lstart="], capture_output=True, text=True).stdout.strip()
        rid = f"{proc.pid}|{started}|{spec.run_id}|{spec.attempt_id}"
        self._procs[rid], self._specs[rid] = proc, spec
        threading.Thread(target=self._supervise, args=(rid,), daemon=True).start()
        return rid

    def _supervise(self, rid: str) -> None:
        """Wall-clock cap (hard kill at timeout_s + grace_s; the job asks itself to stop at timeout_s) and same-attempt
        retries: a retry resumes from the attempt's own latest.pt, so steps are never counted twice."""
        spec, retries = self._specs[rid], 0
        while True:
            proc = self._procs[rid]
            try:
                proc.wait(timeout=spec.timeout_s + spec.grace_s)
            except subprocess.TimeoutExpired:
                self._timed_out.add(rid)
                proc.kill()
                proc.wait()
            if proc.returncode == 0 or rid in self._cancelled or rid in self._timed_out or retries >= spec.max_retries:
                return
            retries += 1
            self._procs[rid] = self._launch(spec)

    def poll(self, remote_id: str):
        pid, started, run_id, attempt_id = remote_id.split("|")
        proc = self._procs.get(remote_id)
        if proc is not None:
            if proc.poll() is None:
                return "running", {}
            rc, detail = proc.returncode, f"exit code {proc.returncode}"
        elif _pid_alive(int(pid), started):
            return "running", {}
        else:
            rc, detail = None, "process gone after controller restart; exit status unknown"
        if remote_id in self._timed_out:
            return "timed_out", "hard-killed after timeout + grace; last checkpoint kept"
        rf = self.root / run_id / attempt_id / "result.json"
        if rc in (0, None) and rf.exists():
            return "done", json.loads(rf.read_text())
        return "failed", detail

    def cancel(self, remote_id: str) -> None:
        self._cancelled.add(remote_id)
        if proc := self._procs.get(remote_id):
            proc.kill()
        else:
            os.kill(int(remote_id.split("|")[0]), 9)

    def write_control(self, run_id, attempt_id, word):
        atomic_write(control_path(self.root, run_id, attempt_id), word.encode())

    def read_file(self, run_id, attempt_id, name):
        p = self.root / run_id / attempt_id / name
        return p.read_bytes() if p.exists() else None

    def list_attempts(self):
        return [(r.name, a.name) for r in sorted(self.root.iterdir()) if r.is_dir() and r.name != "control"
                for a in sorted(r.iterdir()) if a.is_dir() and a.name != "control"] if self.root.exists() else []

    def delete_attempt(self, run_id, attempt_id):
        import shutil
        shutil.rmtree(self.root / run_id / attempt_id, ignore_errors=True)
        control_path(self.root, run_id, attempt_id).unlink(missing_ok=True)
        control_path(self.root, run_id, attempt_id).with_name(attempt_id + ".spec.json").unlink(missing_ok=True)
