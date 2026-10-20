"""Subprocess backend. Control goes through the backend-owned control file (same transport as Modal)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from hospital_rl.infra.backend import JobBackend, JobSpec, JobState, JobStatus
from hospital_rl.infra.job import control_path
from hospital_rl.policies.checkpoint import atomic_write

import json


class LocalBackend(JobBackend):
    name = "local"

    def __init__(self, root: str | Path, python: str = sys.executable, env: dict[str, str] | None = None):
        self.root, self.python, self.env = Path(root), python, env or {}
        self._jobs: dict[str, tuple[subprocess.Popen, JobSpec]] = {}
        self._cancelled: set[str] = set()
        self._timed_out: set[str] = set()

    def submit(self, spec: JobSpec) -> str:
        job_id = f"local:{spec.run_id}:{spec.attempt_id}"
        if job_id in self._jobs:
            return job_id
        ctl = control_path(self.root, spec.run_id, spec.attempt_id)
        ctl.parent.mkdir(parents=True, exist_ok=True)
        ctl.unlink(missing_ok=True)
        spec_file = ctl.with_name(spec.attempt_id + ".spec.json")
        atomic_write(spec_file, spec.model_dump_json().encode())
        env = {**os.environ, **self.env, "OMP_NUM_THREADS": str(max(1, round(spec.cpu))),
               "MKL_NUM_THREADS": str(max(1, round(spec.cpu)))}
        proc = subprocess.Popen([self.python, "-m", "hospital_rl.infra.job", "--spec", str(spec_file),
                                 "--root", str(self.root)], env=env, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        self._jobs[job_id] = (proc, spec)
        threading.Thread(target=self._watch, args=(job_id,), daemon=True).start()
        return job_id

    def _watch(self, job_id: str) -> None:
        """Wall-clock cap: at timeout ask for checkpoint-and-stop, then kill after grace_s."""
        proc, spec = self._jobs[job_id]
        try:
            proc.wait(timeout=spec.timeout_s)
            return
        except subprocess.TimeoutExpired:
            self._timed_out.add(job_id)
        self._write(job_id, "stop")
        try:
            proc.wait(timeout=spec.grace_s)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _write(self, job_id: str, word: str) -> None:
        _, spec = self._jobs[job_id]
        atomic_write(control_path(self.root, spec.run_id, spec.attempt_id), word.encode())

    def status(self, job_id: str) -> JobStatus:
        proc, spec = self._jobs[job_id]
        rc = proc.poll()
        if rc is None:
            return JobStatus(job_id=job_id, state=JobState.RUNNING)
        if job_id in self._cancelled:
            return JobStatus(job_id=job_id, state=JobState.CANCELLED, detail="force-cancelled; work since last checkpoint lost")
        result_file = self.root / spec.run_id / spec.attempt_id / "result.json"
        if rc == 0 and result_file.exists():
            result = json.loads(result_file.read_text())
            state = JobState.TIMED_OUT if job_id in self._timed_out else JobState(result["state"])
            return JobStatus(job_id=job_id, state=state, result=result)
        state = JobState.TIMED_OUT if job_id in self._timed_out else JobState.FAILED
        return JobStatus(job_id=job_id, state=state, detail=f"exit code {rc}")

    def drain(self, job_id: str) -> None:
        self._write(job_id, "drain")

    def checkpoint_and_stop(self, job_id: str) -> None:
        self._write(job_id, "stop")

    def force_cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        self._jobs[job_id][0].kill()

    def wait(self, job_id: str, timeout: float = 120.0) -> JobStatus:
        end = time.time() + timeout
        while time.time() < end:
            st = self.status(job_id)
            if st.state.terminal:
                return st
            time.sleep(0.1)
        return self.status(job_id)
