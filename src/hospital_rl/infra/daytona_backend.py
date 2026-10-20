"""Optional Daytona rollout backend. DEFERRED: interface only, nothing here talks to Daytona.

It must implement the same versioned rollout contract as Modal (rollout request pins scenario split,
seed range, opponent checkpoints, policy version, normalizer version, horizon, output path) once
second-mode distributed rollouts exist. Secret name: DAYTONA_API_KEY (value never read here).
"""
from __future__ import annotations

from hospital_rl.infra.backend import JobBackend, JobSpec, JobStatus

DEFERRED = "DEFERRED: Daytona backend is not implemented or tested; use local or modal."


class DaytonaBackend(JobBackend):
    name = "daytona"

    def submit(self, spec: JobSpec) -> str:
        raise NotImplementedError(DEFERRED)

    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError(DEFERRED)

    def drain(self, job_id: str) -> None:
        raise NotImplementedError(DEFERRED)

    def checkpoint_and_stop(self, job_id: str) -> None:
        raise NotImplementedError(DEFERRED)

    def force_cancel(self, job_id: str) -> None:
        raise NotImplementedError(DEFERRED)
