"""Provider-neutral job and artifact interfaces.

Three distinct stop operations (the dashboard must show them as different things):
  drain               - no new work; the in-flight round finishes (collect + update), then checkpoint and exit.
  checkpoint_and_stop - learner stops at its next safe boundary: in-flight rollout discarded, last
                        completed update checkpointed.
  force_cancel        - kill now. No checkpoint; work since the last published checkpoint is lost.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"  # reached total_steps
    DRAINED = "drained"
    STOPPED = "stopped"  # checkpoint-and-stop
    FAILED = "failed"
    CANCELLED = "cancelled"  # force-cancel
    TIMED_OUT = "timed_out"

    @property
    def terminal(self) -> bool:
        return self not in (JobState.PENDING, JobState.RUNNING)


class JobSpec(BaseModel):
    run_id: str
    attempt_id: str  # new id per (re)start; run lineage lives in checkpoint metadata
    train_config: dict  # TrainConfig fields; total_steps is the enforced step cap
    resume_from_attempt: str | None = None  # copy that attempt's latest.pt into this attempt, then resume
    timeout_s: int = 600
    grace_s: int = 30  # time allowed to checkpoint after a timeout/stop before a hard kill
    cpu: float = 1.0
    memory_mb: int = 2048
    max_containers: int = 1
    max_retries: int = 0


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    detail: str = ""
    result: dict = Field(default_factory=dict)


class JobBackend(ABC):
    name: str

    @abstractmethod
    def submit(self, spec: JobSpec) -> str:
        """Start the job; returns a job id. Submitting the same (run_id, attempt_id) again returns the same id."""

    @abstractmethod
    def status(self, job_id: str) -> JobStatus: ...

    @abstractmethod
    def drain(self, job_id: str) -> None: ...

    @abstractmethod
    def checkpoint_and_stop(self, job_id: str) -> None: ...

    @abstractmethod
    def force_cancel(self, job_id: str) -> None: ...


class ArtifactRef(BaseModel):
    path: str  # relative to the store root
    sha256: str
    size: int


class ArtifactStore(ABC):
    """Immutable-once-published artifacts under unique ``run_id/attempt_id/`` paths."""

    @abstractmethod
    def put(self, run_id: str, attempt_id: str, name: str, data: bytes) -> ArtifactRef:
        """Atomic publish. Refuses to overwrite an existing artifact."""

    @abstractmethod
    def get(self, ref: ArtifactRef) -> bytes:
        """Read and verify the hash; raises on mismatch."""

    @abstractmethod
    def publish_manifest(self, run_id: str, attempt_id: str) -> dict:
        """Hash every file of a finished attempt into manifest.json (written once, atomically)."""

    @abstractmethod
    def verify_manifest(self, run_id: str, attempt_id: str) -> list[str]:
        """Returns a list of problems (missing/changed/extra files); empty means intact."""
