"""Modal backend (first mode: one bounded job = vectorized envs + learner together).

SDK API verified against modal 1.6.2 (installed signatures + docs): ``modal.App``, ``App.function(image,
volumes, secrets, cpu, memory[MiB], timeout, retries, max_containers)``, ``Function.spawn`` ->
``FunctionCall`` (``.get(timeout)``, ``.cancel(terminate_containers)``, ``FunctionCall.from_id``),
``Volume.from_name(create_if_missing)``, ``Volume.commit()/reload()``, ``Volume.batch_upload``,
``App.run(detach=True)``. Everything Modal-specific lives in ``ModalGateway``; ``ModalBackend`` holds
the logic (cap enforcement, idempotent submit, state mapping) and is tested with a fake gateway.
The real gateway is only exercised by an explicit, tiny, credentialed smoke run (see docs).

Auth: the Modal client reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml. This module never
reads their values.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from hospital_rl.infra.backend import JobBackend, JobSpec, JobState, JobStatus
from hospital_rl.infra.job import STATE_BY_REASON, control_path, run_job

VOLUME_MOUNT = "/runs"


class ResourceCaps(BaseModel):
    cpu: float = 1.0
    memory_mb: int = 2048
    timeout_s: int = 600
    max_containers: int = 1
    max_retries: int = 0
    max_steps: int = 100_000


class ModalConfig(BaseModel):
    app_name: str = "hospital-rl-train"
    volume_name: str = "hospital-rl-runs"
    secret_names: list[str] = []  # Modal Secret objects mounted into the job; none needed for training itself
    extra_modules: list[str] = []  # extra local python modules to ship (e.g. a toy env for smoke runs)
    caps: ResourceCaps = ResourceCaps()


def load_modal_config(path: str | Path) -> ModalConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    caps = {**raw.get("caps", {}), "max_steps": raw.get("train", {}).get("total_steps", ResourceCaps().max_steps)}
    return ModalConfig(**{k: raw[k] for k in ("app_name", "volume_name", "secret_names", "extra_modules") if k in raw}, caps=caps)


class ModalGateway(Protocol):
    def spawn(self, spec_json: str) -> str: ...

    def poll(self, call_id: str) -> tuple[str, dict | str]:
        """("running", {}) | ("done", result dict) | ("timed_out", msg) | ("failed", msg)"""

    def cancel(self, call_id: str) -> None: ...

    def write_control(self, run_id: str, attempt_id: str, word: str) -> None: ...


class ModalBackend(JobBackend):
    name = "modal"

    def __init__(self, cfg: ModalConfig, gateway: ModalGateway):
        self.cfg, self.gw = cfg, gateway
        self._calls: dict[str, str] = {}  # job_id -> call id
        self._specs: dict[str, JobSpec] = {}
        self._cancelled: set[str] = set()

    def check_caps(self, spec: JobSpec) -> None:
        c, steps = self.cfg.caps, spec.train_config.get("total_steps", 0)
        over = [f"{n}={v} > cap {cap}" for n, v, cap in (
            ("timeout_s", spec.timeout_s, c.timeout_s), ("cpu", spec.cpu, c.cpu),
            ("memory_mb", spec.memory_mb, c.memory_mb), ("max_containers", spec.max_containers, c.max_containers),
            ("max_retries", spec.max_retries, c.max_retries), ("total_steps", steps, c.max_steps)) if v > cap]
        if over:
            raise ValueError("job exceeds configured resource caps: " + "; ".join(over))

    def submit(self, spec: JobSpec) -> str:
        job_id = f"modal:{spec.run_id}:{spec.attempt_id}"
        if job_id in self._calls:
            return job_id  # idempotent: never launch the same attempt twice
        self.check_caps(spec)
        self._calls[job_id] = self.gw.spawn(spec.model_dump_json())
        self._specs[job_id] = spec
        return job_id

    def status(self, job_id: str) -> JobStatus:
        if job_id in self._cancelled:
            return JobStatus(job_id=job_id, state=JobState.CANCELLED, detail="force-cancelled; work since last checkpoint lost")
        kind, payload = self.gw.poll(self._calls[job_id])
        if kind == "running":
            return JobStatus(job_id=job_id, state=JobState.RUNNING)
        if kind == "done":
            return JobStatus(job_id=job_id, state=JobState(payload["state"]), result=payload)
        return JobStatus(job_id=job_id, state=JobState.TIMED_OUT if kind == "timed_out" else JobState.FAILED, detail=str(payload))

    def _control(self, job_id: str, word: str) -> None:
        s = self._specs[job_id]
        self.gw.write_control(s.run_id, s.attempt_id, word)

    def drain(self, job_id: str) -> None:
        self._control(job_id, "drain")

    def checkpoint_and_stop(self, job_id: str) -> None:
        self._control(job_id, "stop")

    def force_cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)
        self.gw.cancel(self._calls[job_id])


class RealModalGateway:
    """Thin wrapper over the Modal SDK. Importing it requires the ``modal`` package."""

    def __init__(self, cfg: ModalConfig):
        import modal

        self.modal, self.cfg = modal, cfg
        c = cfg.caps
        self.app = modal.App(cfg.app_name)
        self.vol = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
        image = (modal.Image.debian_slim(python_version="3.13")
                 .pip_install("torch", "gymnasium", "numpy", "pydantic", "pyyaml")
                 .add_local_python_source("hospital_rl", *cfg.extra_modules))
        vol = self.vol

        @self.app.function(image=image, volumes={VOLUME_MOUNT: vol}, secrets=[modal.Secret.from_name(n) for n in cfg.secret_names],
                           cpu=c.cpu, memory=c.memory_mb, timeout=c.timeout_s, max_containers=c.max_containers,
                           retries=modal.Retries(max_retries=c.max_retries, initial_delay=1.0) if c.max_retries else None)
        def train_job(spec_json: str) -> dict:
            from hospital_rl.training.train import control_from_file

            spec = JobSpec.model_validate_json(spec_json)
            file_ctl = control_from_file(control_path(VOLUME_MOUNT, spec.run_id, spec.attempt_id))

            def control():
                vol.reload()  # see control files written by the client since our last look
                return file_ctl()

            return run_job(spec, VOLUME_MOUNT, control=control, after_save=lambda p: vol.commit())

        self.fn = train_job

    def spawn(self, spec_json: str) -> str:
        with self.app.run(detach=True):  # detach: the app outlives this client process
            return self.fn.spawn(spec_json).object_id

    def poll(self, call_id):
        m = self.modal
        try:
            return "done", m.FunctionCall.from_id(call_id).get(timeout=0)
        except (TimeoutError, m.exception.TimeoutError):
            return "running", {}
        except m.exception.FunctionTimeoutError as e:
            return "timed_out", str(e)
        except Exception as e:  # noqa: BLE001 - surface any remote failure as FAILED with its message
            return "failed", f"{type(e).__name__}: {e}"

    def cancel(self, call_id):
        self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def write_control(self, run_id, attempt_id, word):
        with self.vol.batch_upload(force=True) as up:
            up.put_file(io.BytesIO(word.encode()), "/" + str(control_path("", run_id, attempt_id)).lstrip("/"))
