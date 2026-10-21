"""Global-scope Modal app (required for ``deploy``). Configured through ``HOSPITAL_RL_MODAL_CFG`` (JSON ModalConfig),
set by ``RealModalGateway`` on the client and baked into the image env for the container. Import needs ``modal``."""
from __future__ import annotations

import os

import modal

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.job import control_path, run_job
from hospital_rl.infra.modal_app import CFG_ENV, FUNCTION, REMOTE_ROOT, VOLUME_MOUNT, ModalConfig, build_image

cfg = ModalConfig.model_validate_json(os.environ[CFG_ENV])
app = modal.App(cfg.app_name)
vol = modal.Volume.from_name(cfg.volume_name, create_if_missing=True)
_c = cfg.caps


@app.function(image=build_image(cfg), volumes={VOLUME_MOUNT: vol}, secrets=[modal.Secret.from_name(n) for n in cfg.secret_names],
              cpu=_c.cpu, memory=_c.memory_mb, timeout=_c.timeout_s + _c.grace_s,  # job soft-stops at timeout_s, hard kill after grace
              max_containers=_c.max_containers, name=FUNCTION,
              retries=modal.Retries(max_retries=_c.max_retries, initial_delay=1.0) if _c.max_retries else None)
def train_job(spec_json: str) -> dict:
    from hospital_rl.training.train import control_from_file

    os.chdir(REMOTE_ROOT)  # relative config paths in bundles resolve against the shipped configs/
    spec = JobSpec.model_validate_json(spec_json)
    file_ctl = control_from_file(control_path(VOLUME_MOUNT, spec.run_id, spec.attempt_id))

    def control():
        vol.reload()  # see control files written by the client since our last look
        return file_ctl()

    try:
        return run_job(spec, VOLUME_MOUNT, control=control, after_save=lambda p: vol.commit())
    finally:
        vol.commit()  # keep partial progress when the job raises, so a new attempt can resume
