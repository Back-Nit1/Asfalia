"""Modal backend: the common lifecycle (``lifecycle.GatewayBackend``) over a Modal gateway.

SDK surface verified against modal 1.6.2 (installed signatures) and the Modal docs: ``modal.App``,
``App.function(image, volumes, secrets, cpu, memory[MiB], timeout, retries, max_containers)``, ``App.run`` /
``App.deploy``, ``Function.from_name(app, name).spawn`` -> ``FunctionCall`` (``.get(timeout)``, ``.cancel(
terminate_containers)``, ``FunctionCall.from_id``; results readable for 7 days), ``Volume.from_name(create_if_missing)``,
``Volume.commit()/reload()/batch_upload/read_file/listdir/remove_file``, ``Image.add_local_dir/
add_local_python_source/env``. Everything Modal-specific is in ``RealModalGateway`` and ``modal_functions``; the
backend is tested with a fake gateway and the real gateway only by an explicit, tiny, credentialed smoke run.

Two lifecycles (``ModalConfig.mode``):
  ephemeral (default, live-verified): the controller holds ``app.run()``; the app and its containers stop when the
      controller calls ``close()`` or the last owned call ends. A *restarted* controller can still poll finished
      calls (``FunctionCall.from_id``), but cannot vouch for a call whose app it does not own: such a call is
      reported FAILED("lost") with its last durable checkpoint so a new attempt can resume. Not live-verified:
      what Modal does to the containers when the owning client dies abruptly.
  deployed (NOT live-verified): ``deploy()`` publishes the app once; ``Function.from_name`` spawns survive the
      controller, and any restarted controller re-attaches via the registry's call ids.

Auth: the Modal client reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml. This module never reads values.
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

from pydantic import BaseModel

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.bundle import repo_root
from hospital_rl.infra.job import control_path
from hospital_rl.infra.lifecycle import GatewayBackend, Registry, ResourceCaps

VOLUME_MOUNT = "/runs"
REMOTE_ROOT = "/root"  # add_local_python_source puts hospital_rl here; configs/ is copied next to it
CFG_ENV = "HOSPITAL_RL_MODAL_CFG"
DEPS = ["numpy", "networkx", "gymnasium", "pettingzoo", "pydantic>=2", "pyyaml"]
TORCH_CPU = "https://download.pytorch.org/whl/cpu"  # default linux torch wheel drags in ~3GB of CUDA
FUNCTION = "train_job"


class ModalConfig(BaseModel):
    app_name: str = "hospital-rl-train"
    volume_name: str = "hospital-rl-runs"
    secret_names: list[str] = []  # Modal Secret objects mounted into the job; none needed for training itself
    mode: str = "ephemeral"  # "ephemeral" | "deployed"
    registry_dir: str = "runs/.registry/modal"
    caps: ResourceCaps = ResourceCaps()


def load_modal_config(path: str | Path) -> ModalConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    caps = {**raw.get("caps", {}), "max_steps": raw.get("train", {}).get("total_steps", ResourceCaps().max_steps)}
    return ModalConfig(**{k: raw[k] for k in ("app_name", "volume_name", "secret_names", "mode", "registry_dir") if k in raw}, caps=caps)


def build_image(cfg: ModalConfig):
    """Clean-container image: CPU torch + runtime deps + the hospital_rl package + configs. Nothing from tests/."""
    import modal

    return (modal.Image.debian_slim(python_version="3.13").pip_install("torch", index_url=TORCH_CPU).pip_install(*DEPS)
            .env({CFG_ENV: cfg.model_dump_json()})
            .add_local_dir(repo_root() / "configs", f"{REMOTE_ROOT}/configs")
            .add_local_python_source("hospital_rl"))


class ModalBackend(GatewayBackend):
    def __init__(self, cfg: ModalConfig, gateway, registry: Registry | str | Path | None = None):
        reg = registry if isinstance(registry, Registry) else Registry(registry or cfg.registry_dir)
        super().__init__("modal", gateway, reg, cfg.caps)
        self.cfg = cfg


class RealModalGateway:
    """Thin wrapper over the Modal SDK. Importing it requires the ``modal`` package."""

    def __init__(self, cfg: ModalConfig):
        import importlib

        import modal

        self.modal, self.cfg = modal, cfg
        os.environ[CFG_ENV] = cfg.model_dump_json()  # modal_functions defines its app at import, from this config
        self.fns = importlib.import_module("hospital_rl.infra.modal_functions")
        if self.fns.cfg != cfg:
            raise RuntimeError("modal_functions was already imported with a different ModalConfig in this process")
        self.vol, self._stack, self._owned = self.fns.vol, None, set()

    def deploy(self) -> None:
        with self.modal.enable_output():
            self.fns.app.deploy(name=self.cfg.app_name)

    def spawn(self, spec: JobSpec) -> str:
        if self.cfg.mode == "deployed":
            fn = self.modal.Function.from_name(self.cfg.app_name, FUNCTION)
        else:
            if self._stack is None:
                self._stack = contextlib.ExitStack()
                self._stack.enter_context(self.fns.app.run())
            fn = self.fns.train_job
        call_id = fn.spawn(spec.model_dump_json()).object_id
        self._owned.add(call_id)
        return call_id

    def close(self) -> None:
        """Stop the ephemeral app (terminating its containers). A deployed app is not affected."""
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def _finished(self, call_id: str) -> None:
        self._owned.discard(call_id)
        if not self._owned:
            self.close()  # a lingering non-detached app keeps containers alive (observed live); stop it with the last call

    def poll(self, call_id):
        m = self.modal
        try:
            out = "done", m.FunctionCall.from_id(call_id).get(timeout=0)
        except (TimeoutError, m.exception.TimeoutError):
            if self.cfg.mode == "ephemeral" and call_id not in self._owned:
                return "lost", ("owning controller is gone; ephemeral Modal app state cannot be verified from here "
                                "(see last durable checkpoint, cancel, then resume with a new attempt)")
            return "running", {}
        except m.exception.FunctionTimeoutError as e:
            out = "timed_out", str(e)
        except m.exception.OutputExpiredError:
            out = "failed", "call output expired (Modal keeps results 7 days); read result.json/manifest from the Volume"
        except Exception as e:  # noqa: BLE001 - quota, startup, preemption, remote exceptions: surface as FAILED with the message
            out = "failed", f"{type(e).__name__}: {e}"
        self._finished(call_id)
        return out

    def cancel(self, call_id):
        self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        self._finished(call_id)

    def write_control(self, run_id, attempt_id, word):
        with self.vol.batch_upload(force=True) as up:
            up.put_file(io.BytesIO(word.encode()), "/" + str(control_path("", run_id, attempt_id)).lstrip("/"))

    def read_file(self, run_id, attempt_id, name):
        try:
            return b"".join(self.vol.read_file(f"/{run_id}/{attempt_id}/{name}"))
        except self.modal.exception.NotFoundError:
            return None

    def list_attempts(self):
        last = lambda e: e.path.strip("/").split("/")[-1]  # noqa: E731
        return [(last(r), last(a)) for r in self.vol.listdir("/") if last(r) != "control"
                for a in self.vol.listdir(r.path) if a.type.name == "DIRECTORY" and last(a) != "control"]

    def delete_attempt(self, run_id, attempt_id):
        self.vol.remove_file(f"/{run_id}/{attempt_id}", recursive=True)
        with contextlib.suppress(self.modal.exception.NotFoundError):
            self.vol.remove_file(f"/{control_path('', run_id, attempt_id)}")
