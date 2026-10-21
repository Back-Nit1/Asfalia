"""Atomic, resumable, compatibility-checked checkpoints.

One ``.pt`` file (loaded with ``weights_only=True``, so no pickle execution) holds model, optimizer,
normalizer, RNG states, counters, and the metadata JSON. ``<name>.pt.meta.json`` is a human-readable
sidecar written after it; the embedded metadata is authoritative.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import random
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable, NamedTuple

import gymnasium
import numpy as np
import torch
from pydantic import BaseModel

from hospital_rl.events import EVENT_SCHEMA_VERSION
from hospital_rl.policies.ppo import PPO, Agent, FrozenAgent, GraphPolicy, Normalizer

CHECKPOINT_FORMAT = 1


class IncompatibleCheckpointError(ValueError):
    pass


class CheckpointMeta(BaseModel):
    checkpoint_format: int = CHECKPOINT_FORMAT
    event_schema_version: int = EVENT_SCHEMA_VERSION
    checkpoint_id: str
    parent_checkpoint_id: str | None = None
    role: str
    catalog_hash: str
    observation_schema_hash: str
    max_nodes: int
    policy_config: dict
    normalizer: dict
    config_hash: str
    code_revision: str | None
    library_versions: dict[str, str]
    training_steps: int
    policy_version: int
    last_rollout_id: int
    seed: int


class Checkpoint(NamedTuple):
    meta: CheckpointMeta
    state: dict


# ---------------------------------------------------------------- atomic io
def atomic_write(path: str | Path, data: bytes) -> None:
    """tmp file in the same dir + fsync + rename: readers see the old or the new file, never a torn one."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def clean_stale_tmp(directory: str | Path) -> None:
    for p in Path(directory).glob(".*.tmp"):
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- rng
def capture_rng() -> dict:
    _, keys, pos, has_gauss, cached = np.random.get_state()
    _, py_state, py_gauss = random.getstate()
    return {"torch": torch.get_rng_state(), "numpy_keys": torch.from_numpy(keys.astype(np.int64)),
            "numpy_rest": [pos, has_gauss, cached], "py_state": list(py_state), "py_gauss": py_gauss}


def restore_rng(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    pos, has_gauss, cached = s["numpy_rest"]
    np.random.set_state(("MT19937", s["numpy_keys"].numpy().astype(np.uint32), pos, has_gauss, cached))
    random.setstate((3, tuple(s["py_state"]), s["py_gauss"]))


# ---------------------------------------------------------------- save / load
def _code_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent, check=True).stdout.strip() or None
    except Exception:
        return None


def save_checkpoint(path: str | Path, ppo: PPO, *, role: str, catalog_hash: str, observation_schema_hash: str,
                    max_nodes: int, config_hash: str, seed: int, parent_id: str | None = None) -> CheckpointMeta:
    path = Path(path)
    model = {k: v.detach().cpu() for k, v in ppo.policy.state_dict().items()}
    buf = io.BytesIO()
    torch.save(model, buf)
    ckpt_id = hashlib.sha256(buf.getvalue() + str(ppo.env_steps).encode()).hexdigest()[:16]
    meta = CheckpointMeta(
        checkpoint_id=ckpt_id, parent_checkpoint_id=parent_id, role=role, catalog_hash=catalog_hash,
        observation_schema_hash=observation_schema_hash, max_nodes=max_nodes,
        policy_config=ppo.policy.config, normalizer=ppo.normalizer.stats_json(), config_hash=config_hash,
        code_revision=_code_revision(),
        library_versions={"torch": torch.__version__, "numpy": np.__version__,
                          "gymnasium": gymnasium.__version__, "python": platform.python_version()},
        training_steps=ppo.env_steps, policy_version=ppo.policy_version,
        last_rollout_id=ppo.last_rollout_id, seed=seed)
    out = io.BytesIO()
    torch.save({"meta": meta.model_dump_json(), "model": model, "optimizer": ppo.optimizer.state_dict(),
                "normalizer": ppo.normalizer.state_dict(), "rng": capture_rng()}, out)
    atomic_write(path, out.getvalue())
    atomic_write(path.with_name(path.name + ".meta.json"), json.dumps(meta.model_dump(), indent=2).encode())
    return meta


def load_checkpoint(path: str | Path) -> Checkpoint:
    state = torch.load(path, weights_only=True, map_location="cpu")
    return Checkpoint(CheckpointMeta.model_validate_json(state["meta"]), state)


def load_for_role(path: str | Path, expected_role: str, expected_catalog_hash: str,
                  expected_obs_hash: str) -> Checkpoint:
    ckpt = load_checkpoint(path)
    m = ckpt.meta
    bad = [f"{name}: checkpoint has {got!r}, expected {want!r}" for name, got, want in (
        ("checkpoint_format", m.checkpoint_format, CHECKPOINT_FORMAT),
        ("role", m.role, expected_role),
        ("catalog_hash", m.catalog_hash, expected_catalog_hash),
        ("observation_schema_hash", m.observation_schema_hash, expected_obs_hash)) if got != want]
    if bad:
        raise IncompatibleCheckpointError(f"checkpoint {path} incompatible - " + "; ".join(bad))
    return ckpt


def build_ppo(ckpt: Checkpoint, ppo_cfg, restore: bool = True) -> PPO:
    """Rebuild policy/normalizer/optimizer from a checkpoint; ``restore`` also restores RNG + counters."""
    c = ckpt.meta.policy_config
    policy = GraphPolicy(**c)
    policy.load_state_dict(ckpt.state["model"])
    norm = Normalizer(c["node_dim"], c["hist_dim"], c["glob_dim"])
    norm.load_state_dict(ckpt.state["normalizer"])
    ppo = PPO(policy, norm, ppo_cfg)
    ppo.optimizer.load_state_dict(ckpt.state["optimizer"])
    ppo.policy_version, ppo.last_rollout_id = ckpt.meta.policy_version, ckpt.meta.last_rollout_id
    ppo.env_steps = ckpt.meta.training_steps
    if restore:
        restore_rng(ckpt.state["rng"])
    return ppo


def load_frozen_agent(path: str | Path, expected_role: str, expected_catalog_hash: str, expected_obs_hash: str,
                      *, seed: int = 0, deterministic: bool = False) -> FrozenAgent:
    """Opponent/eval loader: compat-checked, normalizer frozen, eval mode."""
    ckpt = load_for_role(path, expected_role, expected_catalog_hash, expected_obs_hash)
    c = ckpt.meta.policy_config
    policy = GraphPolicy(**c)
    policy.load_state_dict(ckpt.state["model"])
    norm = Normalizer(c["node_dim"], c["hist_dim"], c["glob_dim"])
    norm.load_state_dict(ckpt.state["normalizer"])
    return FrozenAgent(Agent(policy, norm, ckpt.meta.role), ckpt.meta.checkpoint_id, seed, deterministic)


# ---------------------------------------------------------------- interrupts
@contextlib.contextmanager
def interrupt_flag():
    """Yields a callable that is True once SIGINT/SIGTERM arrived. First signal asks for a final
    checkpoint at the next safe boundary; a second SIGINT restores the default handler (hard stop)."""
    hit = []
    old = {}

    def handler(signum, frame):
        if hit and signum == signal.SIGINT:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        hit.append(signum)

    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            old[s] = signal.signal(s, handler)
    try:
        yield lambda: bool(hit)
    finally:
        for s, h in old.items():
            signal.signal(s, h)
