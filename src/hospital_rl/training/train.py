"""Single-role PPO training loop: vectorized envs, versioned rollouts, resume, interrupt-safe.

Control signals (``control()`` returns None | "drain" | "stop"; SIGINT/SIGTERM count as "stop"):
  drain - finish the in-flight round (collect + update), checkpoint, exit.
  stop  - checkpoint-and-stop: discard an in-flight rollout, checkpoint the last completed update, exit.
Force-cancel is outside the process (SIGKILL): no checkpoint, the last atomic one survives.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from pydantic import BaseModel

from hospital_rl.policies.checkpoint import (atomic_write, build_ppo, clean_stale_tmp, interrupt_flag, load_for_role,
                                             save_checkpoint)
from hospital_rl.policies.ppo import (PPO, Agent, GraphPolicy, Normalizer, PPOConfig, SyncVecEnv,
                                      collect_rollout)


class TrainConfig(BaseModel):
    role: str = "defender"
    seed: int = 0
    total_steps: int = 20_000
    n_envs: int = 8
    rollout_len: int = 64
    hidden: int = 64
    checkpoint_every: int = 10  # rounds
    env_factory: str  # "package.module:callable"
    env_kwargs: dict = {}
    ppo: PPOConfig = PPOConfig()

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TrainResult:
    env_steps: int
    stop_reason: str  # done | drain | stop
    last_checkpoint: Path
    history: list[dict] = field(default_factory=list)


def load_factory(spec: str) -> Callable:
    mod, name = spec.split(":")
    return getattr(importlib.import_module(mod), name)


def control_from_file(path: str | Path) -> Callable[[], str | None]:
    p = Path(path)
    return lambda: (p.read_text().strip() or None) if p.exists() else None


def train(cfg: TrainConfig, out_dir: str | Path, *, resume: bool = False,
          control: Callable[[], str | None] = lambda: None, log: Callable[[str], None] = print) -> TrainResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_stale_tmp(out)
    factory = load_factory(cfg.env_factory)
    venv = SyncVecEnv([lambda: factory(**cfg.env_kwargs)] * cfg.n_envs)
    meta = venv.metadata
    if meta.get("role") != cfg.role:
        raise ValueError(f"env role {meta.get('role')!r} != config role {cfg.role!r}")
    ids = dict(catalog_hash=meta["catalog_hash"], observation_schema_hash=meta["observation_schema_hash"])
    space = venv.observation_space
    max_nodes = space["node_mask"].shape[0]
    n_types = int(venv.action_space.n - 1) // max_nodes
    latest = out / "latest.pt"
    parent_id, grand, resumed = None, None, False  # grand: parent of last saved, for same-version refreshes
    if resume and latest.exists():
        ckpt = load_for_role(latest, cfg.role, ids["catalog_hash"], ids["observation_schema_hash"])
        ppo, parent_id, grand = build_ppo(ckpt, cfg.ppo), ckpt.meta.checkpoint_id, ckpt.meta.parent_checkpoint_id
        resumed = True
        log(f"resumed {parent_id} at {ppo.env_steps} steps (policy v{ppo.policy_version})")
    else:
        torch.manual_seed(cfg.seed), np.random.seed(cfg.seed)
        policy = GraphPolicy(int(space["node_features"].shape[1]), int(space["history"].shape[1]),
                             int(space["globals"].shape[0]), n_types, cfg.hidden)
        ppo = PPO(policy, Normalizer(policy.config["node_dim"], policy.config["hist_dim"],
                                     policy.config["glob_dim"]), cfg.ppo)

    def save() -> Path:
        """Publish ckpt_<version>.pt and refresh latest.pt. If this version was already published
        (state changed only by a discarded rollout), refresh latest.pt alone, keeping lineage."""
        nonlocal parent_id, grand
        path = out / f"ckpt_{ppo.policy_version:06d}.pt"
        kw = dict(role=cfg.role, max_nodes=max_nodes, config_hash=cfg.hash(), seed=cfg.seed, **ids)
        if path.exists():
            save_checkpoint(latest, ppo, parent_id=grand, **kw)
            return latest
        m = save_checkpoint(path, ppo, parent_id=parent_id, **kw)
        grand, parent_id = parent_id, m.checkpoint_id
        atomic_write(latest, path.read_bytes())
        atomic_write(latest.with_name("latest.pt.meta.json"), path.with_name(path.name + ".meta.json").read_bytes())
        return path

    history, reason, rounds, dirty = [], "done", 0, not resumed
    last = latest
    with interrupt_flag() as interrupted:
        ctl = lambda: "stop" if interrupted() else control()  # noqa: E731
        while ppo.env_steps < cfg.total_steps:
            if (signal := ctl()) in ("drain", "stop"):
                reason = signal
                break
            batch = collect_rollout(venv, ppo.policy, ppo.normalizer, cfg.rollout_len,
                                    policy_version=ppo.policy_version, rollout_id=ppo.last_rollout_id + 1,
                                    seed=cfg.seed, gamma=cfg.ppo.gamma)
            if ctl() == "stop":
                dirty = True  # normalizer/RNG advanced during the discarded rollout
                reason = "stop"  # discard in-flight rollout; normalizer drift from it is part of the state we save
                break
            stats = ppo.update(batch)
            rets = [e["return"] for e in batch.episodes]
            row = {"env_steps": ppo.env_steps, "policy_version": ppo.policy_version,
                   "mean_return": float(np.mean(rets)) if rets else None, **stats}
            history.append(row)
            dirty = True
            rounds += 1
            log(json.dumps(row))
            if rounds % cfg.checkpoint_every == 0:
                last, dirty = save(), False
        if dirty:
            last = save()
    return TrainResult(ppo.env_steps, reason, last, history)


def evaluate(ppo_or_agent: PPO | Agent, env_fn: Callable, episodes: int = 20, seed: int = 10_000,
             deterministic: bool = True) -> float:
    """Mean episode return with the normalizer frozen (restored afterwards). Never updates anything."""
    agent = ppo_or_agent if isinstance(ppo_or_agent, Agent) else Agent(ppo_or_agent.policy, ppo_or_agent.normalizer, "eval")
    was = agent.normalizer.frozen
    agent.normalizer.frozen = True
    try:
        rets = []
        for ep in range(episodes):
            env = env_fn()
            obs, _ = env.reset(seed=seed + ep)
            total, done = 0.0, False
            while not done:
                a, _ = agent.act({k: v[None] for k, v in obs.items()}, None, deterministic)
                obs, r, te, tr, _ = env.step(int(a[0]))
                total, done = total + r, te or tr
            rets.append(total)
        return float(np.mean(rets))
    finally:
        agent.normalizer.frozen = was


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.training.train")
    p.add_argument("--config", required=True, help="TrainConfig JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--control-file", help="file whose content is 'drain' or 'stop'")
    a = p.parse_args(argv)
    cfg = TrainConfig.model_validate_json(Path(a.config).read_text())
    ctl = control_from_file(a.control_file) if a.control_file else (lambda: None)
    res = train(cfg, a.out, resume=a.resume, control=ctl)
    print(json.dumps({"env_steps": res.env_steps, "stop_reason": res.stop_reason, "checkpoint": str(res.last_checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
