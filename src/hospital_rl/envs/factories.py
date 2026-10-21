"""Importable env factories ("module:callable") used by training configs."""
from __future__ import annotations

from ..evaluation.splits import scenario_for
from .curricula import build_env, load_config
from .parallel_env import HospitalParallelEnv
from .single_role import ScriptedOpponent, SingleRoleEnv


def school_env(config: str, level: int = 0, split: str = "train", seed: int = 0) -> SingleRoleEnv:
    cfg = load_config(config)
    env = build_env(cfg, level, split, seed)
    if split == "train" and cfg["role"] == "attacker" and cfg.get("shaping", 0):
        from .shaping import AttackerShaping
        env = AttackerShaping(env, cfg["shaping"])      # training only; validation stays unshaped
    return env


def hospital_penv(size: int = 40, split: str = "validation", difficulty: float = 0.7, horizon: int = 60,
                  base_logging: int = 0):
    """Two-role shared hospital world; each reset(seed) draws a scenario from the split."""
    return HospitalParallelEnv(
        lambda seed: scenario_for(split, _in_split(split, seed), size, difficulty, horizon, base_logging)
    )


def _in_split(split: str, seed: int) -> int:
    from ..evaluation.splits import SPLITS
    r = SPLITS[split]["seeds"]
    return r.start + seed % len(r)


def hospital_single(role: str, size: int = 40, split: str = "train", difficulty: float = 0.7, horizon: int = 60,
                    opponent: str = "scripted", base_logging: int = 0) -> SingleRoleEnv:
    """Shared-world single-role env against a scripted opponent (learned opponents are injected by the caller)."""
    from ..policies.scripted import GreedyDefender, ScriptedAttacker
    pol = ScriptedAttacker if role == "defender" else GreedyDefender
    return SingleRoleEnv(hospital_penv(size, split, difficulty, horizon, base_logging), role, ScriptedOpponent(pol))
