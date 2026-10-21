"""Curriculum levels -> scenario generators and scripted opponent mixes (config-driven)."""
from __future__ import annotations

import random

import yaml

from ..evaluation.splits import SPLITS
from ..evaluation.strata import check_training_config, require_trainable
from ..policies.scripted import GreedyDefender, IsolateEverything, Noop, RandomPolicy, ScriptedAttacker, StaticHardening
from ..scenarios import generate
from .parallel_env import HospitalParallelEnv
from .single_role import ScriptedOpponent, SingleRoleEnv

OPPONENTS = {"noop": Noop, "static": StaticHardening, "greedy": GreedyDefender, "isolate_all": IsolateEverything,
             "scripted_attacker": ScriptedAttacker, "random": RandomPolicy}


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    check_training_config(cfg, path)
    return cfg


def scenario_fn(cfg: dict, level: int, split: str = "train"):
    lv = cfg["levels"][level]
    fams, wls, seeds = SPLITS[split]["families"], SPLITS[split]["workloads"], SPLITS[split]["seeds"]

    def make(seed: int):
        r = random.Random(seed)
        s = seeds.start + seed % len(seeds)               # keep seeds inside the split's range
        return generate(r.choice(fams), lv["size"], s, lv.get("difficulty", 0.5), r.choice(wls),
                        cfg.get("horizon", 40), lv.get("base_logging", 0))
    return make


class _Mixture:
    """Samples one scripted opponent from the level's mix at each episode reset."""

    def __init__(self, names, seed=0):
        self.names, self.r, self.inner = names, random.Random(seed), None

    def reset(self):
        n = self.r.choice(self.names)
        self.inner = ScriptedOpponent(OPPONENTS[n])
        self.inner.reset()

    def __call__(self, penv, role):
        return self.inner(penv, role)


def build_env(cfg: dict, level: int = 0, split: str = "train", seed: int = 0) -> SingleRoleEnv:
    require_trainable(split, "curriculum")
    penv = HospitalParallelEnv(scenario_fn(cfg, level, split))
    return SingleRoleEnv(penv, cfg["role"], _Mixture(cfg["levels"][level]["opponents"], seed))
