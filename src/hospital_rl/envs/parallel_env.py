"""PettingZoo ParallelEnv over the shared engine. Both actions are chosen from pre-step observations."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..engine.transitions import World
from ..schemas import MAX_NODES, ScenarioSpec, catalog_hash
from .encoding import (E_DIM, F_DIM, G_DIM, HIST, Codec, encode_obs, n_actions, obs_schema_hash, step_record)

ROLES = ["attacker", "defender"]


class HospitalParallelEnv(ParallelEnv):
    metadata = {"name": "hospital_v0", "render_modes": []}

    def __init__(self, scenario: ScenarioSpec | Callable[[int], ScenarioSpec]):
        self._scenario = scenario if callable(scenario) else (lambda seed, s=scenario: s)
        self.possible_agents = list(ROLES)
        self.agents: list[str] = []
        self.world: World | None = None       # evaluator-side handle; never exposed through obs/info
        spec = self._scenario(0)
        self.metadata = {**self.metadata, "catalog_hash": catalog_hash(), "tie_policy": spec.tie_policy,
                         "observation_schema_hash": {r: obs_schema_hash(r) for r in ROLES},
                         "max_nodes": MAX_NODES}

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Dict:
        return spaces.Dict({
            "node_features": spaces.Box(-1.0, 1.0, (MAX_NODES, F_DIM[agent]), np.float32),
            "node_mask": spaces.MultiBinary(MAX_NODES),
            "adjacency": spaces.MultiBinary((MAX_NODES, MAX_NODES)),
            "history": spaces.Box(-1.0, 8.0, (HIST, E_DIM), np.float32),
            "globals": spaces.Box(-1.0, 2.0, (G_DIM,), np.float32),
            "action_mask": spaces.MultiBinary(n_actions(agent)),
        })

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(n_actions(agent))

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.episode_seed = 0 if seed is None else int(seed)
        self.spec = self._scenario(self.episode_seed)
        self.world = World(self.spec, self.episode_seed)
        self.codec = Codec(self.spec.node_ids(), self.episode_seed)
        self.agents = list(ROLES)
        self._hist = {r: [] for r in ROLES}
        return self._obs(), {r: {} for r in ROLES}

    def _obs(self):
        return {r: encode_obs(self.world, self.codec, r, self._hist[r]) for r in self.agents}

    def step(self, actions: dict[str, int]):
        a = self.codec.decode("attacker", int(actions["attacker"]))
        d = self.codec.decode("defender", int(actions["defender"]))
        info = self.world.step(a, d)             # raises IllegalAction for masked-out actions
        for r, own in (("attacker", a), ("defender", d)):
            self._hist[r].append(step_record(self.world, self.codec, r, info["events"], own))
        done = info["done"]
        rew = info["reward"]
        infos = {r: {"reward_components": info["components"][r]} for r in ROLES}
        trunc = {r: done for r in ROLES}
        term = {r: False for r in ROLES}
        obs = self._obs()
        if done:
            self.agents = []
        return obs, rew, term, trunc, infos
