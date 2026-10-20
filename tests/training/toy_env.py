"""Toy masked-Discrete env with the real padded-graph observation contract.

Task (learnable): one valid node is flagged in node_features[:, 0] each step; action type 0
on that node pays +1. Type-1 actions are legal only on even nodes and cost a little.
Random play earns ~2 per 16-step episode, the optimum is 16.
"""
from __future__ import annotations

import hashlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces

F, H, E, G = 4, 3, 2, 3
N_TYPES = 2


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class ToyGraphEnv(gym.Env):
    def __init__(self, max_nodes: int = 8, episode_len: int = 16, role: str = "defender"):
        self.max_nodes, self.episode_len = max_nodes, episode_len
        n = 1 + N_TYPES * max_nodes
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(-np.inf, np.inf, (max_nodes, F), np.float32),
            "node_mask": spaces.MultiBinary(max_nodes),
            "adjacency": spaces.MultiBinary((max_nodes, max_nodes)),
            "history": spaces.Box(-np.inf, np.inf, (H, E), np.float32),
            "globals": spaces.Box(-np.inf, np.inf, (G,), np.float32),
            "action_mask": spaces.MultiBinary(n),
        })
        self.metadata = {
            "role": role,
            "catalog_hash": _hash(f"toy-catalog-{N_TYPES}"),
            "observation_schema_hash": _hash(f"toy-obs-{F}-{H}-{E}-{G}"),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.n_valid = int(self.np_random.integers(3, self.max_nodes + 1))
        self.t = 0
        self.history = np.zeros((H, E), np.float32)
        self._draw_target()
        return self._obs(), {}

    def _draw_target(self):
        self.target = int(self.np_random.integers(self.n_valid))
        self.noise = self.np_random.standard_normal(self.max_nodes).astype(np.float32)

    def _obs(self):
        m, v = self.max_nodes, self.n_valid
        node_mask = np.arange(m) < v
        nf = np.zeros((m, F), np.float32)
        nf[self.target, 0] = 1.0
        nf[:, 1] = self.noise
        nf[:, 2] = np.arange(m) / m
        nf[:, 3] = 1.0
        nf[~node_mask] = 0.0
        adj = np.zeros((m, m), bool)
        for i in range(v):  # ring over valid nodes
            adj[i, (i + 1) % v] = adj[(i + 1) % v, i] = True
        amask = np.zeros(1 + N_TYPES * m, bool)
        amask[0] = True
        amask[1:1 + m] = node_mask
        amask[1 + m:1 + 2 * m] = node_mask & (np.arange(m) % 2 == 0)
        return {
            "node_features": nf, "node_mask": node_mask, "adjacency": adj,
            "history": self.history.copy(),
            "globals": np.array([self.t / self.episode_len, v / m, 1.0], np.float32),
            "action_mask": amask,
        }

    def step(self, action):
        a = int(action)
        assert self._obs()["action_mask"][a], f"illegal action {a}"
        hit, cost = 0.0, 0.0
        if a > 0:
            t, i = divmod(a - 1, self.max_nodes)
            hit = float(t == 0 and i == self.target)
            cost = -0.05 if t == 1 else 0.0
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = (min(a, 1), hit)
        self.t += 1
        self._draw_target()
        trunc = self.t >= self.episode_len
        return self._obs(), hit + cost, False, trunc, {"reward_components": {"hit": hit, "cost": cost}}


def make_env(**kw) -> ToyGraphEnv:
    return ToyGraphEnv(**kw)


class ToyDuelEnv:
    """PettingZoo-parallel-shaped two-role env: two independent toy tasks in lockstep."""

    possible_agents = ["attacker", "defender"]

    def __init__(self, max_nodes: int = 8, episode_len: int = 16):
        self.envs = {r: ToyGraphEnv(max_nodes, episode_len, r) for r in self.possible_agents}
        self.metadata = dict(self.envs["defender"].metadata)

    def observation_space(self, agent):
        return self.envs[agent].observation_space

    def action_space(self, agent):
        return self.envs[agent].action_space

    def reset(self, seed=None, options=None):
        out = {r: e.reset(seed=None if seed is None else seed + i) for i, (r, e) in enumerate(self.envs.items())}
        return {r: o for r, (o, _) in out.items()}, {r: i for r, (_, i) in out.items()}

    def step(self, actions):
        out = {r: self.envs[r].step(actions[r]) for r in self.envs}
        return tuple({r: out[r][k] for r in out} for k in range(5))
