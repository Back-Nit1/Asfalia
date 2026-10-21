"""Shared policy/env contracts for training, league play, and inference.

Action layout (single-role env): index 0 is wait; otherwise ``1 + action_type * max_nodes + node``.
Observation: Dict of batched-or-single numpy arrays with the keys in ``OBS_KEYS``.

Two-role envs plug in through ``TwoRoleEnv`` (PettingZoo ParallelEnv-shaped: dicts keyed by role).
``FrozenOpponentEnv`` turns one into the single-role Gymnasium env the trainer expects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from torch import nn

Obs = dict[str, np.ndarray]
OBS_KEYS = ("node_features", "node_mask", "adjacency", "history", "globals", "action_mask")
ACTION_WAIT = 0
MASK_VALUE = -1e9  # large finite negative: exp() underflows to exactly 0, no inf-inf NaNs


def action_index(action_type: int, node: int, max_nodes: int) -> int:
    return 1 + action_type * max_nodes + node


def decode_action(action: int, max_nodes: int) -> tuple[int, int] | None:
    """(action_type, node), or None for wait."""
    return None if action == ACTION_WAIT else divmod(action - 1, max_nodes)


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Illegal actions get MASK_VALUE. Wait is forced legal so a row can never be fully masked."""
    mask = action_mask.clone()
    mask[:, ACTION_WAIT] = True
    return logits.masked_fill(~mask, MASK_VALUE)


class Policy(nn.Module, ABC):
    """forward(obs, state) -> (masked logits (B, N), value (B,), new state).

    Recurrent spec: ``state`` is None for feed-forward policies. A recurrent policy returns a
    (B, ...) tensor from ``initial_state`` and clears finished rows in ``reset_state``; the
    collector calls both, so a GRU can be added without touching the training loop.
    """

    @abstractmethod
    def initial_state(self, batch_size: int) -> torch.Tensor | None: ...

    @abstractmethod
    def forward(self, obs: dict[str, torch.Tensor], state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    def reset_state(self, state: torch.Tensor | None, done: torch.Tensor) -> torch.Tensor | None:
        """Reset rows where ``done`` (B,) is True to the initial state."""
        return state


class FrozenActor(Protocol):
    """A fixed opponent: scripted policy or frozen checkpoint. One observation in, one action out."""

    id: str

    def reset(self) -> None: ...

    def act(self, obs: Obs) -> int: ...


class TwoRoleEnv(Protocol):
    """Parallel two-role env (PettingZoo ParallelEnv shape). ``metadata`` carries catalog/obs hashes."""

    metadata: dict[str, Any]
    possible_agents: list[str]

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, Obs], dict[str, dict]]: ...

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, Obs], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]: ...

    def observation_space(self, agent: str) -> gym.Space: ...

    def action_space(self, agent: str) -> gym.Space: ...


class FrozenOpponentEnv(gym.Env):
    """Single-role view of a two-role env; the other role is driven by a FrozenActor."""

    def __init__(self, env: TwoRoleEnv, trainee: str, opponent: FrozenActor):
        roles = list(env.possible_agents)
        assert trainee in roles and len(roles) == 2, f"need two roles incl. {trainee}, got {roles}"
        self.env, self.trainee = env, trainee
        self.opponent_role = next(r for r in roles if r != trainee)
        self.opponent = opponent
        self.observation_space = env.observation_space(trainee)
        self.action_space = env.action_space(trainee)
        self.metadata = {**env.metadata, "role": trainee}

    def set_opponent(self, opponent: FrozenActor) -> None:
        """Takes effect at the next reset (never swap mid-episode)."""
        self._next_opponent = opponent

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.opponent = getattr(self, "_next_opponent", self.opponent)
        obs, infos = self.env.reset(seed=seed, options=options)
        self._opp_obs = obs[self.opponent_role]
        self.opponent.reset()
        return obs[self.trainee], {**infos[self.trainee], "opponent_id": self.opponent.id}

    def step(self, action):
        opp_action = self.opponent.act(self._opp_obs)
        obs, rew, term, trunc, infos = self.env.step({self.trainee: int(action), self.opponent_role: opp_action})
        self._opp_obs = obs[self.opponent_role]
        done_any = lambda d: any(d.values())  # noqa: E731 - episode ends when either role ends
        info = {**infos[self.trainee], "opponent_id": self.opponent.id}
        return obs[self.trainee], rew[self.trainee], done_any(term), done_any(trunc) and not done_any(term), info
