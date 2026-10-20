"""Gymnasium single-agent view of one role against a frozen opponent."""
from __future__ import annotations

import gymnasium as gym

from .encoding import ROLES_OTHER
from .parallel_env import HospitalParallelEnv


class ScriptedOpponent:
    """Wraps a scripted policy (act(view, legal)) as an opponent returning a catalog index."""

    def __init__(self, factory):
        self.factory, self.policy = factory, None

    def reset(self):
        self.policy = self.factory()

    def __call__(self, penv: HospitalParallelEnv, role: str) -> int:
        w = penv.world
        legal = w.legal_attacker() if role == "attacker" else w.legal_defender()
        return penv.codec.encode(role, self.policy.act(w.view(role), legal))


class SingleRoleEnv(gym.Env):
    """opponent: callable(parallel_env, opp_role) -> catalog index. Learned opponents plug in the same way,
    reading their own role-local observation from the parallel env."""

    def __init__(self, penv: HospitalParallelEnv, role: str, opponent):
        self.penv, self.role, self.opp = penv, role, opponent
        self.opp_role = ROLES_OTHER[role]
        self.observation_space = penv.observation_space(role)
        self.action_space = penv.action_space(role)
        self.metadata = {**penv.metadata, "role": role}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, _ = self.penv.reset(seed=seed)
        if hasattr(self.opp, "reset"):
            self.opp.reset()
        return obs[self.role], {}

    def step(self, action):
        opp_idx = self.opp(self.penv, self.opp_role)         # chosen from pre-step state, before our action lands
        obs, rew, term, trunc, info = self.penv.step({self.role: int(action), self.opp_role: opp_idx})
        return obs[self.role], rew[self.role], term[self.role], trunc[self.role], info[self.role]
