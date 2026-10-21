"""Training-only progress shaping for sparse attacker tasks. Reads engine state for the REWARD only; the
observation is untouched. Validation/benchmark use the unshaped env, and unshaped metrics are what we report."""
from __future__ import annotations

import gymnasium as gym


class AttackerShaping(gym.Wrapper):
    BONUS = {"node": 0.02, "access": 0.05, "cred": 0.05, "record": 0.01}

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def _snap(self):
        w, k = self.env.penv.world, self.env.penv.world.st.know
        return len(k.nodes), len(k.access), len(k.creds), w.st.m.records_accessed

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._best = self._snap()           # bonuses only for progress beyond the episode's best so far
        return obs, info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        cur = self._snap()
        bonus = sum(b * max(0, c - p) for b, c, p in zip(self.BONUS.values(), cur, self._best))
        self._best = tuple(max(c, p) for c, p in zip(cur, self._best))
        info = {**info, "unshaped_reward": r}
        return obs, r + self.scale * bonus, te, tr, info
