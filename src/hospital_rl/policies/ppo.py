"""Masked PPO: graph policy, running normalizer, sync vec env, rollout collection, update.

Minimal own implementation (no maintained lib supports Dict obs + action masks + our
padding-invariance and versioned-rollout requirements without wrapper code). One code path
serves both roles. Rollouts are on-policy and versioned; ``PPO.update`` rejects stale/duplicate ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn
from torch.distributions import Categorical

from hospital_rl.policies.base import ACTION_WAIT, OBS_KEYS, Obs, Policy, mask_logits


class StaleRolloutError(ValueError):
    pass


# ---------------------------------------------------------------- normalizer
class Normalizer:
    """Running mean/var for node_features (valid nodes only), history, globals. Fit only while not frozen."""

    DIMS = ("node_features", "history", "globals")

    def __init__(self, node_dim: int, hist_dim: int, glob_dim: int, clip: float = 10.0):
        dims = dict(zip(self.DIMS, (node_dim, hist_dim, glob_dim)))
        self.mean = {k: np.zeros(d, np.float64) for k, d in dims.items()}
        self.var = {k: np.ones(d, np.float64) for k, d in dims.items()}
        self.count = {k: 1e-4 for k in dims}
        self.clip, self.frozen = clip, False

    def _rows(self, obs: Obs) -> dict[str, np.ndarray]:
        return {
            "node_features": obs["node_features"][obs["node_mask"].astype(bool)],
            "history": obs["history"].reshape(-1, obs["history"].shape[-1]),
            "globals": obs["globals"].reshape(-1, obs["globals"].shape[-1]),
        }

    def update(self, obs: Obs) -> None:
        if self.frozen:
            return
        for k, x in self._rows(obs).items():
            if len(x) == 0:
                continue
            x = x.astype(np.float64)
            n, bm, bv = len(x), x.mean(0), x.var(0)
            tot = self.count[k] + n
            d = bm - self.mean[k]
            self.mean[k] = self.mean[k] + d * n / tot
            self.var[k] = (self.var[k] * self.count[k] + bv * n + d**2 * self.count[k] * n / tot) / tot
            self.count[k] = tot

    def __call__(self, obs: Obs) -> Obs:
        out = dict(obs)
        for k in self.DIMS:
            z = (obs[k] - self.mean[k]) / np.sqrt(self.var[k] + 1e-8)
            out[k] = np.clip(z, -self.clip, self.clip).astype(np.float32)
        out["node_features"] = np.where(obs["node_mask"].astype(bool)[..., None], out["node_features"], 0.0).astype(np.float32)
        return out

    def state_dict(self) -> dict:
        return {"mean": {k: torch.tensor(v) for k, v in self.mean.items()},
                "var": {k: torch.tensor(v) for k, v in self.var.items()},
                "count": dict(self.count), "frozen": self.frozen, "clip": self.clip}

    def load_state_dict(self, s: dict) -> None:
        self.mean = {k: v.numpy().astype(np.float64) for k, v in s["mean"].items()}
        self.var = {k: v.numpy().astype(np.float64) for k, v in s["var"].items()}
        self.count, self.frozen, self.clip = dict(s["count"]), bool(s["frozen"]), s["clip"]

    def stats_json(self) -> dict:
        return {"mean": {k: v.tolist() for k, v in self.mean.items()},
                "var": {k: v.tolist() for k, v in self.var.items()},
                "count": self.count, "frozen": self.frozen, "clip": self.clip}


# ---------------------------------------------------------------- policy
def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class GraphPolicy(Policy):
    """Node MLP + one masked neighbor-mean round, pooled (mean/max over valid nodes) context,
    per-node action-type heads + wait head + value head. Weights are independent of max_nodes,
    and padded nodes cannot influence valid-node outputs (masking uses ``where``, so even
    inf/NaN garbage in padding is inert)."""

    def __init__(self, node_dim: int, hist_dim: int, glob_dim: int, n_action_types: int, hidden: int = 64):
        super().__init__()
        self.config = dict(node_dim=node_dim, hist_dim=hist_dim, glob_dim=glob_dim,
                           n_action_types=n_action_types, hidden=hidden)
        self.n_types = n_action_types
        self.node_enc = _mlp(node_dim, hidden, hidden)
        self.msg = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        self.hist_enc = _mlp(hist_dim, hidden, hidden)
        self.glob_enc = _mlp(glob_dim, hidden, hidden)
        self.ctx = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU())
        self.node_head = _mlp(2 * hidden, hidden, n_action_types)
        self.wait_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def initial_state(self, batch_size):
        return None

    def forward(self, obs, state=None):
        nf, nm, adj = obs["node_features"], obs["node_mask"].bool(), obs["adjacency"].bool()
        b, m, _ = nf.shape
        mk = nm.unsqueeze(-1)
        zero = torch.zeros((), dtype=nf.dtype)
        nf = torch.where(mk, nf, zero)
        h = torch.where(mk, self.node_enc(nf), zero)
        a = (adj & nm.unsqueeze(1) & nm.unsqueeze(2)).to(h.dtype)
        nb = (a @ h) / a.sum(-1, keepdim=True).clamp(min=1)
        h = torch.where(mk, self.msg(torch.cat([h, nb], -1)), zero)
        mean = h.sum(1) / nm.sum(1, keepdim=True).clamp(min=1)
        mx = h.masked_fill(~mk, -1e9).amax(1)
        mx = torch.where(nm.any(1, keepdim=True), mx, zero)
        ctx = self.ctx(torch.cat([mean, mx, self.hist_enc(obs["history"]).mean(1),
                                  self.glob_enc(obs["globals"])], -1))
        node_logits = self.node_head(torch.cat([h, ctx.unsqueeze(1).expand(-1, m, -1)], -1))  # (B, M, T)
        logits = torch.cat([self.wait_head(ctx), node_logits.permute(0, 2, 1).reshape(b, -1)], 1)
        return mask_logits(logits, obs["action_mask"].bool()), self.value_head(ctx).squeeze(-1), None


def to_torch(obs: Obs) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(obs[k]) for k in OBS_KEYS}


# ---------------------------------------------------------------- inference wrapper
class Agent:
    """Policy + normalizer for acting. Normalizer is applied but never updated here."""

    def __init__(self, policy: Policy, normalizer: Normalizer, role: str):
        self.policy, self.normalizer, self.role = policy, normalizer, role

    @torch.no_grad()
    def act(self, obs: Obs, state=None, deterministic: bool = False, generator: torch.Generator | None = None):
        """Batched raw obs -> (actions (B,) int64 np, new state)."""
        logits, _, state = self.policy(to_torch(self.normalizer(obs)), state)
        act = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample() if generator is None \
            else torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)
        return act.numpy(), state


class FrozenAgent:
    """FrozenActor backed by an Agent (e.g. loaded from a checkpoint). Private RNG: never touches training RNG."""

    def __init__(self, agent: Agent, id: str, seed: int = 0, deterministic: bool = False):
        self.agent, self.id, self.deterministic = agent, id, deterministic
        self.gen = torch.Generator().manual_seed(seed)
        self.state = None
        self.agent.normalizer.frozen = True
        self.agent.policy.eval()

    def reset(self):
        self.state = self.agent.policy.initial_state(1)

    def act(self, obs: Obs) -> int:
        batch = {k: v[None] for k, v in obs.items()}
        a, self.state = self.agent.act(batch, self.state, self.deterministic, self.gen)
        return int(a[0])


# ---------------------------------------------------------------- vec env
class SyncVecEnv:
    """Single-process vectorized single-role envs with autoreset. Tracks per-episode return/components."""

    def __init__(self, fns: list[Callable]):
        self.envs = [f() for f in fns]
        self.n = len(self.envs)
        self.metadata = self.envs[0].metadata
        self.observation_space, self.action_space = self.envs[0].observation_space, self.envs[0].action_space

    @staticmethod
    def _stack(obs: list[Obs]) -> Obs:
        return {k: np.stack([o[k] for o in obs]) for k in OBS_KEYS}

    def reset(self, seeds: list[int]) -> Obs:
        self.ret, self.len, self.comp = np.zeros(self.n), np.zeros(self.n, int), [{} for _ in range(self.n)]
        return self._stack([e.reset(seed=s)[0] for e, s in zip(self.envs, seeds)])

    def step(self, actions):
        obs, rew, term, trunc, infos = [], np.zeros(self.n, np.float32), np.zeros(self.n, bool), np.zeros(self.n, bool), []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr, info = e.step(int(a))
            rew[i], term[i], trunc[i] = r, te, tr
            self.ret[i] += r
            self.len[i] += 1
            for k, v in info.get("reward_components", {}).items():
                self.comp[i][k] = self.comp[i].get(k, 0.0) + v
            info = dict(info)
            if te or tr:
                info["final_obs"] = o
                info["episode"] = {"return": float(self.ret[i]), "length": int(self.len[i]), "components": self.comp[i]}
                self.ret[i], self.len[i], self.comp[i] = 0.0, 0, {}
                o, _ = e.reset()  # env rng continues deterministically from the seeded first reset
            obs.append(o)
            infos.append(info)
        return self._stack(obs), rew, term, trunc, infos


# ---------------------------------------------------------------- rollouts
@dataclass
class Rollout:
    obs: dict[str, torch.Tensor]  # normalized, (T, B, ...)
    actions: torch.Tensor  # (T, B)
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor  # truncation bootstrap already folded in
    dones: torch.Tensor  # episode ended (terminated or truncated) at this step
    last_value: torch.Tensor  # (B,)
    policy_version: int
    rollout_id: int
    initial_state: torch.Tensor | None = None
    episodes: list[dict] = field(default_factory=list)


@torch.no_grad()
def collect_rollout(venv: SyncVecEnv, policy: Policy, normalizer: Normalizer, n_steps: int, *,
                    policy_version: int, rollout_id: int, seed: int, gamma: float) -> Rollout:
    """Envs are reset at the start of every rollout with seeds derived from (seed, rollout_id, env),
    so resume needs no env snapshot and seeds never repeat across rollouts or envs."""
    obs = venv.reset([seed * 1_000_003 + rollout_id * 1009 + i for i in range(venv.n)])
    state = init_state = policy.initial_state(venv.n)
    buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
    episodes = []
    for _ in range(n_steps):
        assert obs["action_mask"][:, ACTION_WAIT].all(), "contract violation: wait must always be legal"
        normalizer.update(obs)
        tobs = to_torch(normalizer(obs))
        logits, value, state = policy(tobs, state)
        dist = Categorical(logits=logits)
        act = dist.sample()
        obs, rew, term, trunc, infos = venv.step(act.numpy())
        rew = torch.as_tensor(rew)
        for i in np.flatnonzero(trunc & ~term):  # time-limit: bootstrap from the final obs
            fo = {k: v[None] for k, v in infos[i]["final_obs"].items()}
            rew[i] += gamma * policy(to_torch(normalizer(fo)), None if state is None else state[i:i + 1])[1][0]
        done = torch.as_tensor(term | trunc)
        episodes += [x["episode"] for x in infos if "episode" in x]
        for k, v in zip(buf, (tobs, act, dist.log_prob(act), value, rew, done)):
            buf[k].append(v)
        state = policy.reset_state(state, done)
    last_value = policy(to_torch(normalizer(obs)), state)[1]
    stack = lambda xs: torch.stack(xs)  # noqa: E731
    return Rollout(
        obs={k: stack([o[k] for o in buf["obs"]]) for k in OBS_KEYS},
        actions=stack(buf["act"]), logp=stack(buf["logp"]), values=stack(buf["val"]),
        rewards=stack(buf["rew"]), dones=stack(buf["done"]), last_value=last_value,
        policy_version=policy_version, rollout_id=rollout_id, initial_state=init_state, episodes=episodes)


# ---------------------------------------------------------------- PPO
class PPOConfig(BaseModel):
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5


class PPO:
    def __init__(self, policy: Policy, normalizer: Normalizer, cfg: PPOConfig):
        self.policy, self.normalizer, self.cfg = policy, normalizer, cfg
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.policy_version = 0
        self.last_rollout_id = -1
        self.env_steps = 0

    def gae(self, r: Rollout) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        adv = torch.zeros_like(r.rewards)
        nxt_v, nxt_a = r.last_value, torch.zeros_like(r.last_value)
        for t in reversed(range(len(r.rewards))):
            live = 1.0 - r.dones[t].float()
            delta = r.rewards[t] + c.gamma * live * nxt_v - r.values[t]
            nxt_a = delta + c.gamma * c.gae_lambda * live * nxt_a
            adv[t], nxt_v = nxt_a, r.values[t]
        return adv, adv + r.values

    def update(self, r: Rollout) -> dict[str, float]:
        if r.policy_version != self.policy_version:
            raise StaleRolloutError(f"rollout from policy v{r.policy_version}, learner is v{self.policy_version}")
        if r.rollout_id <= self.last_rollout_id:
            raise StaleRolloutError(f"duplicate/old rollout_id {r.rollout_id} (last consumed {self.last_rollout_id})")
        c = self.cfg
        adv, ret = self.gae(r)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        obs, act, old_logp, adv, ret = {k: flat(v) for k, v in r.obs.items()}, flat(r.actions), flat(r.logp), flat(adv), flat(ret)
        n = len(act)
        stats: dict[str, list[float]] = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for idx in perm.chunk(c.minibatches):
                logits, value, _ = self.policy({k: v[idx] for k, v in obs.items()}, None)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[idx])
                ratio = (logp - old_logp[idx]).exp()
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8) if len(a) > 1 else a
                pl = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vl = 0.5 * (value - ret[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + c.vf_coef * vl - c.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.optimizer.step()
                for k, v in zip(stats, (pl, vl, ent, (old_logp[idx] - logp).mean(), ((ratio - 1).abs() > c.clip).float().mean())):
                    stats[k].append(float(v.detach()))
        self.policy_version += 1
        self.last_rollout_id = r.rollout_id
        self.env_steps += r.rewards.numel()
        return {k: float(np.mean(v)) for k, v in stats.items()}
