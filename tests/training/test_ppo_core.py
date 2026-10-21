import numpy as np
import pytest
import torch

from hospital_rl.policies.base import FrozenOpponentEnv
from hospital_rl.policies.ppo import (PPO, Agent, FrozenAgent, GraphPolicy, Normalizer, PPOConfig,
                                      StaleRolloutError, SyncVecEnv, collect_rollout, to_torch)
from toy_env import E, F, G, H, N_TYPES, ToyDuelEnv, ToyGraphEnv


def make_policy(seed=0):
    torch.manual_seed(seed)
    return GraphPolicy(F, E, G, N_TYPES, hidden=16)


def batch_obs(env, n, seed=0):
    obs, _ = env.reset(seed=seed)
    out = []
    for _ in range(n):
        out.append(obs)
        obs, *_ = env.step(int(np.random.default_rng(seed + len(out)).choice(np.flatnonzero(obs["action_mask"]))))
        if env.t == 0:
            obs, _ = env.reset()
    return {k: np.stack([o[k] for o in out]) for k in out[0]}


def test_masked_logits_never_pick_illegal():
    env, pol = ToyGraphEnv(), make_policy()
    obs = batch_obs(env, 2000)
    for lo in range(0, 2000, 500):
        sl = {k: v[lo:lo + 500] for k, v in obs.items()}
        logits, _, _ = pol(to_torch(sl))
        for _ in range(5):
            a = torch.distributions.Categorical(logits=logits).sample().numpy()
            assert sl["action_mask"][np.arange(500), a].all()
        assert sl["action_mask"][np.arange(500), logits.argmax(-1).numpy()].all()


def test_wait_forced_legal():
    pol = make_policy()
    obs = to_torch(batch_obs(ToyGraphEnv(), 4))
    obs["action_mask"] = torch.zeros_like(obs["action_mask"])
    logits, _, _ = pol(obs)
    assert torch.isfinite(logits).all() and (logits.argmax(-1) == 0).all()


def test_padding_invariance_and_node_permutation():
    pol, env = make_policy(), ToyGraphEnv(max_nodes=8)
    pol.eval()
    obs = batch_obs(env, 6)
    nm = obs["node_mask"].astype(bool)
    base, bval, _ = pol(to_torch(obs))

    # garbage (even inf/nan) in padded node rows and adjacency must not change anything
    junk = {k: v.copy() for k, v in obs.items()}
    junk["node_features"][~nm] = np.where(np.random.default_rng(0).random(((~nm).sum(), F)) > 0.5, np.inf, np.nan)
    junk["adjacency"][~nm] = True
    junk["adjacency"] |= ~nm[:, None, :]
    out, val, _ = pol(to_torch(junk))
    assert torch.equal(out, base) and torch.equal(val, bval)

    # embedding in a larger max_nodes: same logits for the same (type, node) actions
    m2 = 12
    big = {"node_features": np.zeros((6, m2, F), np.float32), "node_mask": np.zeros((6, m2), bool),
           "adjacency": np.zeros((6, m2, m2), bool), "history": obs["history"], "globals": obs["globals"],
           "action_mask": np.zeros((6, 1 + N_TYPES * m2), bool)}
    big["node_features"][:, :8], big["node_mask"][:, :8], big["adjacency"][:, :8, :8] = obs["node_features"], nm, obs["adjacency"]
    big["action_mask"][:, 0] = True
    for t in range(N_TYPES):
        big["action_mask"][:, 1 + t * m2:1 + t * m2 + 8] = obs["action_mask"][:, 1 + t * 8:1 + t * 8 + 8]
    bl, bv, _ = pol(to_torch(big))
    assert torch.allclose(bv, bval, atol=1e-5)
    for t in range(N_TYPES):
        a = base[:, 1 + t * 8:1 + t * 8 + 8]
        b = bl[:, 1 + t * m2:1 + t * m2 + 8]
        legal = torch.as_tensor(obs["action_mask"][:, 1 + t * 8:1 + t * 8 + 8])
        assert torch.allclose(a[legal], b[legal], atol=1e-5)

    # permuting valid nodes permutes node logits and leaves value / wait unchanged
    perm = np.random.default_rng(1).permutation(8)
    pm = {k: v.copy() for k, v in obs.items()}
    pm["node_features"], pm["node_mask"] = obs["node_features"][:, perm], nm[:, perm]
    pm["adjacency"] = obs["adjacency"][:, perm][:, :, perm]
    for t in range(N_TYPES):
        pm["action_mask"][:, 1 + t * 8:1 + t * 8 + 8] = obs["action_mask"][:, 1 + t * 8:1 + t * 8 + 8][:, perm]
    pl, pv, _ = pol(to_torch(pm))
    assert torch.allclose(pv, bval, atol=1e-5)
    for t in range(N_TYPES):
        sl = slice(1 + t * 8, 1 + t * 8 + 8)
        legal = torch.as_tensor(pm["action_mask"][:, sl])
        assert torch.allclose(pl[:, sl][legal], base[:, sl][:, perm][legal], atol=1e-5)


def make_ppo(seed=0, **kw):
    pol = make_policy(seed)
    return PPO(pol, Normalizer(F, E, G), PPOConfig(**kw))


def roll(ppo, venv, rid=None, **kw):
    return collect_rollout(venv, ppo.policy, ppo.normalizer, 32, policy_version=ppo.policy_version,
                           rollout_id=ppo.last_rollout_id + 1 if rid is None else rid, seed=0, gamma=0.99)


def test_weights_change_and_stale_rejected():
    torch.manual_seed(0)
    ppo, venv = make_ppo(), SyncVecEnv([ToyGraphEnv] * 4)
    before = [p.clone() for p in ppo.policy.parameters()]
    r = roll(ppo, venv)
    ppo.update(r)
    assert any(not torch.equal(a, b) for a, b in zip(before, ppo.policy.parameters()))
    with pytest.raises(StaleRolloutError, match="policy v0"):
        ppo.update(r)  # same batch: stale version (and duplicate)
    r2 = roll(ppo, venv)
    r2.rollout_id = 0
    with pytest.raises(StaleRolloutError, match="duplicate"):
        ppo.update(r2)
    assert ppo.policy_version == 1


def test_normalizer_frozen_does_not_fit():
    n = Normalizer(F, E, G)
    obs = batch_obs(ToyGraphEnv(), 50)
    n.update(obs)
    mean = n.mean["node_features"].copy()
    n.frozen = True
    n.update({**obs, "node_features": obs["node_features"] + 5})
    assert np.array_equal(mean, n.mean["node_features"])
    assert (n(obs)["node_features"][~obs["node_mask"]] == 0).all()


def test_frozen_opponent_adapter():
    class Const:
        id = "const"
        resets = 0

        def reset(self):
            self.resets += 1

        def act(self, obs):
            return 0

    opp = Const()
    env = FrozenOpponentEnv(ToyDuelEnv(), "attacker", opp)
    assert env.metadata["role"] == "attacker" and env.action_space.n == 1 + N_TYPES * 8
    obs, info = env.reset(seed=3)
    assert info["opponent_id"] == "const" and opp.resets == 1
    done, steps = False, 0
    while not done:
        obs, r, te, tr, info = env.step(0)
        done, steps = te or tr, steps + 1
    assert steps == 16
    # a frozen learned agent also works as opponent, with a private RNG
    pol = make_policy()
    fa = FrozenAgent(Agent(pol, Normalizer(F, E, G), "defender"), "ckpt-x")
    state = torch.get_rng_state()
    env = FrozenOpponentEnv(ToyDuelEnv(), "attacker", fa)
    env.reset(seed=1)
    env.step(0)
    assert torch.equal(state, torch.get_rng_state())
    assert fa.agent.normalizer.frozen
