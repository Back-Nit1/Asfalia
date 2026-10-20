import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from hospital_rl.engine.transitions import IllegalAction
from hospital_rl.envs.encoding import Codec, n_actions
from hospital_rl.envs.parallel_env import HospitalParallelEnv
from hospital_rl.envs.single_role import ScriptedOpponent, SingleRoleEnv
from hospital_rl.policies.scripted import GreedyDefender, ScriptedAttacker
from hospital_rl.schemas import load_scenario

SPEC = load_scenario("configs/tiny.yaml")


def test_pettingzoo_api():
    parallel_api_test(HospitalParallelEnv(SPEC), num_cycles=50)


def sample(env, obs, rng):
    return {r: int(rng.choice(np.flatnonzero(obs[r]["action_mask"]))) for r in env.agents}


def test_random_legal_episodes_keep_spaces_and_masks_valid():
    env = HospitalParallelEnv(SPEC)
    rng = np.random.default_rng(0)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        while env.agents:
            for r in env.agents:
                assert env.observation_space(r).contains(obs[r]) and obs[r]["action_mask"][0] == 1
            obs, rew, term, trunc, _ = env.step(sample(env, obs, rng))
            assert all(np.isfinite(v) for v in rew.values())


def test_padding_slots_masked():
    env = HospitalParallelEnv(SPEC)
    obs, _ = env.reset(seed=1)
    n = len(SPEC.nodes)
    for r in env.agents:
        m = obs[r]["action_mask"][1:].reshape(-1, 64)
        assert not m[:, n:].any() and not obs[r]["node_features"][n:].any()


def test_index_errors():
    env = HospitalParallelEnv(SPEC)
    env.reset(seed=1)
    with pytest.raises(IllegalAction):
        env.step({"attacker": n_actions("attacker"), "defender": 0})
    with pytest.raises(IllegalAction):
        env.step({"attacker": 1 + 63, "defender": 0})          # padding slot
    env.reset(seed=1)
    with pytest.raises(IllegalAction):
        env.step({"attacker": 0, "defender": 1 + 64 * 6})   # restore_connectivity on a non-isolated node


def test_slot_permutation_deterministic_and_varies():
    ids = SPEC.node_ids()
    assert Codec(ids, 3).slot_node == Codec(ids, 3).slot_node
    assert Codec(ids, 3).slot_node != Codec(ids, 4).slot_node


def test_hidden_state_perturbation_does_not_change_obs():
    e1, e2 = HospitalParallelEnv(SPEC), HospitalParallelEnv(SPEC)
    o1, _ = e1.reset(seed=2)
    o2, _ = e2.reset(seed=2)
    e2.world.st.nodes["pacs"].implant = True
    e2.world.topo.node["ehr"].vulns.append("priv_esc")
    e2.world.st.cred_valid["c_admin"] = False
    for _ in range(3):
        a = {"attacker": 0, "defender": 0}
        # identical observations as long as no emitted evidence differs (perturbation emits none while idle)
        for r in ("attacker", "defender"):
            for k in o1[r]:
                np.testing.assert_array_equal(o1[r][k], o2[r][k])
        o1, *_ = e1.step(a)
        o2, *_ = e2.step(a)


def test_simultaneous_actions_use_pre_step_obs():
    class Spy:
        def __init__(self): self.seen = None
        def __call__(self, penv, role):
            self.seen = penv.world.st.tick
            return 0
    spy = Spy()
    env = SingleRoleEnv(HospitalParallelEnv(SPEC), "defender", spy)
    env.reset(seed=1)
    env.step(0)
    assert spy.seen == 0                                           # opponent chose before the tick advanced


def test_single_role_runs_full_episode_vs_scripted():
    env = SingleRoleEnv(HospitalParallelEnv(SPEC), "defender", ScriptedOpponent(ScriptedAttacker))
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    done, total = False, 0
    while not done:
        a = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        obs, r, term, trunc, info = env.step(a)
        done, total = term or trunc, total + r
    assert total <= 0


def test_no_execution_surface():
    # actions are catalog indices only; there is no string/command channel in the env API
    env = HospitalParallelEnv(SPEC)
    env.reset(seed=1)
    with pytest.raises((TypeError, ValueError)):
        env.step({"attacker": "rm -rf /", "defender": 0})
