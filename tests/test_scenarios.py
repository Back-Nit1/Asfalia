import pytest

from hospital_rl.engine.transitions import World
from hospital_rl.envs.parallel_env import HospitalParallelEnv
from hospital_rl.evaluation.splits import SPLITS, scenario_for, split_hash
from hospital_rl.policies.scripted import GreedyDefender, ScriptedAttacker
from hospital_rl.runner import run_episode
from hospital_rl.scenarios import FAMILIES, generate


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("size", [8, 24, 40])
def test_generated_worlds_valid_and_connected(family, size):
    spec = generate(family, size, seed=3)
    ids = set(spec.node_ids())
    assert len(ids) == len(spec.nodes) <= 64 and len(ids) == size
    assert all(e.src in ids and e.dst in ids for e in spec.edges)
    assert all(set(w.path) <= ids for w in spec.workflows)
    w = World(spec, 0)
    # every workflow is satisfiable at t=0 (benign-only episode has no failures)
    from hospital_rl.policies.scripted import Noop
    r = run_episode(spec, 0, Noop(), Noop())
    assert r["metrics"]["critical_success"] == 1.0, r["metrics"]


def test_splits_disjoint_by_family_and_seed():
    fams = [set(v["families"]) for v in SPLITS.values()]
    assert not (fams[0] & fams[1]) and not (fams[1] & fams[2]) and not (fams[0] & fams[2])
    with pytest.raises(ValueError):
        scenario_for("test", 5)
    assert scenario_for("test", 20_001).name.startswith("segmented")
    assert len(split_hash()) == 16


def test_scripted_episode_on_hospital40():
    spec = generate("tiered", 40, seed=5)
    r = run_episode(spec, 5, ScriptedAttacker(), GreedyDefender())
    assert len(r["steps"]) == spec.horizon


def test_env_on_largest_world():
    env = HospitalParallelEnv(lambda s: scenario_for("train", s, size=40))
    obs, _ = env.reset(seed=1)
    assert env.observation_space("defender").contains(obs["defender"])
