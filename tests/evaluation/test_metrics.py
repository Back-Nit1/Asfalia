import pytest

from hospital_rl.evaluation import benchmark
from hospital_rl.evaluation.metrics import _runs, aggregate, episode_objectives, visibility_shift_degradation
from hospital_rl.policies.scripted import GreedyDefender, Noop, ScriptedAttacker
from hospital_rl.runner import run_episode
from hospital_rl.schemas import load_scenario

SPEC = load_scenario("configs/tiny.yaml")


def test_outage_runs_censor_open_outage():
    assert _runs([2, 3, 4, 8, 18, 19], 20) == ([3, 1], [2])
    assert _runs([], 20) == ([], [])


def test_objectives_never_zero_fill_censored():
    ep = run_episode(SPEC, 7, ScriptedAttacker(), Noop())
    o = episode_objectives(ep)
    assert o["attacker_success"] == (ep["metrics"]["records_exfiltrated"] > 0)
    if not ep["metrics"]["contained"] and ep["metrics"]["first_access_tick"] is not None:
        assert o["containment"]["observed"] == [] and o["containment"]["censored"][0] > 0
    agg = aggregate([o])
    assert agg["containment"]["observed_mean"] is None or agg["containment"]["observed"]


def test_benign_episode_feeds_only_false_positives():
    class Idle:
        def act(self, view, legal):
            from hospital_rl.engine.transitions import WAIT
            return WAIT
    b = episode_objectives(run_episode(SPEC, 3, Idle(), GreedyDefender()), benign=True)
    assert not b["attacker_success"]
    agg = aggregate([b])
    assert agg["n_attack"] == 0 and "benign_false_positive_alerts" in agg and "attacker_success_rate" not in agg


def test_visibility_degradation_requires_paired_seeds():
    assert visibility_shift_degradation({1: 0.5, 2: 0.7}, {1: 0.4, 2: 0.4})["per_seed"] == pytest.approx([0.1, 0.3])
    with pytest.raises(ValueError):
        visibility_shift_degradation({1: 0.5}, {2: 0.4})
    with pytest.raises(ValueError):
        visibility_shift_degradation({}, {})


def test_benchmark_rows_carry_objectives():
    out = benchmark.run("validation", 2, size=12, horizon=20)
    o = out["defenders"]["noop"]["objectives"]
    assert o["n_attack"] == 2 and o["n_benign"] == 2 and "recovery" in o
