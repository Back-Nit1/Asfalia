import json

from hospital_rl.data import trajectory
from hospital_rl.envs.curricula import build_env, load_config
from hospital_rl.evaluation import counterfactual as cf
from hospital_rl.policies.scripted import GreedyDefender, Noop, ScriptedAttacker, StaticHardening
from hospital_rl.runner import run_episode
from hospital_rl.schemas import load_scenario

SPEC = load_scenario("configs/tiny.yaml")


def test_trajectory_export_validates_and_separates_privileged(tmp_path):
    ep = run_episode(SPEC, 7, ScriptedAttacker(), GreedyDefender(), record=True)
    info = trajectory.export(ep, str(tmp_path), "e1", {"attacker": "scripted", "defender": "scripted"})
    assert trajectory.validate(str(tmp_path / "e1.jsonl")) == info["rows"] == 2 * SPEC.horizon
    assert json.load(open(tmp_path / "e1.privileged.json"))["PRIVILEGED"] is True
    assert "truth" not in open(tmp_path / "e1.jsonl").read()


def test_fixed_intent_blocked_actions_fail():
    ep = run_episode(SPEC, 7, ScriptedAttacker(), Noop())
    out = cf.fixed_intent(ep, {"noop": Noop, "greedy": GreedyDefender})
    r = out["results"]
    assert r["noop"]["records_exfiltrated"] == ep["metrics"]["records_exfiltrated"]   # same branch reproduces itself
    assert r["greedy"]["records_exfiltrated"] <= r["noop"]["records_exfiltrated"]


def test_adaptive_reports_uncertainty():
    out = cf.adaptive(SPEC, list(range(6)), ScriptedAttacker, {"noop": Noop, "static": StaticHardening})
    assert out["results"]["noop"]["weighted_exposure_frac"]["n"] == 6 and "ADAPTIVE" in out["estimand"]


def test_school_envs_build_and_run():
    for path in ("configs/attacker_school.yaml", "configs/defender_school.yaml"):
        cfg = load_config(path)
        for lv in range(len(cfg["levels"])):
            env = build_env(cfg, lv)
            obs, _ = env.reset(seed=lv)
            obs, r, te, tr, _ = env.step(0)
            assert env.observation_space.contains(obs)
