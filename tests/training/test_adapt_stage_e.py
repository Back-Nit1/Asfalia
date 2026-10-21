import json
import random
from pathlib import Path

import pytest

from hospital_rl.policies.checkpoint import IncompatibleCheckpointError, build_ppo, load_checkpoint
from hospital_rl.policies.ppo import PPOConfig, StaleRolloutError, SyncVecEnv, collect_rollout
from hospital_rl.evaluation.transfer import CheckpointError, load_actor
from hospital_rl.training import adapt as A
from hospital_rl.training.league import Entry, League
from hospital_rl.training.lineage import LineageError
from hospital_rl.training.world import WorldConfigError

FAST = dict(eval_every=64, val_episodes=1, n_envs=2, rollout_len=16)


def run(school_ckpts, world_path, out, **kw):
    a = dict(role="defender", attacker_checkpoint=school_ckpts["attacker"], defender_checkpoint=school_ckpts["defender"],
             world_config=world_path, steps=64, seed=3, out=out, league=False, log=lambda s: None, **FAST)
    return A.adapt(**{**a, **kw})


def test_one_side_adapt_trains_only_role_and_freezes_opponent(school_ckpts, world_path, tmp_path):
    opp = load_actor(school_ckpts["attacker"], "attacker")
    before = A.frozen_digest(opp)
    m = run(school_ckpts, world_path, tmp_path / "a")
    meta = load_checkpoint(m["checkpoints"]["best_path"]).meta
    assert meta.role == "defender" and meta.training_steps == 64 and meta.parent_checkpoint_id is not None
    assert m["opponent_frozen"]["unchanged"] and m["budget"]["adapt_env_steps"] == 64
    assert A.frozen_digest(load_actor(school_ckpts["attacker"], "attacker")) == before
    assert all(not p.requires_grad for p in opp.agent.agent.policy.parameters()) and opp.agent.agent.normalizer.frozen


def test_frozen_opponent_never_updates_through_env(school_ckpts, world_path, tmp_path):
    lg = tmp_path / "l.json"
    A.build_league(lg, "attacker", school_ckpts["attacker"], False)
    env = A.adapt_env("defender", {"size": 12, "horizon": 20, "difficulty": 0.5, "base_logging": 1}, str(lg), "train")
    cur = env.opp
    env.reset(seed=1)
    d0 = A.frozen_digest(cur.cur)
    for _ in range(20):
        env.step(0)
    assert A.frozen_digest(cur.cur) == d0


def test_scratch_has_no_parent_and_same_budget(school_ckpts, world_path, tmp_path):
    m = run(school_ckpts, world_path, tmp_path / "s", init="scratch")
    assert load_checkpoint(m["checkpoints"]["best_path"]).meta.parent_checkpoint_id is None
    assert m["budget"] == {"requested_adapt_steps": 64, "adapt_env_steps": 64, "pretrain_env_steps": 0, "total_env_steps": 64}


def test_pretrain_budget_accounting(school_ckpts, world_path, tmp_path):
    m = run(school_ckpts, world_path, tmp_path / "p", pretrain_steps=1000)
    assert m["budget"]["total_env_steps"] == 1064


def test_resume_extends_without_recount_and_finished_run_is_noop(school_ckpts, world_path, tmp_path):
    out = tmp_path / "r"
    run(school_ckpts, world_path, out, steps=64)
    rid = load_checkpoint(out / "latest.pt").meta.last_rollout_id
    m = run(school_ckpts, world_path, out, steps=128)
    meta = load_checkpoint(out / "latest.pt").meta
    assert m["resumed"] and meta.training_steps == 128 and meta.last_rollout_id > rid
    n = len(list(out.glob("ckpt_*.pt")))
    run(school_ckpts, world_path, out, steps=128)
    assert len(list(out.glob("ckpt_*.pt"))) == n and load_checkpoint(out / "latest.pt").meta.training_steps == 128


def test_lineage_rejects_unrelated_runs(school_ckpts, world_path, tmp_path):
    out = tmp_path / "l"
    run(school_ckpts, world_path, out)
    with pytest.raises(LineageError, match="seed"):
        run(school_ckpts, world_path, out, seed=4)
    with pytest.raises(LineageError, match="init"):
        run(school_ckpts, world_path, out, init="scratch")
    stray = tmp_path / "stray"; stray.mkdir(); (stray / "x.txt").write_text("hi")
    with pytest.raises(LineageError, match="no lineage"):
        run(school_ckpts, world_path, stray)
    assert (stray / "x.txt").read_text() == "hi"


def test_missing_and_incompatible_inputs(school_ckpts, world_path, tmp_path):
    with pytest.raises(CheckpointError, match="attacker checkpoint not found"):
        run(school_ckpts, world_path, tmp_path / "m", attacker_checkpoint=str(tmp_path / "gone.pt"))
    with pytest.raises(IncompatibleCheckpointError, match="role"):
        run(school_ckpts, world_path, tmp_path / "i", attacker_checkpoint=school_ckpts["defender"])
    with pytest.raises(WorldConfigError, match="not found"):
        run(school_ckpts, world_path, tmp_path / "w", world_config=str(tmp_path / "nope.yaml"))
    assert not (tmp_path / "m").exists() and not (tmp_path / "i").exists()      # nothing created on bad input
    args = ["--trainable-role", "defender", "--attacker-checkpoint", str(tmp_path / "gone.pt"),
            "--defender-checkpoint", school_ckpts["defender"], "--world-config", world_path, "--steps", "64",
            "--seed", "1", "--out", str(tmp_path / "c"), "--league", "off"]
    assert A.main(args) == 2
    with pytest.raises(SystemExit):                       # --league must be explicit
        A.main(args[:-2])


def test_league_persistence_disjoint_and_eviction(tmp_path):
    lg = League(max_checkpoints=2)
    lg.add_train(Entry("scripted:X", "scripted"))
    for i in range(4):
        lg.add_train(Entry(f"c{i}", "checkpoint", f"/p/{i}"))
    lg.add_eval(Entry("scripted:Held", "scripted"))
    with pytest.raises(ValueError, match="evaluation pool"):
        lg.add_train(Entry("scripted:Held", "scripted"))
    with pytest.raises(ValueError, match="training pool"):
        lg.add_eval(Entry("c3", "checkpoint"))
    assert [e["id"] for e in lg.evictions] == ["c0", "c1"]
    lg.save(tmp_path / "l.json")
    back = League.load(tmp_path / "l.json")
    assert back.to_dict() == lg.to_dict()
    rng = random.Random(0)
    assert {back.sample(rng).id for _ in range(300)} <= {"scripted:X", "c2", "c3"}


def test_league_adapt_samples_train_pool_only(school_ckpts, world_path, tmp_path):
    m = run(school_ckpts, world_path, tmp_path / "lg", league=True)
    lg = League.load(m["league"]["manifest"])
    train, ev = {e.id for e in lg.train_pool}, {e.id for e in lg.eval_pool}
    assert train and ev and not train & ev and "scripted:ScriptedAttacker" in train
    assert lg.provenance["eval_pool_used_for_selection"] is False and lg.provenance["selection_split"] == "validation"
    rep = json.loads(Path(m["league"]["eval_pool_report"]).read_text())
    assert {r["opponent"] for r in rep} == ev
    opp = A.LeagueOpponent(m["league"]["manifest"], "attacker", 0)
    for _ in range(40):
        opp.reset()
    assert set(opp.counts) <= train and len(opp.counts) > 1


def test_stale_and_duplicate_rollouts_rejected(school_ckpts, tmp_path):
    lg = tmp_path / "l.json"
    A.build_league(lg, "attacker", school_ckpts["attacker"], False)
    w = {"size": 12, "horizon": 20, "difficulty": 0.5, "base_logging": 1}
    venv = SyncVecEnv([lambda: A.adapt_env("defender", w, str(lg), "train")] * 2)
    ppo = build_ppo(load_checkpoint(school_ckpts["defender"]), PPOConfig(), restore=False)
    mk = lambda rid: collect_rollout(venv, ppo.policy, ppo.normalizer, 8, policy_version=ppo.policy_version,  # noqa: E731
                                     rollout_id=rid, seed=0, gamma=0.99)
    r = mk(ppo.last_rollout_id + 1)
    ppo.update(r)
    with pytest.raises(StaleRolloutError):
        ppo.update(r)                                      # already consumed / old policy version


def test_alternating_adaptation_phases(school_ckpts, world_path, tmp_path):
    out = tmp_path / "alt"
    kw = dict(first_role="attacker", attacker_checkpoint=school_ckpts["attacker"],
              defender_checkpoint=school_ckpts["defender"], world_config=world_path, steps_per_phase=64, seed=2,
              out=out, league=True, log=lambda s: None, **FAST)
    m = A.alternate(rounds=1, **kw)
    assert [p["tag"] for p in m["phases"]] == ["r0_attacker", "r0_defender"]
    att_snap, def_in = m["phases"][0]["snapshot"], m["phases"][1]["opponent_snapshot"]
    assert load_checkpoint(def_in).meta.checkpoint_id == load_checkpoint(att_snap).meta.checkpoint_id  # defender faces the adapted attacker
    assert load_checkpoint(m["phases"][0]["opponent_snapshot"]).meta.role == "defender"
    m2 = A.alternate(rounds=2, **kw)                      # resume: round 0 untouched, round 1 appended
    assert [p["tag"] for p in m2["phases"]][:2] == ["r0_attacker", "r0_defender"] and len(m2["phases"]) == 4
    assert m2["phases"][0] == m["phases"][0]
    for role in ("attacker", "defender"):
        assert load_checkpoint(out / role / "latest.pt").meta.training_steps == 128
