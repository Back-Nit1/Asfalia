import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

from hospital_rl.policies.checkpoint import (IncompatibleCheckpointError, load_checkpoint, load_for_role,
                                             load_frozen_agent)
from hospital_rl.training.train import TrainConfig, evaluate, train
from toy_env import make_env

TESTS = str(Path(__file__).parent)


def cfg(**kw):
    base = dict(role="defender", seed=3, total_steps=512, n_envs=4, rollout_len=32, hidden=16, checkpoint_every=1,
                env_factory="toy_env:make_env", ppo={"epochs": 2, "minibatches": 2})
    return TrainConfig(**{**base, **kw})


def ids():
    m = make_env().metadata
    return m["role"], m["catalog_hash"], m["observation_schema_hash"]


def test_resume_matches_uninterrupted_bitwise(tmp_path):
    train(cfg(), tmp_path / "full", log=lambda s: None)
    train(cfg(total_steps=256), tmp_path / "part", log=lambda s: None)  # 2 rounds
    res = train(cfg(), tmp_path / "part", resume=True, log=lambda s: None)
    a = load_checkpoint(tmp_path / "full" / "ckpt_000004.pt")
    b = load_checkpoint(tmp_path / "part" / "ckpt_000004.pt")
    assert res.env_steps == 512 and a.meta.training_steps == b.meta.training_steps
    for k in a.state["model"]:
        assert torch.equal(a.state["model"][k], b.state["model"][k]), k
    for k in a.state["normalizer"]["mean"]:
        assert torch.equal(a.state["normalizer"]["mean"][k], b.state["normalizer"]["mean"][k])
    assert torch.equal(a.state["rng"]["torch"], b.state["rng"]["torch"])
    # lineage survives resume
    assert b.meta.parent_checkpoint_id == load_checkpoint(tmp_path / "part" / "ckpt_000003.pt").meta.checkpoint_id


def test_incompatible_checkpoint_names_field(tmp_path):
    train(cfg(total_steps=64), tmp_path, log=lambda s: None)
    role, cat, obs = ids()
    p = tmp_path / "latest.pt"
    load_for_role(p, role, cat, obs)
    with pytest.raises(IncompatibleCheckpointError, match="catalog_hash") as e:
        load_for_role(p, role, "deadbeef", obs)
    assert "observation_schema_hash" not in str(e.value)
    with pytest.raises(IncompatibleCheckpointError, match="role: checkpoint has 'defender', expected 'attacker'"):
        load_for_role(p, "attacker", cat, obs)
    with pytest.raises(IncompatibleCheckpointError, match="observation_schema_hash"):
        load_for_role(p, role, cat, "x")


def test_metadata_fields(tmp_path):
    train(cfg(total_steps=64), tmp_path, log=lambda s: None)
    m = load_checkpoint(tmp_path / "latest.pt").meta
    assert m.max_nodes == 8 and m.seed == 3 and m.training_steps == 128 and m.normalizer["frozen"] is False
    assert {"torch", "numpy", "gymnasium"} <= set(m.library_versions) and m.config_hash
    assert (tmp_path / "latest.pt.meta.json").exists()


def test_frozen_agent_loader_freezes_normalizer(tmp_path):
    train(cfg(total_steps=64), tmp_path, log=lambda s: None)
    fa = load_frozen_agent(tmp_path / "latest.pt", *ids())
    before = fa.agent.normalizer.mean["node_features"].copy()
    evaluate(fa.agent, make_env, episodes=2)
    assert fa.agent.normalizer.frozen and (before == fa.agent.normalizer.mean["node_features"]).all()


def test_control_stop_vs_drain(tmp_path):
    n = {"c": 0}

    def stop_after_collect():  # polled once at loop top, once after collect
        n["c"] += 1
        return "stop" if n["c"] >= 2 else None

    r = train(cfg(), tmp_path / "stop", control=stop_after_collect, log=lambda s: None)
    assert r.stop_reason == "stop" and r.env_steps == 0  # in-flight rollout discarded, nothing counted
    r = train(cfg(), tmp_path / "drain", control=lambda: None, log=lambda s: None)
    assert r.stop_reason == "done"
    r = train(cfg(), tmp_path / "drain2", control=lambda: "drain", log=lambda s: None)
    assert r.stop_reason == "drain" and (tmp_path / "drain2" / "latest.pt").exists()


def test_atomic_write_failure_keeps_previous(tmp_path, monkeypatch):
    train(cfg(total_steps=64), tmp_path, log=lambda s: None)
    before = (tmp_path / "latest.pt").read_bytes()
    import hospital_rl.policies.checkpoint as ck
    monkeypatch.setattr(ck.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("killed")))
    with pytest.raises(OSError):
        train(cfg(total_steps=512), tmp_path, resume=True, log=lambda s: None)
    assert (tmp_path / "latest.pt").read_bytes() == before
    load_for_role(tmp_path / "latest.pt", *ids())


def test_sigkill_mid_write_leaves_loadable_checkpoint(tmp_path):
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {TESTS!r})
        from hospital_rl.training.train import TrainConfig, train
        cfg = TrainConfig(role="defender", total_steps=10**9, n_envs=2, rollout_len=8, hidden=256,
                          checkpoint_every=1, env_factory="toy_env:make_env")
        train(cfg, {str(tmp_path)!r}, log=lambda s: None)
    """)
    p = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.time() + 60
        while time.time() < deadline and len(list(tmp_path.glob("ckpt_*.pt"))) < 5:
            time.sleep(0.05)
        assert len(list(tmp_path.glob("ckpt_*.pt"))) >= 5
        p.kill()
        p.wait()
    finally:
        p.kill()
    for f in tmp_path.glob("*.pt"):  # every published file is complete
        load_checkpoint(f)
    load_for_role(tmp_path / "latest.pt", *ids())


def test_sigterm_triggers_final_checkpoint(tmp_path):
    script = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {TESTS!r})
        from hospital_rl.training.train import TrainConfig, train
        cfg = TrainConfig(role="defender", total_steps=10**9, n_envs=2, rollout_len=8, hidden=16,
                          checkpoint_every=10**6, env_factory="toy_env:make_env")
        print("ready", flush=True)
        r = train(cfg, {str(tmp_path)!r}, log=lambda s: None)
        print("reason", r.stop_reason, flush=True)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    time.sleep(3)
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert "reason stop" in out and p.returncode == 0
    assert load_for_role(tmp_path / "latest.pt", *ids()).meta.training_steps > 0


def test_toy_learning_signal(tmp_path):
    """One-seed smoke: trained policy beats its own untrained init on the toy task."""
    t0 = time.time()
    c = cfg(seed=0, total_steps=30_000, n_envs=8, rollout_len=64, hidden=32, checkpoint_every=10**6,
            ppo={"lr": 1e-3, "ent_coef": 0.01})
    res = train(c, tmp_path, log=lambda s: None)
    trained = load_frozen_agent(tmp_path / "latest.pt", *ids(), deterministic=True)
    score = evaluate(trained.agent, make_env, 20)
    first = res.history[0]["mean_return"]
    last = res.history[-1]["mean_return"]
    print(f"\nSMOKE one-seed: first-round train return {first:.2f} -> last-round {last:.2f}; "
          f"greedy eval {score:.2f} (max 16); {time.time() - t0:.0f}s")
    assert last > first + 3 and score > 6
    assert time.time() - t0 < 60
