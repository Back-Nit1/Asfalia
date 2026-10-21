"""School kind under the common contract + reconcile/cleanup protections (fakes only)."""
import json
import time

import pytest
from helpers import FakeGateway, ScriptedGateway, spec, wait_state

from hospital_rl.infra.backend import JobSpec, JobState
from hospital_rl.infra.bundle import resolve
from hospital_rl.infra.job import run_job
from hospital_rl.infra.lifecycle import GatewayBackend, Registry
from hospital_rl.infra.reconcile import reconcile


def school(attempt="s1", steps=96, **kw):
    tc = dict(role="defender", config="configs/defender_school.yaml", steps=steps, seed=7, eval_every=32, n_val=2,
              n_envs=2, rollout_len=16, ppo={"epochs": 1, "minibatches": 1})
    return JobSpec(run_id="sch", attempt_id=attempt, kind="school", train_config=tc, timeout_s=300, **kw)


def test_school_drain_resumes_in_new_attempt_with_cumulative_steps(tmp_path):
    calls = iter(range(100))
    r1 = run_job(resolve(school("s1")), tmp_path, control=lambda: "drain" if next(calls) >= 1 else None, log=lambda s: None)
    assert r1["state"] == "drained" and 0 < r1["env_steps"] < 96  # honored at the chunk boundary
    r2 = run_job(resolve(school("s2", resume_from_attempt="s1")), tmp_path, log=lambda s: None)
    assert r2["state"] == "done" and r2["env_steps"] >= 96
    rows = [json.loads(l) for l in (tmp_path / "sch" / "s2" / "metrics.jsonl").read_text().splitlines()]
    assert [r["env_steps"] for r in rows] == sorted(r["env_steps"] for r in rows) and len(rows) >= 3  # no step counted twice


def test_school_bundle_rejects_missing_config():
    bad = school()
    bad.train_config["config"] = "configs/nope.yaml"
    with pytest.raises(ValueError, match="existing config path"):
        from hospital_rl.infra.bundle import validate
        validate(resolve(bad))


# ------------------------------------------------------------------------------------------- reconcile
def finished_backend(tmp_path):
    gw = FakeGateway(tmp_path / "store")
    be = GatewayBackend("fake", gw, Registry(tmp_path / "reg"))
    for a, kw in (("a1", {}), ("a2", {"resume_from_attempt": "a1"})):
        wait_state(be, be.submit(spec(attempt=a, steps=64 if a == "a1" else 128, **kw)))
    return be, gw


def actions(rep):
    return {a["job_id"].split(":")[2]: (a["action"], a["reason"]) for a in rep["actions"]}


def test_dry_run_deletes_nothing_and_protects_newest_attempt(tmp_path):
    be, gw = finished_backend(tmp_path)
    rep = reconcile(be, ttl_s=0, now=time.time() + 10)
    assert rep["dry_run"] and actions(rep)["a1"][0] == "delete" and actions(rep)["a2"] == ("keep", "newest terminal attempt of its run (resume point)")
    assert (tmp_path / "store" / "r1" / "a1" / "manifest.json").exists() and rep["deleted"] == []


def test_apply_deletes_only_verified_old_owned_attempts_and_keeps_lineage_record(tmp_path):
    be, gw = finished_backend(tmp_path)
    rep = reconcile(be, dry_run=False, ttl_s=0, now=time.time() + 10)
    assert rep["deleted"] == ["fake:r1:a1"] and not (tmp_path / "store" / "r1" / "a1").exists()
    assert (tmp_path / "store" / "r1" / "a2" / "latest.pt").exists()
    assert be.reg.get("r1", "a1")["artifacts_deleted"] and be.reg.get("r1", "a1")["resume_from_attempt"] is None
    assert actions(reconcile(be, ttl_s=0, now=time.time() + 10))["a1"] == ("skip", "already cleaned")


def test_ttl_corruption_unowned_and_unresolved_are_never_deleted(tmp_path):
    be, gw = finished_backend(tmp_path)
    assert actions(reconcile(be, ttl_s=3600))["a1"][1].startswith("younger than ttl")
    (tmp_path / "store" / "r1" / "a1" / "latest.pt").write_bytes(b"corrupt")  # artifact verification gate
    assert "verification failed" in actions(reconcile(be, ttl_s=0, now=time.time() + 10))["a1"][1]
    be.reg.update("r1", "a1", labels={})  # not ours
    assert actions(reconcile(be, ttl_s=0, now=time.time() + 10))["a1"][0] == "skip"
    be.reg.update("r1", "a1", labels={"owner": "asfalia"}, state="running", finished_at=None)
    sg = ScriptedGateway(tmp_path / "store")
    live = GatewayBackend("fake", sg, be.reg)
    assert "unresolved" in actions(reconcile(live, ttl_s=0, now=time.time() + 10))["a1"][1]  # polled: still running


def test_orphans_are_reported_not_deleted_and_partials_need_opt_in(tmp_path):
    be, gw = finished_backend(tmp_path)
    (tmp_path / "store" / "stray" / "x1").mkdir(parents=True)
    rep = reconcile(be, dry_run=False, ttl_s=10**9)
    assert rep["orphans_not_deleted"] == ["stray/x1"] and (tmp_path / "store" / "stray" / "x1").exists()
    be.reg.update("r1", "a1", state="failed")
    assert "partial artifacts kept" in actions(reconcile(be, ttl_s=0, now=time.time() + 10))["a1"][1]
    assert actions(reconcile(be, ttl_s=0, now=time.time() + 10, include_partial=True))["a1"][0] == "delete"
