import pytest

from hospital_rl.experiments.config import RunRequest
from hospital_rl.experiments.controller import ConfigError, EvalRequest
from hospital_rl.experiments.fakes import FakeBackend
from hospital_rl.experiments.states import IllegalRunTransition, LEGAL, RunState

from .conftest import drive

REQ = RunRequest(preset="attacker-school-smoke", total_steps=512)


def test_create_is_async_and_durable(ctl):
    r = ctl.create(REQ, actor="alice")
    assert r["state"] == "queued" and r["owner"] == "alice" and ctl.backend.submits == []   # nothing submitted by the command
    assert r["resolved"]["caps"]["total_steps"] == 512 and r["attempts"][0]["job_id"] is None
    drive(ctl)
    r = ctl.get(r["id"])
    assert r["state"] == "running" and r["attempts"][0]["job_id"] == "fake:run-0001:a1"


def test_full_lifecycle_metrics_checkpoints_and_completion(ctl):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 12)
    r = ctl.get(rid)
    assert r["state"] == "completed" and r["artifact_problems"] == []
    m = ctl.metrics(rid)
    assert [x["env_steps"] for x in m] == [64 * i for i in range(1, 9)]
    cks = ctl.checkpoints(rid)
    assert cks and all(len(c["sha256"]) == 64 for c in cks) and cks[-1]["env_steps"] == 512
    assert ctl.artifacts(rid)[0]["partial"] is False


def test_checkpoint_and_stop_then_resume_lineage(ctl):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 4)                                       # running, 3 rounds done
    assert ctl.checkpoint_and_stop(rid)["state"] == "checkpointing"
    drive(ctl)
    r = ctl.get(rid)
    assert r["state"] == "stopped" and r["resumable"]
    latest = ctl.checkpoints(rid)[-1]
    ctl.resume(rid, latest["id"], actor="op")
    assert ctl.get(rid)["state"] == "queued"
    drive(ctl, 12)
    r = ctl.get(rid)
    a1, a2 = r["attempts"]
    assert r["state"] == "completed" and a2["resume_from"] == "a1" and a2["resume_checkpoint"] == latest["id"]
    steps = [x["env_steps"] for x in ctl.metrics(rid)]
    assert steps == sorted(set(steps)) and steps[-1] == 512          # resumed from the checkpoint; nothing counted twice
    assert ctl.checkpoints(rid)[-1]["parent_id"] is not None


def test_resume_only_from_latest_and_only_with_checkpoint(ctl):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl)
    ctl.cancel(rid, actor="op")
    with pytest.raises(ConfigError, match="no checkpoint"):
        ctl.resume(rid, actor="op")
    assert ctl.get(rid)["state"] == "cancelled"


def test_drain_finishes_round_then_stops(ctl):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 3)
    assert ctl.drain(rid, actor="op")["state"] == "draining"
    assert ctl.drain(rid, actor="op")["state"] == "draining"       # repeat is a no-op
    drive(ctl)
    assert ctl.get(rid)["state"] == "stopped" and ctl.checkpoints(rid)


def test_force_cancel_is_immediate_and_loses_work(ctl):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 2)
    r = ctl.cancel(rid, actor="op")
    assert r["state"] == "cancelled" and "lost" in r["error"]
    drive(ctl, 3)
    assert ctl.get(rid)["state"] == "cancelled"                     # nothing resurrects a terminal run
    with pytest.raises(IllegalRunTransition):
        ctl.drain(rid, actor="op")


def test_legal_transition_table_is_closed():
    assert LEGAL[RunState.completed] == set()
    assert RunState.running not in LEGAL[RunState.stopped] and RunState.queued in LEGAL[RunState.stopped]


def test_capacity_cap(ctl):
    ctl.create(REQ, actor="op")
    with pytest.raises(IllegalRunTransition, match="cap"):
        ctl.create(REQ, actor="op")


@pytest.mark.parametrize("bad,msg", [(dict(total_steps=10**9), "total_steps"), (dict(cpu=99), "cpu"), (dict(preset="nope"), "unknown preset"),
                                     (dict(backend="modal"), "configured backend")])
def test_validation_rejects(ctl, bad, msg):
    with pytest.raises(ConfigError, match=msg):
        ctl.validate(RunRequest(**{"preset": "attacker-school-smoke", **bad}))
    assert ctl.list() == []                                          # validation never creates anything


def test_estimate_is_unavailable_until_measured(ctl):
    assert ctl.validate(REQ).estimate.basis == "unavailable"
    ctl.create(REQ, actor="op")
    drive(ctl, 4)
    est = ctl.validate(REQ).estimate
    assert est.basis == "measured" and est.steps_per_sec == 64.0


def test_restart_recovery_resubmits_unsubmitted_and_never_invents_completion(make, tmp_path):
    c1 = make()
    rid = c1.create(REQ, actor="op")["id"]
    drive(c1, 3)                                                     # running, mid-flight
    c2 = make(backend=FakeBackend(tmp_path / "art"))                 # new process: backend forgot the job, DB remembers
    drive(c2)
    r = c2.get(rid)
    assert r["state"] == "failed" and "outcome unknown" in r["error"]
    assert c2.metrics(rid)                                           # what really ran is kept
    assert r["resumable"]                                            # and it can be resumed from its checkpoint


def test_restart_recovery_uses_verified_result_files(make, tmp_path):
    c1 = make()
    rid = c1.create(REQ, actor="op")["id"]
    drive(c1, 12)                                                    # job really finished
    c1.db.x("UPDATE exp_runs SET state='running' WHERE id=?", (rid,))   # simulate crash before the controller recorded it
    c2 = make(backend=FakeBackend(tmp_path / "art"))
    drive(c2)
    assert c2.get(rid)["state"] == "completed"


def test_crash_between_queued_and_submit_resubmits_once(make, tmp_path):
    c1 = make()
    rid = c1.create(REQ, actor="op")["id"]
    c1.db.x("UPDATE exp_runs SET state='starting' WHERE id=?", (rid,))
    c2 = make(backend=FakeBackend(tmp_path / "art"))
    drive(c2, 2)
    assert c2.get(rid)["state"] == "running" and c2.backend.submits == ["fake:run-0001:a1"]


def test_stale_heartbeat_and_backend_error(ctl, clock):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 2)
    assert ctl.get(rid)["stale"] is False
    clock.t += 31
    assert ctl.get(rid)["stale"] is True and ctl.get(rid)["state"] == "running"   # flagged, state not invented
    ctl.backend.status = lambda j: (_ for _ in ()).throw(RuntimeError("boom token=abc123"))
    drive(ctl)
    r = ctl.get(rid)
    assert r["state"] == "running" and "abc123" not in r["backend_error"] and r["stale"]


def test_submit_failure_marks_failed_with_redacted_error(ctl):
    ctl.backend.submit = lambda s: (_ for _ in ()).throw(RuntimeError("quota exceeded api_key=sk-live-zzz"))
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl)
    r = ctl.get(rid)
    assert r["state"] == "failed" and "quota" in r["error"] and "sk-live-zzz" not in r["error"]


def test_done_without_result_is_not_completed(ctl, tmp_path):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 2)
    orig = ctl.reader.result
    ctl.reader.result = lambda r, a: None
    for _ in range(12):
        ctl.tick()
    ctl.reader.result = orig
    assert ctl.get(rid)["state"] == "failed"


def test_partial_artifacts_reported(ctl, tmp_path):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 12)
    (tmp_path / "art" / rid / "a1" / "ckpt_000001.pt").write_bytes(b"tampered")
    a = ctl.artifacts(rid)[0]
    assert a["partial"] and any("changed" in p for p in a["problems"])


def _two_ckpts(ctl, clock):
    a = ctl.create(RunRequest(preset="attacker-school-smoke", total_steps=128), actor="op")["id"]
    drive(ctl, 6)
    d = ctl.create(RunRequest(preset="defender-school-smoke", total_steps=128), actor="op")["id"]
    drive(ctl, 6)
    return ctl.checkpoints(a)[-1]["id"], ctl.checkpoints(d)[-1]["id"]


def test_eval_frozen_results_and_debug_fork_isolation(ctl, clock):
    atk, dfn = _two_ckpts(ctl, clock)
    ev = ctl.submit_eval(EvalRequest(attacker_checkpoint_id=atk, defender_checkpoint_id=dfn, episodes=4), actor="op")
    assert ev["state"] == "queued" and ev["spec"]["frozen"]
    drive(ctl)
    done = ctl.get_eval(ev["id"])
    assert done["state"] == "done" and done["result"]["pairs"][0]["ci_low"] < done["result"]["pairs"][0]["ci_high"]
    fork = ctl.debug_fork("run-0001", atk, tick=3, actor="op")
    drive(ctl)
    assert ctl.get_eval(fork["id"])["label"] == "DEBUG"
    assert [e["id"] for e in ctl.evals()] == [ev["id"]]                       # benchmark query excludes DEBUG
    assert [e["id"] for e in ctl.evals(debug=True)] == [fork["id"]]


def test_eval_compatibility_errors(ctl, clock):
    atk, dfn = _two_ckpts(ctl, clock)
    with pytest.raises(ConfigError) as e:
        ctl.submit_eval(EvalRequest(attacker_checkpoint_id=dfn, defender_checkpoint_id=atk), actor="op")
    assert len(e.value.errors) == 2 and "expected attacker" in e.value.errors[0]
    with pytest.raises(ConfigError, match="episodes"):
        ctl.submit_eval(EvalRequest(attacker_checkpoint_id=atk, defender_checkpoint_id=dfn, episodes=999), actor="op")
    with pytest.raises(ConfigError, match="unknown checkpoint"):
        ctl.submit_eval(EvalRequest(attacker_checkpoint_id="x", defender_checkpoint_id=dfn), actor="op")


def test_eval_fails_if_checkpoint_changed_after_submission(ctl, clock, tmp_path):
    atk, dfn = _two_ckpts(ctl, clock)
    ev = ctl.submit_eval(EvalRequest(attacker_checkpoint_id=atk, defender_checkpoint_id=dfn), actor="op")
    c = ctl._ckpt(atk)
    (tmp_path / "art" / c["path"]).write_bytes(b"x")
    ctl.db.x("UPDATE exp_checkpoints SET sha256='deadbeef' WHERE id=?", (atk,))
    drive(ctl)
    e = ctl.get_eval(ev["id"])
    assert e["state"] == "failed" and "changed" in e["error"]


def test_cleanup_dry_run_ownership_and_protections(ctl, clock, tmp_path):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 4)
    ctl.checkpoint_and_stop(rid, actor="op")
    drive(ctl)
    (tmp_path / "art" / "stranger" / "a1").mkdir(parents=True)
    out = ctl.cleanup(actor="op")
    assert out["dry_run"] and out["plan"][0]["action"] == "skip" and "newest checkpoint" in out["plan"][0]["reason"]
    assert out["orphans_reported_not_deleted"] == [{"run_id": "stranger", "attempt_id": "a1"}]
    ctl.resume(rid, actor="op")
    drive(ctl, 12)                                              # completes; a1 no longer holds the newest checkpoint
    plan = {p["attempt_id"]: p for p in ctl.cleanup(actor="op")["plan"]}
    assert plan["a1"]["action"] == "delete" and plan["a2"]["action"] == "delete"
    ctl.cleanup(dry_run=False, actor="op")
    assert not (tmp_path / "art" / rid / "a1").exists() and (tmp_path / "art" / "stranger" / "a1").exists()
    assert ctl.get(rid)["attempts"][0]["cleaned_at"] and ctl.artifacts(rid)[0]["cleaned"]


def test_cleanup_refuses_unverified_manifest(ctl, tmp_path):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 12)
    (tmp_path / "art" / rid / "a1" / "stray.bin").write_bytes(b"?")
    p = ctl.cleanup(actor="op")["plan"][0]
    assert p["action"] == "skip" and "manifest" in p["reason"]


def test_audit_records_ok_and_refused(ctl):
    rid = ctl.create(REQ, actor="alice")["id"]
    with pytest.raises(IllegalRunTransition):
        ctl.drain(rid, actor="bob")
    log = ctl.audit_log()
    assert [(x["actor"], x["action"], x["outcome"]) for x in reversed(log)] == [("alice", "create_run", "ok"), ("bob", "drain", "refused")]


def test_streams_are_separated_by_source_kind(ctl, store_events=None):
    rid = ctl.create(REQ, actor="op")["id"]
    drive(ctl, 3)
    kinds = {e.source_kind for _, e in ctl.store.events_after(0)}
    assert kinds == {"experiment", "world"}                          # never "development"
    world = [e for _, e in ctl.store.events_after(0, source_kinds=("world",))]
    assert world and all(e.run_id == rid for e in world)
