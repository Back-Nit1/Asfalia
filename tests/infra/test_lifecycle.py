"""Every lifecycle / recovery / error branch of the common backend, against fake gateways (no network)."""
import json
import os
import signal
import time

import pytest
from helpers import FakeGateway, ScriptedGateway, ids, spec, wait_state

from hospital_rl.infra.backend import JobState
from hospital_rl.infra.job import PROGRESS
from hospital_rl.infra.lifecycle import GatewayBackend, Registry, ResourceCaps
from hospital_rl.infra.local_backend import LocalBackend


def backend(tmp_path, gw=None, caps=None):
    gw = gw or ScriptedGateway(tmp_path / "store")
    return GatewayBackend("fake", gw, Registry(tmp_path / "reg"), caps), gw


def test_invalid_bundle_rejected_before_anything_launches(tmp_path):
    be, gw = backend(tmp_path)
    bad = spec()
    bad.train_config["env_factory"] = "toy_env:make_env"  # not a package module
    with pytest.raises(ValueError, match="must live in the hospital_rl package"):
        be.submit(bad)
    with pytest.raises(ValueError, match="not a safe path component"):
        be.submit(spec(attempt="../evil"))
    assert gw.spawned == 0 and be.reg.all() == []


def test_caps_checked_before_launch(tmp_path):
    be, gw = backend(tmp_path, caps=ResourceCaps(max_steps=128, timeout_s=60))
    with pytest.raises(ValueError, match="total_steps=256 > cap 128"):
        be.submit(spec())
    with pytest.raises(ValueError, match="timeout_s=120 > cap 60"):
        be.submit(spec(steps=64))
    assert gw.spawned == 0


def test_submit_is_idempotent_across_controller_restart(tmp_path):
    be, gw = backend(tmp_path)
    jid = be.submit(spec())
    be2 = GatewayBackend("fake", gw, Registry(tmp_path / "reg"))  # new process, same durable registry
    assert be2.submit(spec()) == jid and gw.spawned == 1
    assert be2.status(jid).state == JobState.RUNNING


def test_crash_between_intent_and_spawn_never_double_launches(tmp_path):
    be, gw = backend(tmp_path)
    jid = be.submit(spec())
    be.reg.update("r1", "a1", remote_id=None)  # as if the controller died before recording the remote id
    st = backend(tmp_path, gw)[0].status(jid)
    assert st.state == JobState.FAILED and "burned" in st.detail and gw.spawned == 1


@pytest.mark.parametrize("outcome,state", [(("timed_out", "function timeout"), JobState.TIMED_OUT),
                                           (("failed", "quota exceeded"), JobState.FAILED),
                                           (("lost", "owner gone"), JobState.FAILED),
                                           (("done", {"state": "drained", "env_steps": 5}), JobState.DRAINED)])
def test_poll_outcomes_map_to_states_and_persist(tmp_path, outcome, state):
    be, gw = backend(tmp_path, ScriptedGateway(tmp_path / "store", outcome))
    jid = be.submit(spec())
    st = be.status(jid)
    assert st.state == state
    gw.outcome = ("running", {})  # a terminal state is final: later polls cannot resurrect the attempt
    assert be.status(jid).state == state and be.reg.get("r1", "a1")["finished_at"]


def test_failure_reports_partial_artifacts_and_last_checkpoint(tmp_path):
    be, gw = backend(tmp_path, ScriptedGateway(tmp_path / "store", ("failed", "container preempted")))
    jid = be.submit(spec())
    d = tmp_path / "store" / "r1" / "a1"
    d.mkdir(parents=True)
    (d / PROGRESS).write_text(json.dumps({"checkpoint": "latest.pt", "env_steps": 64, "written_at": time.time() - 30}))
    st = be.status(jid)
    assert st.state == JobState.FAILED and "preempted" in st.detail
    assert "no manifest" in st.detail and "resume_from_attempt='a1'" in st.detail
    assert st.checkpoint["env_steps"] == 64 and st.checkpoint["age_s"] >= 30


def test_controls_reach_gateway_only_while_live_and_cancel_is_distinct(tmp_path):
    be, gw = backend(tmp_path)
    jid = be.submit(spec())
    be.drain(jid), be.checkpoint_and_stop(jid)
    assert ("control", "drain") in gw.log and ("control", "stop") in gw.log
    be.force_cancel(jid)
    assert be.status(jid).state == JobState.CANCELLED and ("cancel", "sc-1") in gw.log
    n = len(gw.log)
    be.drain(jid), be.force_cancel(jid)  # terminal: no further remote effects
    assert len(gw.log) == n


def test_cannot_resume_from_a_live_attempt(tmp_path):
    be, _ = backend(tmp_path)
    be.submit(spec(attempt="a1"))
    with pytest.raises(ValueError, match="still pending|still running"):
        be.submit(spec(attempt="a2", resume_from_attempt="a1"))


def test_resume_lineage_is_recorded_and_steps_not_double_counted(tmp_path):
    gw = FakeGateway(tmp_path / "store")
    be, _ = backend(tmp_path, gw)
    wait_state(be, be.submit(spec(attempt="a1", steps=64)))
    st = wait_state(be, be.submit(spec(attempt="a2", steps=128, resume_from_attempt="a1")))
    assert st.state == JobState.DONE and st.result["env_steps"] == 128
    assert be.reg.get("r1", "a2")["resume_from_attempt"] == "a1"
    assert st.checkpoint["env_steps"] == 128


def test_resume_refuses_a_tampered_source(tmp_path):
    gw = FakeGateway(tmp_path / "store")
    be, _ = backend(tmp_path, gw)
    wait_state(be, be.submit(spec(attempt="a1", steps=64)))
    (tmp_path / "store" / "r1" / "a1" / "latest.pt").write_bytes(b"corrupt")
    st = wait_state(be, be.submit(spec(attempt="a2", steps=128, resume_from_attempt="a1")))
    assert st.state == JobState.FAILED and "manifest verification" in st.detail


def test_retry_after_completion_is_a_no_op_not_a_second_manifest(tmp_path):
    from hospital_rl.infra.job import run_job
    be, gw = backend(tmp_path, FakeGateway(tmp_path / "store"))
    sp = spec(steps=64)
    wait_state(be, be.submit(sp))
    from hospital_rl.infra.bundle import resolve
    again = run_job(resolve(sp), tmp_path / "store", log=lambda s: None)  # e.g. Modal retry after the result was lost
    assert again["state"] == "done" and again["env_steps"] == 64


def test_local_kill_then_same_attempt_retry_resumes_without_double_counting(tmp_path):
    be = LocalBackend(tmp_path / "store")
    jid = be.submit(spec(steps=4096, max_retries=1))
    latest = tmp_path / "store" / "r1" / "a1" / "latest.pt"
    end = time.time() + 90
    while not latest.exists() and time.time() < end:
        time.sleep(0.2)
    assert latest.exists()
    os.kill(int(be._rec(jid)["remote_id"].split("|")[0]), signal.SIGKILL)  # preemption-style death, not a cancel
    st = be.wait(jid, 120)
    assert st.state == JobState.DONE and st.result["env_steps"] == 4096  # exactly the cap, not cap + redone steps


def test_local_attach_after_controller_restart(tmp_path):
    be = LocalBackend(tmp_path / "store")
    jid = be.submit(spec(steps=64))
    be.wait(jid)
    be2 = LocalBackend(tmp_path / "store")  # fresh process view: same registry beside the store
    assert be2.status(jid).state == JobState.DONE and be2.submit(spec(steps=64)) == jid
    # a live job from a previous controller is re-attachable by pid; a dead one without result.json is FAILED
    be2.reg.update("r1", "a1", state="running", finished_at=None, result={}, detail="")
    (tmp_path / "store" / "r1" / "a1" / "result.json").unlink()
    st = be2.status(jid)
    assert st.state == JobState.FAILED and "exit status unknown" in st.detail


def test_job_soft_deadline_checkpoints_and_reports_timed_out(tmp_path):
    from hospital_rl.infra.bundle import resolve
    from hospital_rl.infra.job import run_job
    sp = resolve(spec(steps=10**9).model_copy(update={"timeout_s": 3, "grace_s": 5}))
    r = run_job(sp, tmp_path, log=lambda s: None)
    assert r["state"] == "timed_out" and (tmp_path / "r1" / "a1" / "latest.pt").exists()
    assert json.loads((tmp_path / "r1" / "a1" / PROGRESS).read_text())["env_steps"] == r["env_steps"]
