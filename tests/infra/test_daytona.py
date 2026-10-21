"""Daytona backend against a fake provider: lifecycle, retry, timeout, partial transfer, restart, orphans, cleanup.
Nothing here contacts Daytona or reads credentials."""
import json
import time

import pytest
from fake_daytona import FakeSandboxClient
from helpers import spec, wait_state

from hospital_rl.infra.backend import JobState
from hospital_rl.infra.daytona_backend import LABEL_ATTEMPT, LABEL_RUN, DaytonaBackend, DaytonaConfig, DaytonaGateway
from hospital_rl.infra.lifecycle import ResourceCaps
from hospital_rl.infra.reconcile import reconcile

OWNER = {"owner": "asfalia"}


def make(tmp_path, script=None, caps=None, now=time.time, client=None):
    cfg = DaytonaConfig(mirror_dir=str(tmp_path / "mirror"), registry_dir=str(tmp_path / "reg"), caps=caps or ResourceCaps(max_steps=1000, max_retries=1))
    cl = client or FakeSandboxClient(tmp_path / "boxes", script)
    return DaytonaBackend(cfg, DaytonaGateway(cfg, cl, now=now)), cl


def test_full_lifecycle_mirrors_verifies_then_deletes_sandbox(tmp_path):
    be, cl = make(tmp_path)
    jid = be.submit(spec(steps=64))
    assert cl.boxes["sb-1"]["labels"] == {**OWNER, LABEL_RUN: "r1", LABEL_ATTEMPT: "a1"}
    st = be.wait(jid, 120)
    assert st.state == JobState.DONE and st.result["env_steps"] >= 64
    assert cl.deleted == ["sb-1"]  # only after the mirror verified
    assert (tmp_path / "mirror/r1/a1/manifest.json").exists()
    assert not be.gw.mirror.joinpath("r1/a1/.result.json.tmp").exists()


def test_partial_transfer_keeps_sandbox_and_does_not_report_done(tmp_path):
    be, cl = make(tmp_path)
    cl.trim = {"result.json"}
    jid = be.submit(spec(steps=64))
    st = be.wait(jid, 120)
    assert st.state == JobState.FAILED and "sandbox kept" in st.detail and cl.deleted == []


def test_bounded_retry_then_success(tmp_path):
    be, cl = make(tmp_path, script=["fail", "run"])
    st = be.wait(be.submit(spec(steps=64, max_retries=1)), 120)
    assert st.state == JobState.DONE and len(cl.boxes["sb-1"]["cmds"]) == 2


def test_retries_exhausted_fails_and_cleans_sandbox(tmp_path):
    be, cl = make(tmp_path, script=["fail", "fail", "fail"])
    st = be.wait(be.submit(spec(steps=64, max_retries=1)), 60)
    assert st.state == JobState.FAILED and len(cl.boxes["sb-1"]["cmds"]) == 2 and cl.deleted == ["sb-1"]


def test_timeout_hard_kills_and_keeps_mirrored_state(tmp_path):
    clock = [1000.0]
    be, cl = make(tmp_path, script=["hang"], now=lambda: clock[0])
    jid = be.submit(spec(steps=64))
    assert be.status(jid).state == JobState.RUNNING
    clock[0] += 120 + 30 + 1800 + 5  # timeout_s + grace_s + setup budget
    assert be.status(jid).state == JobState.TIMED_OUT and cl.deleted == ["sb-1"]


def test_restart_recovery_reattaches_without_second_sandbox(tmp_path):
    be, cl = make(tmp_path, script=["hang"])
    jid = be.submit(spec(steps=64))
    be2, _ = make(tmp_path, client=cl)  # new controller, same registry + provider
    assert be2.submit(spec(steps=64)) == jid and cl.n == 1
    assert be2.status(jid).state == JobState.RUNNING
    cl.boxes["sb-1"]["cmds"][0]["code"] = 1  # job dies while controller is down
    assert be2.status(jid).state == JobState.FAILED


def test_force_cancel_mirrors_then_deletes(tmp_path):
    be, cl = make(tmp_path, script=["hang"])
    jid = be.submit(spec(steps=64))
    be.force_cancel(jid)
    assert be.status(jid).state == JobState.CANCELLED and cl.deleted == ["sb-1"]


def test_drain_control_word_reaches_the_sandbox(tmp_path):
    be, cl = make(tmp_path, script=["hang"])
    jid = be.submit(spec(steps=64))
    be.drain(jid)
    assert (cl.base / "sb-1/runs/r1/control/a1").read_text() == "drain"


def test_setup_failure_leaves_no_sandbox(tmp_path):
    be, cl = make(tmp_path)
    cl.run = lambda *a: 3
    with pytest.raises(RuntimeError, match="setup failed"):
        be.submit(spec(steps=64))
    assert cl.deleted == ["sb-1"]


def test_caps_enforced_before_any_sandbox(tmp_path):
    be, cl = make(tmp_path, caps=ResourceCaps(max_steps=10))
    with pytest.raises(ValueError, match="exceeds configured resource caps"):
        be.submit(spec(steps=64))
    assert cl.n == 0


def test_orphans_foreign_and_cleanup_only_touch_owned(tmp_path):
    be, cl = make(tmp_path)
    jid = be.submit(spec(steps=64))
    be.wait(jid, 120)
    cl.adopt({**OWNER, LABEL_RUN: "ghost", LABEL_ATTEMPT: "a1"})  # ours by label, unknown to the registry
    foreign = cl.adopt({"team": "other"})  # someone else's sandbox
    rep = reconcile(be, dry_run=False, ttl_s=0)
    assert "ghost/a1" in rep["orphans_not_deleted"] and foreign not in cl.deleted and "sb-3" not in cl.deleted
    assert "sb-2" not in cl.deleted  # orphan reported, never auto-deleted
    assert [a["action"] for a in rep["actions"]] == ["keep"]  # newest terminal attempt is the resume point


def test_resume_lineage_uploads_previous_mirror(tmp_path):
    be, cl = make(tmp_path)
    be.wait(be.submit(spec(steps=64)), 120)
    j2 = be.submit(spec(attempt="a2", steps=128, resume_from_attempt="a1"))
    st = be.wait(j2, 120)
    assert st.state == JobState.DONE and json.loads(be.gw.read_file("r1", "a2", "result.json"))["env_steps"] >= 128
