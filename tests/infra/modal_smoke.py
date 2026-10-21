"""Bounded LIVE Modal smoke (spends real, small compute on your Modal account). Skipped unless explicitly enabled:

  ASFALIA_LIVE_MODAL=1 PYTHONPATH=src python tests/infra/modal_smoke.py

Needs Modal credentials (`modal setup` or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET; values are never read here). Hard caps:
1 cpu, 1 GiB, 120 s (+30 s grace) per job, 1 container, 0 retries, 2048 steps per job, at most 3 jobs. Real hospital env,
no test modules. Exercises: clean-container import + run -> manifest verification, checkpoint-and-stop, resume in a new
attempt. Cleans up its own Volume run directories. Prints every call id and state so results can be recorded exactly.
"""
import os
import sys
import tempfile
import time
import uuid

if os.environ.get("ASFALIA_LIVE_MODAL") != "1":
    sys.exit("skipped: set ASFALIA_LIVE_MODAL=1 to run the live Modal smoke (nothing launched)")

from hospital_rl.infra.backend import JobSpec, JobState  # noqa: E402
from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, RealModalGateway  # noqa: E402
from hospital_rl.infra.plan import credentials_present  # noqa: E402
from hospital_rl.infra.reconcile import verify_remote  # noqa: E402

if not credentials_present("modal"):
    sys.exit("no Modal credentials: run `modal setup`. Nothing launched.")

cfg = ModalConfig(app_name="hospital-rl-smoke", volume_name="hospital-rl-smoke-runs", registry_dir=tempfile.mkdtemp(prefix="asfalia-reg-"),
                  caps=dict(cpu=1, memory_mb=1024, timeout_s=120, grace_s=30, max_containers=1, max_retries=0, max_steps=2048))
gw = RealModalGateway(cfg)
be = ModalBackend(cfg, gw)
run = f"smoke-{uuid.uuid4().hex[:8]}"
TC = dict(role="defender", total_steps=2048, n_envs=4, rollout_len=64, hidden=32, checkpoint_every=2,
          env_factory="hospital_rl.envs.factories:hospital_single",
          env_kwargs=dict(role="defender", size=8, split="train", difficulty=0.5, horizon=20, opponent="scripted"))


def job(attempt, **kw):
    return JobSpec(run_id=run, attempt_id=attempt, timeout_s=120, grace_s=30, cpu=1, memory_mb=1024, train_config=TC, **kw)


def wait(jid, limit=300):
    t0 = time.time()
    while not (st := be.status(jid)).state.terminal and time.time() - t0 < limit:
        time.sleep(3)
    return st


try:
    t0 = time.time()
    a = be.submit(job("a1"))
    print("submitted", a, "call", be._rec(a)["remote_id"])
    st = wait(a)
    print("a1:", st.state.value, st.detail, st.result, f"{time.time() - t0:.0f}s", "checkpoint", st.checkpoint)
    assert st.state == JobState.DONE and verify_remote(gw, run, "a1") == [], "a1 must finish with a verified manifest"
    b = be.submit(job("b1"))
    while not be.status(b).checkpoint and time.time() - t0 < 200:
        time.sleep(2)
    be.checkpoint_and_stop(b)
    st = wait(b)
    print("b1:", st.state.value, st.result, "checkpoint", st.checkpoint)
    assert st.state in (JobState.STOPPED, JobState.DONE) and verify_remote(gw, run, "b1") == []
    c = be.submit(job("b2", resume_from_attempt="b1"))
    st = wait(c)
    print("b2:", st.state.value, st.result)
    assert st.state == JobState.DONE and st.result["env_steps"] == 2048, "resume must not double count"
    print("SMOKE OK; run", run)
finally:
    gw.close()  # stops the ephemeral app + containers
    for att in ("a1", "b1", "b2"):
        try:
            gw.delete_attempt(run, att)
        except Exception as e:  # noqa: BLE001
            print("cleanup note:", att, type(e).__name__)
