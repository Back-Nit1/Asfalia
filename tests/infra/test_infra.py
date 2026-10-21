import time
from pathlib import Path

import pytest
from helpers import FakeGateway, ids, spec, wait_state

from hospital_rl.infra.artifacts import DirArtifactStore
from hospital_rl.infra.backend import JobState
from hospital_rl.infra.local_backend import LocalBackend
from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, load_modal_config
from hospital_rl.infra.plan import cloud_plan
from hospital_rl.policies.checkpoint import load_for_role

CONFIGS = Path(__file__).parents[2] / "configs"


@pytest.fixture
def local(tmp_path):
    return LocalBackend(tmp_path / "store")


def test_local_job_completes_with_verified_manifest(local, tmp_path):
    jid = local.submit(spec())
    assert local.submit(spec()) == jid  # idempotent
    st = local.wait(jid)
    assert st.state == JobState.DONE and st.result["env_steps"] == 256
    store = DirArtifactStore(tmp_path / "store")
    assert store.verify_manifest("r1", "a1") == []
    (tmp_path / "store" / "r1" / "a1" / "latest.pt").write_bytes(b"corrupt")
    assert any("changed latest.pt" in p for p in store.verify_manifest("r1", "a1"))


def test_drain_stop_cancel_are_distinct(local, tmp_path):
    big = 10**9
    states = {}
    for name in ("drain", "stop", "cancel"):
        jid = local.submit(spec(attempt=name, steps=big))
        time.sleep(6)  # let it import torch and run a few rounds
        {"drain": local.drain, "stop": local.checkpoint_and_stop, "cancel": local.force_cancel}[name](jid)
        states[name] = local.wait(jid, 60)
    assert states["drain"].state == JobState.DRAINED
    assert states["stop"].state == JobState.STOPPED
    assert states["cancel"].state == JobState.CANCELLED
    for name in ("drain", "stop"):  # graceful paths publish a loadable final checkpoint + manifest
        load_for_role(tmp_path / "store" / "r1" / name / "latest.pt", *ids())
        assert DirArtifactStore(tmp_path / "store").verify_manifest("r1", name) == []
    assert not (tmp_path / "store" / "r1" / "cancel" / "manifest.json").exists()  # force-cancel publishes nothing
    for f in (tmp_path / "store" / "r1" / "cancel").glob("*.pt"):  # whatever was published is whole
        load_for_role(f, *ids())


def test_timeout_checkpoints_then_reports_timed_out(tmp_path):
    be = LocalBackend(tmp_path / "store")
    s = spec(steps=10**9).model_copy(update={"timeout_s": 6, "grace_s": 20})
    st = be.wait(be.submit(s), 60)
    assert st.state == JobState.TIMED_OUT
    load_for_role(tmp_path / "store" / "r1" / "a1" / "latest.pt", *ids())


def test_new_attempt_resumes_original_lineage_without_double_counting(local, tmp_path):
    local.wait(local.submit(spec(attempt="a1", steps=64)))
    st = local.wait(local.submit(spec(attempt="a2", steps=128, resume_from_attempt="a1")))
    assert st.state == JobState.DONE and st.result["env_steps"] == 128  # 64 inherited + 64 new, not 64 + 128
    m1 = load_for_role(tmp_path / "store" / "r1" / "a1" / "latest.pt", *ids()).meta
    m2 = load_for_role(tmp_path / "store" / "r1" / "a2" / "latest.pt", *ids()).meta
    assert m2.parent_checkpoint_id is not None and m2.training_steps == 128 and m1.training_steps == 64


def test_artifact_store_immutable_and_hashed(tmp_path):
    s = DirArtifactStore(tmp_path)
    ref = s.put("r", "a", "x.bin", b"abc")
    assert s.get(ref) == b"abc"
    with pytest.raises(FileExistsError):
        s.put("r", "a", "x.bin", b"other")
    (tmp_path / ref.path).write_bytes(b"zzz")
    with pytest.raises(ValueError, match="hash mismatch"):
        s.get(ref)
    s.publish_manifest("r", "a")
    with pytest.raises(FileExistsError):
        s.publish_manifest("r", "a")


def test_modal_backend_with_fake_gateway(tmp_path):
    cfg = ModalConfig(caps={"timeout_s": 120, "max_steps": 512, "cpu": 1, "memory_mb": 2048})
    gw = FakeGateway(tmp_path / "store")
    be = ModalBackend(cfg, gw, tmp_path / "reg")
    with pytest.raises(ValueError, match="total_steps=1000 > cap 512"):
        be.submit(spec(steps=1000))
    assert gw.spawned == 0  # rejected before anything launched
    jid = be.submit(spec(steps=64))
    assert be.submit(spec(steps=64)) == jid and gw.spawned == 1
    st = wait_state(be, jid)
    assert st.state == JobState.DONE and st.result["env_steps"] == 64
    # drain / stop travel through the control file; cancel is its own state
    for name, op, want in (("d", be.drain, JobState.DRAINED), ("s", be.checkpoint_and_stop, JobState.STOPPED)):
        j = be.submit(spec(attempt=name, steps=512))
        op(j)
        assert wait_state(be, j).state == want
    j = be.submit(spec(attempt="c", steps=512))
    be.force_cancel(j)
    assert be.status(j).state == JobState.CANCELLED


def test_modal_failure_surfaces_as_failed(tmp_path):
    class Boom(FakeGateway):
        def poll(self, call_id):
            return "failed", "quota exceeded"

    be = ModalBackend(ModalConfig(caps={"max_steps": 512}), Boom(tmp_path / "store"), tmp_path / "reg")
    st = be.status(be.submit(spec(steps=64)))
    assert st.state == JobState.FAILED and "quota" in st.detail


def test_real_modal_sdk_surface_matches():
    """Guards the verified SDK surface used by RealModalGateway (no network)."""
    modal = pytest.importorskip("modal")
    import inspect
    assert {"volumes", "cpu", "memory", "timeout", "retries", "max_containers", "secrets", "name"} <= set(
        inspect.signature(modal.App.function).parameters)
    assert "name" in inspect.signature(modal.App.deploy).parameters
    assert {"commit", "reload", "batch_upload", "read_file", "listdir", "remove_file"} <= set(dir(modal.Volume))
    assert {"cancel", "from_id", "get"} <= set(dir(modal.FunctionCall))
    assert {"from_name"} <= set(dir(modal.Function)) and {"add_local_dir", "add_local_python_source", "env"} <= set(dir(modal.Image))
    assert {"FunctionTimeoutError", "OutputExpiredError", "NotFoundError"} <= set(dir(modal.exception))
    cfg = load_modal_config(CONFIGS / "cloud_modal.yaml")
    assert cfg.caps.max_steps == 20000 and cfg.caps.timeout_s == 300


def test_cloud_plan_needs_no_credentials(tmp_path):
    plan = cloud_plan("modal", CONFIGS / "cloud_modal.yaml", env={}, home=tmp_path)
    assert plan.effective_backend == "local" and not plan.credentials["present"]
    assert any("modal setup" in n for n in plan.notes)
    assert "ESTIMATE" in plan.estimate["label"] and plan.estimate["cpu_core_hours_upper_bound"] == round(2 * 300 / 3600, 4)
    assert plan.hard_caps["max_steps"] == 20000
    # secret *names* only, never values
    ok = cloud_plan("modal", CONFIGS / "cloud_modal.yaml", env={"MODAL_TOKEN_ID": "SECRETVALUE", "MODAL_TOKEN_SECRET": "x"}, home=tmp_path)
    assert ok.effective_backend == "modal" and "SECRETVALUE" not in ok.model_dump_json()
    assert cloud_plan("local", CONFIGS / "cloud_local.yaml", env={}, home=tmp_path).effective_backend == "local"
    d = cloud_plan("daytona", CONFIGS / "cloud_daytona.yaml", env={"DAYTONA_API_KEY": "SECRETVALUE"}, home=tmp_path)
    assert d.effective_backend == "daytona" and "SECRETVALUE" not in d.model_dump_json()
    n = cloud_plan("daytona", CONFIGS / "cloud_daytona.yaml", env={}, home=tmp_path)
    assert n.effective_backend == "local" and any("DAYTONA_API_KEY" in x for x in n.notes)
    with pytest.raises(ValueError):
        cloud_plan("local", CONFIGS / "cloud_modal.yaml")
