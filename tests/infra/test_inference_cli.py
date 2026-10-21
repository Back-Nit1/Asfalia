"""export-policy / benchmark-inference CLIs, metadata gate, golden hospital corpus.

Checkpoints: set HOSPITAL_CKPT_ATTACKER / HOSPITAL_CKPT_DEFENDER to real hospital checkpoints to run against them;
otherwise a tiny hospital checkpoint (12-node worlds, 64 env steps) is trained per role so the tests always run.
"""
import json
import os

import numpy as np
import pytest

from hospital_rl.infra import inference_bench as bench
from hospital_rl.infra import inference_cli
from hospital_rl.infra.inference import (EXPORT_FORMAT, InferenceOutput, OnnxRuntimeBackend, PythonReferenceBackend,
                                         check_meta, export_onnx)
from hospital_rl.policies.checkpoint import IncompatibleCheckpointError
from hospital_rl.training.train import TrainConfig, train

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

ROLES = ["attacker", "defender"]


@pytest.fixture(scope="module", params=ROLES)
def ckpt(request, tmp_path_factory):
    role = request.param
    if os.environ.get(f"HOSPITAL_CKPT_{role.upper()}"):
        return role, os.environ[f"HOSPITAL_CKPT_{role.upper()}"]
    d = tmp_path_factory.mktemp(role)
    train(TrainConfig(role=role, total_steps=64, n_envs=2, rollout_len=16, hidden=16, checkpoint_every=10 ** 6,
                      env_factory="hospital_rl.envs.factories:hospital_single",
                      env_kwargs={"role": role, "size": 12, "split": "train"}), d, log=lambda s: None)
    return role, str(d / "latest.pt")


def run_cli(*argv):
    return inference_cli.main([str(a) for a in argv])


def test_export_cli_roundtrip(ckpt, tmp_path, capsys):
    role, path = ckpt
    out = tmp_path / "m.onnx"
    assert run_cli("--checkpoint", path, "--out", out, "--expect-role", role) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] and res["role"] == role and out.is_file() and not list(tmp_path.glob("*.tmp"))
    on_disk = json.loads((tmp_path / "m.onnx.meta.json").read_text())
    be = OnnxRuntimeBackend(out, on_disk["catalog_hash"], on_disk["observation_schema_hash"], expected_role=role)
    assert be.meta == on_disk and on_disk["export_format"] == EXPORT_FORMAT
    assert run_cli("--checkpoint", path, "--out", out) == 2  # refuses to overwrite
    assert run_cli("--checkpoint", path, "--out", out, "--overwrite") == 0


def test_export_cli_refusals_leave_nothing(ckpt, tmp_path, capsys):
    role, path = ckpt
    other = "defender" if role == "attacker" else "attacker"
    for argv in (["--checkpoint", path, "--out", tmp_path / "a.onnx", "--expect-role", other],
                 ["--checkpoint", path, "--out", tmp_path / "a.onnx", "--expect-catalog-hash", "nope"],
                 ["--checkpoint", tmp_path / "missing.pt", "--out", tmp_path / "a.onnx"]):
        assert run_cli(*argv) == 2
        assert json.loads(capsys.readouterr().out.splitlines()[-1])["ok"] is False
    assert list(tmp_path.iterdir()) == []


def test_check_meta_rejects_missing_and_incompatible(ckpt, tmp_path):
    role, path = ckpt
    meta = export_onnx(path, tmp_path / "m.onnx")
    assert check_meta(meta, "x", role) is meta
    with pytest.raises(IncompatibleCheckpointError, match="missing"):
        check_meta({k: v for k, v in meta.items() if k != "catalog_hash"}, "x")
    with pytest.raises(IncompatibleCheckpointError, match="export_format"):
        check_meta({**meta, "export_format": 99}, "x")
    with pytest.raises(IncompatibleCheckpointError, match="observation_schema_hash"):
        check_meta(meta, "x", expect_obs_hash="nope")
    import onnx

    m = onnx.load(str(tmp_path / "m.onnx"))
    del m.metadata_props[:]
    onnx.save(m, str(tmp_path / "bare.onnx"))
    with pytest.raises(IncompatibleCheckpointError, match="no hospital_rl metadata"):
        OnnxRuntimeBackend(tmp_path / "bare.onnx")


def test_golden_corpus_parity_both_roles(ckpt, tmp_path):
    role, path = ckpt
    export_onnx(path, tmp_path / "m.onnx")
    ref, onx = PythonReferenceBackend(path), OnnxRuntimeBackend(tmp_path / "m.onnx")
    for extreme in (False, True):
        batch = bench.golden_corpus(role, 48, seed=3, extreme=extreme)
        assert bench.parity_error(ref, onx, batch) is None
        assert np.isfinite(batch["node_features"]).all()
    n = batch["node_mask"].sum(-1)  # topology sizes differ, so padding differs, and padding exists
    assert len(set(n.tolist())) >= 4 and n.max() < batch["node_mask"].shape[1]
    assert (batch["action_mask"].sum(-1) >= 1).all() and (~batch["action_mask"]).any()
    assert bench.corpus_sha(bench.golden_corpus(role, 48, seed=3)) == bench.corpus_sha(bench.golden_corpus(role, 48, seed=3))


class _Stub:
    def __init__(self, logits, value=(0.0,)):
        self.out = InferenceOutput(np.asarray(logits, np.float32), np.asarray(value, np.float32), None)

    def run(self, batch, state=None):
        return self.out


def test_near_tie_exception_is_scoped():
    batch = {"action_mask": np.array([[True, True, True]])}
    ref = _Stub([[0.0, 1.0, 1.00001]])
    assert bench.parity_error(ref, _Stub([[0.0, 1.00001, 1.0]]), batch) is None  # argmax flips inside the 1e-4 band
    ref = _Stub([[0.0, 1.0, 2.0]])
    assert bench.parity_error(ref, _Stub([[0.0, 2.0, 1.0]]), batch) is not None  # real disagreement is reported


def test_benchmark_cli_schema_and_refusals(ckpt, tmp_path, monkeypatch, capsys):
    role, path = ckpt
    out = tmp_path / "r.json"
    argv = ["--checkpoint", path, "--backends", "python,onnxruntime", "--batches", "1,4", "--warmup", "1",
            "--repetitions", "3", "--env-steps", "5", "--out", out]
    assert bench.main([str(x) for x in argv]) == 0
    r = json.loads(out.read_text())
    bench.validate_report(r)
    assert r["model"]["role"] == role and {x["backend"] for x in r["rows"]} == {"python", "onnxruntime"}
    assert r["backends"]["onnxruntime"]["env_steps_per_s"] > 0 and r["provenance"]["threads"] == 1
    assert out.read_text().endswith("\n") and list(r) == sorted(r)  # sorted keys: stable layout
    assert bench.main(["--checkpoint", path, "--backends", "bogus"]) == 2

    import hospital_rl.infra.inference as inf

    class Skewed(inf.OnnxRuntimeBackend):
        def run(self, batch, state=None):
            o = super().run(batch)
            return InferenceOutput(o.logits + np.float32(0.5), o.value, None)

    monkeypatch.setattr(inf, "OnnxRuntimeBackend", Skewed)
    assert bench.main(["--checkpoint", path, "--backends", "onnxruntime", "--env-steps", "0"]) == 2  # incomparable
