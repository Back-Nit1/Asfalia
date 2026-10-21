import numpy as np
import pytest

from hospital_rl.infra.inference import OnnxRuntimeBackend, PythonReferenceBackend, export_onnx
from hospital_rl.policies.checkpoint import IncompatibleCheckpointError
from hospital_rl.training.train import TrainConfig, train
from toy_env import ToyGraphEnv, make_env

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

TOL = 1e-4


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("inf")
    cfg = TrainConfig(role="defender", total_steps=256, n_envs=4, rollout_len=32, hidden=32,
                      checkpoint_every=100, env_factory="toy_env:make_env")
    train(cfg, d, log=lambda s: None)  # trained => non-trivial normalizer stats
    meta = export_onnx(d / "latest.pt", d / "policy.onnx")
    return d, meta


def corpus(b, seed=0, max_nodes=8, extreme=False):
    env, rng = ToyGraphEnv(max_nodes=max_nodes), np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(b):
        rows.append(obs)
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        if env.t == 0:
            obs, _ = env.reset()
    batch = {k: np.ascontiguousarray(np.stack([r[k] for r in rows])) for k in rows[0]}
    if extreme:
        batch["node_features"] = (batch["node_features"] * 1e6).astype(np.float32)
        batch["globals"] = (batch["globals"] - 5e5).astype(np.float32)
    return batch


@pytest.mark.parametrize("b", [1, 7, 64])
@pytest.mark.parametrize("extreme", [False, True])
def test_onnx_parity(artifacts, b, extreme):
    d, _ = artifacts
    ref, onnx_b = PythonReferenceBackend(d / "latest.pt"), OnnxRuntimeBackend(d / "policy.onnx")
    batch = corpus(b, seed=b, extreme=extreme)
    r, o = ref.run(batch), onnx_b.run(batch)
    legal = batch["action_mask"]
    assert r.logits.dtype == o.logits.dtype == np.float32 and o.logits.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(o.logits[legal], r.logits[legal], atol=TOL, rtol=TOL)
    assert (o.logits[~legal] == r.logits[~legal]).all()  # masked entries identical (-1e9)
    np.testing.assert_allclose(o.value, r.value, atol=TOL, rtol=TOL)
    top2 = np.sort(np.where(legal, r.logits, -np.inf), -1)[:, -2:]
    tie = (top2[:, 1] - top2[:, 0]) < TOL
    assert (o.logits.argmax(-1)[~tie] == r.logits.argmax(-1)[~tie]).all()


def test_all_nonwait_invalid_is_safe(artifacts):
    d, _ = artifacts
    batch = corpus(4)
    batch["action_mask"][:] = False
    for be in (PythonReferenceBackend(d / "latest.pt"), OnnxRuntimeBackend(d / "policy.onnx")):
        out = be.run(batch)
        assert np.isfinite(out.logits).all() and (out.logits.argmax(-1) == 0).all()


def test_padding_garbage_ignored_by_onnx(artifacts):
    d, _ = artifacts
    be = OnnxRuntimeBackend(d / "policy.onnx")
    batch = corpus(8)
    junk = {k: v.copy() for k, v in batch.items()}
    junk["node_features"][~batch["node_mask"]] = 123.0
    a, b = be.run(batch), be.run(junk)
    np.testing.assert_array_equal(a.logits, b.logits)


def test_metadata_and_incompatible_rejected(artifacts):
    d, meta = artifacts
    m = make_env().metadata
    be = OnnxRuntimeBackend(d / "policy.onnx", m["catalog_hash"], m["observation_schema_hash"])
    assert be.meta["checkpoint_id"] == meta["checkpoint_id"] and be.meta["max_nodes"] == 8
    assert be.meta["normalizer"]["frozen"] is False and "mean" in be.meta["normalizer"]
    with pytest.raises(IncompatibleCheckpointError, match="catalog_hash"):
        OnnxRuntimeBackend(d / "policy.onnx", expected_catalog_hash="nope")


def test_contract_violations_rejected(artifacts):
    d, _ = artifacts
    be = OnnxRuntimeBackend(d / "policy.onnx")
    batch = corpus(3)
    with pytest.raises(ValueError, match="node_features"):
        be.run({**batch, "node_features": batch["node_features"].astype(np.float64)})
    with pytest.raises(ValueError, match="state must be None"):
        be.run(batch, state=np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="adjacency"):
        be.run({**batch, "adjacency": batch["adjacency"][:, :4, :4]})
    with pytest.raises(ValueError, match="missing"):
        be.run({k: v for k, v in batch.items() if k != "globals"})
