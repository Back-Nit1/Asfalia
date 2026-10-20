"""Native parity. Skipped (not passed) unless native/ was built: see docs/training_infra.md."""
import numpy as np
import pytest

from hospital_rl.infra.inference import NativeBackend, PythonReferenceBackend, import_native
from test_inference import TOL, artifacts, corpus  # noqa: F401  (artifacts is a fixture)

try:
    import_native()
except ImportError:
    pytest.skip("native module not built (needs cmake, pybind11, ONNX Runtime C++ release)", allow_module_level=True)


@pytest.mark.parametrize("b", [1, 7, 64])
def test_native_parity_with_python(artifacts, b):  # noqa: F811
    d, _ = artifacts
    ref, nat = PythonReferenceBackend(d / "latest.pt"), NativeBackend(d / "policy.onnx")
    batch = corpus(b, seed=b)
    r, n = ref.run(batch), nat.run(batch)
    legal = batch["action_mask"]
    np.testing.assert_allclose(n.logits[legal], r.logits[legal], atol=TOL, rtol=TOL)
    np.testing.assert_allclose(n.value, r.value, atol=TOL, rtol=TOL)
    top2 = np.sort(np.where(legal, r.logits, -np.inf), -1)[:, -2:]
    tie = (top2[:, 1] - top2[:, 0]) < TOL
    assert (n.logits.argmax(-1)[~tie] == r.logits.argmax(-1)[~tie]).all()


def test_native_rejects_bad_input(artifacts):  # noqa: F811
    d, _ = artifacts
    nat = NativeBackend(d / "policy.onnx")
    batch = corpus(2)
    with pytest.raises(ValueError):
        nat.run({**batch, "node_features": batch["node_features"].astype(np.float64)})
    with pytest.raises(TypeError):  # bypass the Python validator: C++ side must still refuse
        nat.runner.run(batch["node_features"].astype(np.float64), batch["node_mask"], batch["adjacency"],
                       batch["history"], batch["globals"], batch["action_mask"])
