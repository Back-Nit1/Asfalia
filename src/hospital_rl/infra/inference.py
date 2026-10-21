"""Inference backends and ONNX export for the feed-forward policy.

``InferenceBackend.run(batch, state=None) -> InferenceOutput``

Input semantics (all backends):
  * ``batch`` keys/dtypes/shapes, B = batch size, M = max_nodes (fixed by the exported model):
      node_features float32 (B, M, F)   node_mask bool (B, M)       adjacency bool (B, M, M)
      history       float32 (B, H, E)   globals   float32 (B, G)    action_mask bool (B, N)
    N = 1 + n_action_types * M. Features are RAW (un-normalized); the backend applies the
    checkpoint's frozen normalizer. Arrays must be C-contiguous (``np.ascontiguousarray`` otherwise
    -> ValueError, never a silent copy-and-cast). Dtype or shape mismatch -> ValueError.
  * ``state``: None for feed-forward models; recurrent models take float32 (B, S) and the caller
    zeroes rows at episode boundaries. Passing state to a feed-forward model -> ValueError.
  * CPU only. Inputs are never mutated and never aliased by the outputs.
Output: logits float32 (B, N) C-contiguous with illegal actions at -1e9 (wait forced legal);
  value float32 (B,); new state (None for feed-forward). Action selection (argmax/sampling) is the caller's.
Parity: logits agree within atol=rtol=1e-4 on legal entries; argmax agrees except where the top-2
legal logits differ by < 1e-4 (documented tie band).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from hospital_rl.policies.base import OBS_KEYS
from hospital_rl.policies.checkpoint import (IncompatibleCheckpointError, build_ppo, load_checkpoint)
from hospital_rl.policies.ppo import PPOConfig, to_torch

DTYPES = {"node_features": np.float32, "node_mask": np.bool_, "adjacency": np.bool_,
          "history": np.float32, "globals": np.float32, "action_mask": np.bool_}


class InferenceOutput(NamedTuple):
    logits: np.ndarray
    value: np.ndarray
    state: np.ndarray | None


def expected_shapes(meta: dict, b: int) -> dict[str, tuple]:
    c, m = meta["policy_config"], meta["max_nodes"]
    return {"node_features": (b, m, c["node_dim"]), "node_mask": (b, m), "adjacency": (b, m, m),
            "history": (b, 0, c["hist_dim"]), "globals": (b, c["glob_dim"]),
            "action_mask": (b, 1 + c["n_action_types"] * m)}


def validate_batch(batch: dict[str, np.ndarray], meta: dict, state) -> int:
    if state is not None:
        raise ValueError("this model is feed-forward; state must be None")
    missing = set(OBS_KEYS) - set(batch)
    if missing:
        raise ValueError(f"missing batch keys: {sorted(missing)}")
    b = batch["node_mask"].shape[0]
    for k, want in expected_shapes(meta, b).items():
        x = batch[k]
        shape_ok = len(x.shape) == len(want) and all(w in (0, s) for w, s in zip(want, x.shape))  # 0 = free (history length)
        if not shape_ok or x.dtype != DTYPES[k] or not x.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{k}: got {x.dtype}{x.shape} contiguous={x.flags['C_CONTIGUOUS']}, "
                             f"want {np.dtype(DTYPES[k])}{tuple('*' if w == 0 else w for w in want)} C-contiguous")
    return b


class InferenceBackend(ABC):
    name: str
    meta: dict

    @abstractmethod
    def run(self, batch: dict[str, np.ndarray], state: np.ndarray | None = None) -> InferenceOutput: ...


class PythonReferenceBackend(InferenceBackend):
    """Eager PyTorch with the checkpoint's real Normalizer. The oracle for every other backend."""
    name = "python"

    def __init__(self, checkpoint: str | Path):
        ckpt = load_checkpoint(checkpoint)
        self.meta = ckpt.meta.model_dump()
        ppo = build_ppo(ckpt, PPOConfig(), restore=False)
        ppo.normalizer.frozen = True
        self.policy, self.normalizer = ppo.policy.eval(), ppo.normalizer

    @torch.no_grad()
    def run(self, batch, state=None):
        validate_batch(batch, self.meta, state)
        logits, value, _ = self.policy(to_torch(self.normalizer(batch)), None)
        return InferenceOutput(np.ascontiguousarray(logits.numpy(), np.float32),
                               np.ascontiguousarray(value.numpy(), np.float32), None)


class _ExportWrapper(nn.Module):
    """Raw obs -> normalize (constants baked in) -> policy. What ONNX/native runtimes execute."""

    def __init__(self, policy: nn.Module, norm_stats: dict):
        super().__init__()
        self.policy = policy
        for k in ("node_features", "history", "globals"):
            self.register_buffer(f"mean_{k}", torch.tensor(norm_stats["mean"][k], dtype=torch.float32))
            self.register_buffer(f"std_{k}", torch.tensor(norm_stats["var"][k], dtype=torch.float32).add(1e-8).sqrt())
        self.clip = float(norm_stats["clip"])

    def _n(self, k, x):
        return ((x - getattr(self, f"mean_{k}")) / getattr(self, f"std_{k}")).clamp(-self.clip, self.clip)

    def forward(self, node_features, node_mask, adjacency, history, globals_, action_mask):
        nf = torch.where(node_mask.unsqueeze(-1), self._n("node_features", node_features), torch.zeros(()))
        logits, value, _ = self.policy({"node_features": nf, "node_mask": node_mask, "adjacency": adjacency,
                                        "history": self._n("history", history), "globals": self._n("globals", globals_),
                                        "action_mask": action_mask})
        return logits, value


def export_onnx(checkpoint: str | Path, out: str | Path) -> dict:
    """Export the feed-forward policy; metadata goes in the model's metadata_props and ``<out>.meta.json``."""
    import onnx

    ckpt = load_checkpoint(checkpoint)
    meta = ckpt.meta.model_dump()
    ppo = build_ppo(ckpt, PPOConfig(), restore=False)
    wrapper = _ExportWrapper(ppo.policy.eval(), meta["normalizer"]).eval()
    shapes = expected_shapes(meta, 2)
    h = meta["policy_config"].get("hist_len", 3)  # history length is free in the graph; 3 only for tracing
    shapes["history"] = (2, h, shapes["history"][2])
    dummy = tuple(torch.zeros(shapes[k], dtype=torch.from_numpy(np.zeros(0, DTYPES[k])).dtype) for k in OBS_KEYS)
    dummy[1][:, :3] = True  # a few valid nodes
    dummy[5][:, 0] = True
    names = ["node_features", "node_mask", "adjacency", "history", "globals", "action_mask"]
    torch.onnx.export(wrapper, dummy, str(out), input_names=names, output_names=["logits", "value"],
                      dynamic_axes={n: {0: "batch"} for n in names + ["logits", "value"]} | {"history": {0: "batch", 1: "hist_len"}},
                      opset_version=17, dynamo=False)
    export_meta = {k: meta[k] for k in ("checkpoint_id", "role", "catalog_hash", "observation_schema_hash", "max_nodes",
                                         "policy_config", "normalizer", "policy_version", "training_steps")}
    export_meta["exporter"] = {"torch": torch.__version__, "onnx": onnx.__version__, "opset": 17}
    model = onnx.load(str(out))
    del model.metadata_props[:]
    entry = model.metadata_props.add()
    entry.key, entry.value = "hospital_rl", json.dumps(export_meta)
    onnx.save(model, str(out))
    Path(str(out) + ".meta.json").write_text(json.dumps(export_meta, indent=2))
    return export_meta


class OnnxRuntimeBackend(InferenceBackend):
    name = "onnxruntime"

    def __init__(self, path: str | Path, expected_catalog_hash: str | None = None,
                 expected_obs_hash: str | None = None, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        props = self.sess.get_modelmeta().custom_metadata_map
        if "hospital_rl" not in props:
            raise IncompatibleCheckpointError(f"{path} has no hospital_rl metadata; refusing to run an unidentified model")
        self.meta = json.loads(props["hospital_rl"])
        for name, got, want in (("catalog_hash", self.meta["catalog_hash"], expected_catalog_hash),
                                ("observation_schema_hash", self.meta["observation_schema_hash"], expected_obs_hash)):
            if want is not None and got != want:
                raise IncompatibleCheckpointError(f"onnx model {name}: model has {got!r}, expected {want!r}")

    def run(self, batch, state=None):
        validate_batch(batch, self.meta, state)
        logits, value = self.sess.run(None, {k: batch[k] for k in OBS_KEYS})
        return InferenceOutput(np.ascontiguousarray(logits, np.float32), np.ascontiguousarray(value, np.float32), None)


def import_native():
    """Import the pybind11 module built from native/ (dir: $HOSPITAL_NATIVE_DIR or <repo>/native/build)."""
    import importlib
    import os
    import sys

    d = os.environ.get("HOSPITAL_NATIVE_DIR") or str(Path(__file__).parents[3] / "native" / "build")
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("hospital_policy_runner")


class NativeBackend(InferenceBackend):
    """ONNX Runtime C++ via pybind11 (in-process, GIL released during run). Same contract as the others;
    the C++ side re-checks dtype/contiguity (noconvert) so a bad batch is an error, never a silent cast."""
    name = "cpp"

    def __init__(self, path: str | Path, expected_catalog_hash: str | None = None,
                 expected_obs_hash: str | None = None, threads: int = 1):
        self.runner = import_native().PolicyRunner(str(path), threads)
        self.meta = json.loads(self.runner.metadata_json())
        for name, got, want in (("catalog_hash", self.meta["catalog_hash"], expected_catalog_hash),
                                ("observation_schema_hash", self.meta["observation_schema_hash"], expected_obs_hash)):
            if want is not None and got != want:
                raise IncompatibleCheckpointError(f"onnx model {name}: model has {got!r}, expected {want!r}")

    def run(self, batch, state=None):
        validate_batch(batch, self.meta, state)
        logits, value = self.runner.run(*(batch[k] for k in OBS_KEYS))
        return InferenceOutput(logits, value, None)
