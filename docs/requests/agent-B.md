# Requests from Agent B to the lead

## 1. Dependencies (pyproject.toml) — needed for ONNX export/parity and cloud
Why: `infra/inference.py` exports/validates with `onnx` + `onnxruntime`; `infra/modal_app.py` needs `modal`;
the native runner build needs `cmake` + `pybind11` plus an ONNX Runtime C++ release (not a pip package).
Tests importorskip these, so nothing breaks without them — but nothing is verified either.
```diff
 [project.optional-dependencies]
-dev = ["pytest", "httpx"]
+dev = ["pytest", "httpx", "onnx", "onnxruntime"]
+cloud = ["modal"]
+native = ["cmake", "pybind11"]
```
Verified versions: torch 2.14.1, onnx 1.23.2, onnxruntime 1.31.0, modal 1.6.2, pybind11 3.1.0, cmake 4.4.4, ORT C++ release 1.31.0.

## 2. Action index layout — please confirm
I assume `index = 1 + action_type * max_nodes + node` (type-major, wait = 0). The policy's logits are built
this way; the real env's catalog must match or checkpoints are meaningless. If the env is node-major, tell me
and I change `GraphPolicy.forward` (one reshape) and `base.action_index`.

## 3. Env contract I rely on
- `env_factory` config string `"package.module:callable"`, called with `env_kwargs`, returning a Gymnasium env.
- `observation_space` is a Dict with exactly `node_features (M,F)`, `node_mask (M,)`, `adjacency (M,M)`,
  `history (H,E)`, `globals (G,)`, `action_mask (N,)`; `N = 1 + n_types*M`; wait (index 0) always legal.
- `env.metadata` has `role`, `catalog_hash`, `observation_schema_hash` (checked against config + checkpoint).
- Truncation (time limit) must be `truncated=True`, true terminal `terminated=True` (truncation is bootstrapped).
- Two-role envs: PettingZoo-ParallelEnv-shaped (`policies/base.py::TwoRoleEnv`). Scripted opponents need a
  `FrozenActor` (`id`, `reset()`, `act(obs)->int`); I did not write adapters for your scripted policies.

## 4. CLI wiring (cli.py is yours)
- `train` -> `hospital_rl.training.train.main(argv)` (`--config` TrainConfig JSON, `--out`, `--resume`, `--control-file`)
- `cloud-plan` -> `hospital_rl.infra.plan.main(argv)` (`--backend modal|local|daytona --config configs/cloud_<x>.yaml`)
- `export-policy` -> `hospital_rl.infra.inference.export_onnx(checkpoint, out)`
- `benchmark-inference` -> `benchmarks/inference/run.py` (`main(argv)`)
- Cloud `train --backend modal` -> build `ModalConfig` via `infra.modal_app.load_modal_config`, `ModalBackend(cfg, RealModalGateway(cfg))`.

## 5. Process-level constraints you should know
- `train()` seeds and uses the process-global torch/numpy/python RNG: one training run per process.
- Bitwise resume was verified single-process on CPU with default threads; see docs/training_infra.md.
