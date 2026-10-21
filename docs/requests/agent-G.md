# Requests from Agent G (native inference) to the lead

> **HARD RULE: commit at least every 4 minutes** (see CLAUDE.md).

## 1. `cli.py` wiring
`export-policy` is already delegated to `hospital_rl.infra.inference_cli` (now a real `main(argv)`).
`benchmark-inference` is not. Exact change:

```diff
 _DELEGATE = {..., "export-policy": "hospital_rl.infra.inference_cli",
+             "benchmark-inference": "hospital_rl.infra.inference_bench"}
```
(Also needs Agent C's `add_help=False` fix from `docs/requests/agent-C.md` so `--help` reaches these parsers.)
Exit codes: **0** done; **2** unusable input / refusal / incomparable backends (argparse errors are also 2).

## 2. Dependencies (`pyproject.toml`)
No new packages. Suggested, optional: a lighter extra for people who only export/benchmark ONNX:
```toml
inference = ["onnx", "onnxruntime"]      # dev already contains both
```
`native = ["cmake", "pybind11"]` stays; the ONNX Runtime **C++ release** is not pip-installable (see `docs/training_infra.md`).

## 3. Dashboard metadata
`benchmark-inference` writes `schema_version: 1` JSON with a flat `rows` list
(`backend, batch, p50_us, p95_us, throughput_samples_per_s`) next to the nested per-backend results, so a dashboard can
chart it without knowing backend internals. I read "dashboard-enabled comparison" as this; if it meant something else
(e.g. emitting events while stepping), tell me. No dashboard/API code was touched.

## 4. Test inputs
`tests/infra/test_inference_cli.py` trains tiny hospital checkpoints per role unless `HOSPITAL_CKPT_ATTACKER` /
`HOSPITAL_CKPT_DEFENDER` point at real ones. No binaries are committed.
