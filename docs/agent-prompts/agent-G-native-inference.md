# Agent G — productize policy export and native inference benchmarking

Finish the optional native inference path around a real compatible hospital checkpoint. Python remains the reference
and default unless measurement shows another backend is better.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` section 18, `docs/training_infra.md`,
`src/hospital_rl/infra/inference.py`, `native/`, `benchmarks/inference/`, and inference tests.

## Ownership

- `src/hospital_rl/infra/inference.py`
- New `src/hospital_rl/infra/inference_cli.py`
- `native/`
- `benchmarks/inference/`
- Inference/native tests under `tests/infra/`
- Native/inference documentation
- `docs/requests/agent-G.md` for shared CLI/dependency/dashboard metadata changes

Do not edit training semantics, engine/env code, general infrastructure lifecycle, API/frontend, or `cli.py`.

## Deliverables

1. Implement a real `inference_cli.main(argv)` supporting explicit checkpoint, output, format, role/hash expectations,
   overwrite policy, and machine-readable result. Export atomically and fail clearly on incompatible/missing input.
2. Make the benchmark importable through a package entry point suitable for `hospital-rl benchmark-inference`, with
   explicit checkpoint, backends, batches, warmup, repetitions, threads, output, and dashboard-enabled comparison.
3. Benchmark Python eager, Python batched/compiled when supported, ONNX Runtime, and native C++ on identical raw inputs.
   Measure preprocessing + inference + postprocessing p50/p95, throughput, memory when reliably available, cold load,
   and full environment steps/sec. Record hardware/runtime/thread provenance.
4. Use a real hospital-role checkpoint and a golden observation corpus spanning both roles, several topology sizes,
   padding, masks, and extreme finite inputs. Keep generated model/corpus binaries out of git unless they are tiny,
   justified fixtures.
5. Preserve metadata/hash/normalization validation across every backend. Reject missing metadata and incompatible
   catalog/observation/policy versions.
6. Keep stochastic parity scoped to distributions or deterministic action selection. Preserve the documented near-tie
   exception and test it.
7. Make the CMake build portable and provide precise prerequisite diagnostics. Test macOS arm64 when available; add CI-
   friendly Linux instructions/tests without claiming an unexecuted platform.
8. If native is slower, keep Python/ONNX as default and report the result faithfully. Do not create a speedup target.
9. Leave GRU export unsupported unless it is implemented end-to-end with sequence/state/reset parity tests.

## Acceptance

- Export CLI works against a real compatible checkpoint and produces verifiable metadata.
- Benchmark CLI produces deterministic schema-valid JSON and refuses incomparable backends.
- Existing ONNX parity tests and new CLI/metadata/golden-corpus tests pass.
- Native tests pass when the toolchain is available and skip with one precise reason otherwise.
- `PYTHONPATH=src .venv/bin/pytest -q tests/infra/test_inference.py tests/infra/test_native.py` passes or reports the
  documented toolchain-only skip.
- Submit exact CLI wiring and optional dependency changes to the lead.

## Non-goals

- Do not port the whole simulator to C++ without profiling evidence.
- Do not make native execution mandatory for normal training/evaluation.
