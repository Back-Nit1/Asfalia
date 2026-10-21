# Local release acceptance — 2026-10-21

This is execution evidence for the merged release candidate, not a policy-quality or live-cloud claim. Generated
artifacts were written under `/private/tmp/asfalia-release.EALPBY` and were not committed.

## Automated suites

| Check | Result |
| --- | --- |
| `UV_CACHE_DIR=/private/tmp/asfalia-uv-cache uv run pytest -q` | 411 passed, 4 skipped, 87 warnings in 78.14 s |
| `cd frontend && npm run build` | TypeScript check and Vite production build passed |
| `cd frontend && npm run test:e2e` | 2 Playwright workflows passed in 11.3 s |

The four Python skips were the opt-in live Modal smoke, unavailable Modal SDK tests in the default environment, and the
optional native C++ module (requires CMake, pybind11, and an ONNX Runtime C++ release). Python and ONNX Runtime inference
tests did run. Deprecation warnings are from FastAPI event hooks and the legacy PyTorch ONNX exporter.

## Bounded local workflow

All commands used the locked environment and completed successfully:

- `hospital-rl demo ...` and `hospital-rl replay ...`: seed 7, real tiny engine episode.
- `hospital-rl train ... --steps 128 --n-envs 2` for attacker and defender: both wrote compatible real checkpoints.
- `hospital-rl evaluate ... --episodes 2`: frozen Stage D cross-play report written.
- `hospital-rl adapt --trainable-role defender ... --steps 128 --league off`: 128 adaptation steps; opponent remained frozen.
- A second `hospital-rl evaluate ... --episodes 1` loaded the adapted checkpoint successfully.
- `hospital-rl export-trajectories ... --episodes 2` and `validate-trajectories`: two manifests validated, 40 rows each.
- `hospital-rl report` over the two checked fixture seeds: deterministic report/SVG set written and labelled
  `INSUFFICIENT: 2 seeds < 5`.
- `hospital-rl cloud-plan --backend local ...` and local `cleanup` without `--apply`: resolved caps and performed a dry run.
- `hospital-rl export-policy` and `benchmark-inference --backends python,onnxruntime --batches 1,4`: export, parity,
  environment stepping, and benchmark report passed. These tiny timings are machine-specific and are not a general
  performance claim.
- `hospital-rl orchestrate --plan configs/orchestrator.yaml --dry-run`: loaded the seven-task acyclic plan; every assigned
  path exists.

## Explicitly not executed

- Modal and Daytona live smoke scripts: require provider credentials, spend external resources, and were not authorized
  as part of this bounded release pass. Their lifecycle, safety, and fake-provider contracts are covered by tests.
- Native C++ inference: the optional ONNX Runtime C++ distribution was not installed. Python and ONNX Runtime backends
  were exercised instead.
- Powered multi-seed policy experiments: the bounded 128-step runs only verify the pipeline. No superiority,
  clinical-effectiveness, or real-world transfer claim follows.
- Engine-changing diagnostic ablations (`no_workflow_cost`, `no_domain_randomization`,
  `full_observability_diagnostic`, alternate telemetry-loss levels): definitions and refusal paths are tested; engine
  support remains deferred rather than simulated or mislabeled.
