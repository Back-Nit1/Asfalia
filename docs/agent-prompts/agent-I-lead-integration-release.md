# Agent I — lead integration, CLI repair, task-graph repair, and release acceptance

Run this prompt only after specialist branches and request files are available. You own shared integration and must
review evidence rather than trusting completion claims. Preserve the separation between development orchestration,
experiment control, and learned policies.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md`, every `docs/requests/agent-*.md`, specialist commit histories,
and all changed code/tests. Inspect the working tree before editing.

## Ownership

- `src/hospital_rl/cli.py`
- `src/hospital_rl/events.py`, `src/hospital_rl/schemas.py`, and `pyproject.toml`/`uv.lock` when justified
- `configs/orchestrator.yaml`
- Top-level `README.md`, contracts/status/limitations documentation, and agent-request cleanup
- Cross-component integration tests and release scripts
- Conflict resolution across specialist branches

## Deliverables

1. Review each specialist diff, commit list, acceptance output, and request file. Reject placeholder-only work, fake
   metrics, credential leakage, cross-plane coupling, test-split leakage, or unverifiable claims.
   Explicitly verify that no runtime source reads `.env`, including cloud-school launchers.
2. Repair the CLI as a real command surface. `--help` for delegated commands must show their actual arguments. Implement
   and test `train`, `evaluate`, `adapt`, `orchestrate`, `dashboard`, `cloud-plan`, `cleanup`, `export-policy`, and
   `benchmark-inference`; add structured `workers` commands only if the backend supports them. Remove commands that
   remain intentionally unavailable rather than leaving imports that crash.
3. Make config resolution explicit and consistent. Ensure `benchmark --config` either fully honors the config or is
   removed. Ensure evaluation/adaptation use complete hospital settings and record the resolved form.
4. Repair `configs/orchestrator.yaml` so every assigned path and acceptance command exists and represents current work.
   Add plan validation for missing paths/commands where safe. Make orchestration dry-run and live behavior discoverable.
5. Integrate experiment-controller events/DTOs without weakening attacker/defender/evaluator authorization. Provider
   and coding-agent credentials must never enter policy inputs, browser responses, acceptance subprocesses, or logs.
6. Replace stale documentation and resolved request notes. Document exact clean setup, local end-to-end commands,
   bounded cloud opt-in, artifact locations, stop/resume semantics, failure cases, and what remains deferred.
7. Regenerate figures/reports only from checked scripts and actual artifacts. Add dashboard screenshots only if they are
   captured from the real working UI and label fixture/replay/live states correctly.
8. Run the clean end-to-end acceptance sequence:
   - install/sync from the lockfile;
   - full Python tests;
   - frontend typecheck/build/browser workflow;
   - tiny scripted demo and replay;
   - attacker and defender bounded school smoke;
   - frozen Stage D hospital evaluation;
   - one-side Stage E adaptation and cross-play;
   - trajectory export/validation and report generation;
   - local stop/resume/cleanup;
   - ONNX export and inference benchmark;
   - Modal/Daytona only with explicit authorization, credentials, and hard caps.
9. Record exact commands, wall times, skips, failures, seeds, checkpoint/report paths, and hashes. Do not commit large
   generated run artifacts unless a small fixture is necessary and reviewed.
10. Update the status table so implemented, unit-tested, smoke-tested, experimentally evaluated, and deferred are
    distinct. Never convert a skipped or fake-provider test into a live-platform claim.

## Acceptance

- `git status` is clean after generated artifacts are removed or ignored appropriately.
- `uv run pytest -q` passes from a clean environment; any skip is listed with a precise prerequisite.
- `cd frontend && npm run build && npm run test:e2e` passes.
- Every documented command has been executed against the current implementation or explicitly labeled optional and
  unexecuted.
- Broken imports, ignored config flags, nonexistent orchestrator paths, stale request notes, and disabled core UI
  controls are gone.
- The final report lists evidence for each original gate and every remaining deferred item.

## Non-goals

- Do not run expensive multi-seed or cloud experiments without an explicit budget.
- Do not merge merely because individual branches pass in isolation; run the combined suite after each integration.
- Do not weaken safety or information-separation rules to make the demo easier.
