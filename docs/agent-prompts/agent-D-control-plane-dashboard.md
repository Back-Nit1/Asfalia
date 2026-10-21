# Agent D — wire the experiment control plane into the API and dashboard

Turn the currently honest `not_configured` surfaces into a durable local experiment controller and a functional UI.
Keep the existing development-task orchestrator separate from experiment scheduling.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` sections 14–16, `docs/dashboard.md`, all files under
`src/hospital_rl/api/`, `src/hospital_rl/orchestration/`, `frontend/src/`, and their tests.

## Ownership

- `src/hospital_rl/api/`
- New `src/hospital_rl/experiments/` control-plane package if needed
- `frontend/`
- `tests/api/`, new `tests/experiments/`, and `frontend/e2e/`
- `docs/dashboard.md`
- `docs/requests/agent-D.md` for shared CLI, training, infrastructure, or event-schema changes

Do not edit PPO/training semantics, cloud backend internals, engine/env code, or `src/hospital_rl/cli.py`.

## Deliverables

1. Add a deterministic experiment controller with typed run commands and durable states: queued, starting, running,
   draining, checkpointing, stopped, failed, cancelled, and completed. Persist run ownership, attempts, backend job ID,
   resolved config, resource caps, artifact references, timestamps, errors, and resume lineage in SQLite.
2. Start local jobs asynchronously through an injected backend interface. API requests must return a durable run ID
   immediately. Reconstruct honest state after process restart; never manufacture completion.
3. Implement typed, idempotent APIs for config validation, run create/status/list, checkpoint-and-stop, drain, force
   cancel, resume, checkpoints/metadata, metrics, artifact listing/download metadata, evaluation submission/results,
   infrastructure status, and owned-resource cleanup. Authorize and audit every mutation.
4. Ingest real training metrics, checkpoint metadata, job state, and selected episode events. Keep development,
   experiment, world, and infrastructure streams separated by source kind and authorization.
5. Connect completed episode exports to the existing replay library without copying evaluator truth into attacker or
   defender DTOs. Add a sampled read-only live feed at bounded frequency; training must not wait for rendering.
6. Replace disabled Training controls with validated preset selection, resolved-config preview, resource caps/estimate,
   start, drain, checkpoint-and-stop, resume, and bounded evaluation flows.
7. Implement Experiments views for baseline/transfer/adaptation comparisons, confidence intervals, cross-play matrices,
   and compatibility errors. Render only stored real results.
8. Implement Infrastructure and Artifacts views with lifecycle state, heartbeat/age, failures, policy versions,
   estimate freshness, checkpoint lineage, hashes, provenance, and cleanup status.
9. Preserve and extend replay controls. Add a separate debug-fork command whose outputs are marked DEBUG and excluded
   from benchmark result queries.
10. Add accessible loading, empty, stale, disconnected, partial-artifact, and backend-error states. Do not replace
    honest absence with animated fake counters.

## Acceptance

- API tests cover idempotency, auth, restart recovery, legal run transitions, async submission, cancellation modes,
  resume lineage, stale heartbeats, artifact metadata, evaluator isolation, and SSE resync.
- Browser tests cover: validate preset → start bounded local run → observe real metrics → checkpoint-and-stop → resume →
  select two checkpoints → frozen evaluation → view replay/comparison. Keep budgets tiny.
- Existing build-board and perspective-isolation tests remain green.
- `PYTHONPATH=src .venv/bin/pytest -q tests/api tests/orchestration tests/experiments` passes.
- `cd frontend && npm run build && npm run test:e2e` passes.
- Write a precise backend contract request for Agent E and CLI/event-schema requests for the lead.

## Non-goals

- Do not launch cloud work from tests.
- Do not place provider credentials in SQLite, responses, frontend storage, or logs.
- Do not make the coding-agent scheduler responsible for RL experiments.
