# Dashboard, API and orchestrator (Agent A)

## Run

```bash
# API + built UI (localhost only by default); fixture episode is always available
PYTHONPATH=src python -m hospital_rl.api --port 8000 --plan configs/orchestrator.yaml --run-scheduler
# UI dev server (proxies /api to :8000)
cd frontend && npm install && npm run dev
# plan check without running anything
PYTHONPATH=src python -m hospital_rl.orchestration --dry-run
```

Flags: `--db` (SQLite WAL file), `--episodes-dir` (extra episode JSON exports), `--no-agent` (never start the coding
runtime), `--stream-interval` (SSE poll seconds, 0.2–0.5 = 2–5 Hz; UI also takes `?hz=` and `?mhz=`).
The lead's `hospital-rl dashboard|orchestrate` stubs should call these entry points (see `docs/requests/agent-A.md`).

## Security model

- Default bind is `127.0.0.1`. Any other host requires `HOSPITAL_RL_OPERATOR_TOKEN`; the process refuses to start otherwise.
- `HOSPITAL_RL_EVALUATOR_TOKEN` must differ from the operator token. Evaluator endpoints (`/api/world/evaluator/*`) always
  require it, return 403 for the operator token, 503 when unset, and carry `privileged: true` plus a notice in every payload.
- `/api/world/{attacker,defender,evaluator}/…` are the only routes meant for policy-facing use. They read world-source events
  only and filter by `visibility` server-side with separate response models; the attacker DTO has no `edges` or `final_metrics`
  field. Tasks and development events are not reachable from them.
- Provider keys never enter responses. Acceptance commands run with secret-looking env vars removed; stored stdout/stderr is
  redacted. Chain-of-thought blocks from the agent stream are dropped, never stored.

## Events and storage

SQLite (WAL, one writer connection). `events.cursor` is the ingestion cursor used by SSE and `Last-Event-ID`. Dedupe is by
`event_id`; a different `event_id` reusing a `(source_kind, worker_id, sequence)` is rejected. Order is guaranteed per
`(source, worker)` via `Store.stream_events`; the cursor feed is arrival order and has no cross-worker ordering. `prune_events(keep)`
bounds retention; a client whose cursor is older than `pruned_through` receives a `resync` event with a fresh snapshot.
Mutations require an `Idempotency-Key`; repeating a key on the same method+path replays the stored response.

## Orchestrator

State machine: queued → ready → running → review → passed → integrated, plus failed / blocked / cancelled
(`orchestration/tasks.py`). A task is ready only when every dependency is `integrated`. `passed` happens only after the
orchestrator itself ran every acceptance command in the task worktree and all exited 0 (no commands = not passable; changed
files outside `assigned_paths` fail the review). Leases are renewed by agent events or `heartbeat`; expiry, timeout, agent
failure or a rejected merge consume an attempt and retry up to `retry_cap` (2) before `blocked`. At most one active
(running/review/passed-not-integrated) task may own overlapping paths, so lockfile and shared-schema work is serialized.
`integrate` merges into a dedicated `integration_repo` checkout, runs `combined_suite`, and resets the merge if it fails.
Without a `claude` binary the UI says "coding runtime not configured"; human-owned tasks can be claimed and submitted.
`ClaudeCodeCliAdapter` runs `claude -p <prompt> --output-format stream-json --verbose` in the task worktree.

## Fixture vs real

- `api/fixtures/fixture_tiny_episode.json`: a recorded scripted tiny episode (engine at `92b49cd`, seed 7) with topology copied
  from `configs/tiny.yaml`. The UI labels it FIXTURE and REPLAY. Node overlays are derived from role-visible events; the
  export carries no per-tick engine state, budget or workflow metrics, so those cards say "not exported".
- `api/replay.py::parse_episode` is the single adapter to the episode export format.
- Runs, checkpoints, metrics (beyond task counts) and infrastructure return `configured: false`. No live episode producer exists,
  so LIVE shows "not configured".
