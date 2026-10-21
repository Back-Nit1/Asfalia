# Dashboard, API, orchestrator (Agent A) and experiment control plane (Agent D)

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
- Without `--experiments-root` runs, checkpoints, metrics (beyond task counts) and infrastructure return `configured: false`.
  No live episode producer exists, so LIVE shows "not configured".

## Experiment control plane (Agent D)

```bash
# real local jobs (infra.local_backend.LocalBackend, one subprocess per attempt); no evaluator is wired, so evaluations are refused
PYTHONPATH=src python -m hospital_rl.api --experiments-root runs/exp --run-scheduler
# deterministic FAKE backend + evaluator (UI/e2e/dev only; every surface says "fake")
PYTHONPATH=src python -m hospital_rl.api --experiments-root runs/exp --fake-backend --tick-seconds 0.4
```

Code: `src/hospital_rl/experiments/` (`states`, `config`, `db`, `reader`, `controller`, `fakes`), routes in `api/experiment_routes.py`.
The coding-agent scheduler (`orchestration/`) is untouched and never schedules RL runs.

**Controller.** Commands (`create`, `drain`, `checkpoint_and_stop`, `cancel`, `resume`, `submit_eval`, `debug_fork`, `cleanup`)
only write intent/state to SQLite and return; `tick()` (a thread in the API process) submits to the injected `JobBackend`
(`infra/backend.py`), polls it, and ingests the files the job wrote (`metrics.jsonl`, `ckpt_*.pt(.meta.json)`, `result.json`,
`manifest.json`). `POST /api/runs` therefore returns a durable `run-NNNN` in `queued` immediately.
States: queued → starting → running → draining | checkpointing → stopped | failed | cancelled | completed. The legal table is
`experiments/states.py`; anything else is HTTP 409. `stopped`/`failed`/`cancelled` runs with a checkpoint can `resume`
(new attempt `aN`, `resume_from`/`resume_checkpoint` recorded; the backend resumes from the attempt's `latest.pt`, so only the
latest checkpoint is resumable and another id is a 422). One active-or-queued run at a time (`max_active`).

**Honesty rules.** `completed` needs backend `DONE` *and* a `result.json`; the manifest check result is stored (`artifact_problems`).
If the backend no longer knows a job (process restart with the in-memory local backend) the controller looks only at durable
files: a verified `result.json` decides, otherwise the run becomes `failed: outcome unknown`, never `completed`. Heartbeat is the
last successful backend poll; older than `stale_after` (30 s) flags `stale` without changing state; backend exceptions are shown as
`backend_error` (credential-looking text redacted) and do not refresh the heartbeat. Step-rate estimates come only from stored
metric rows (`t` + `env_steps`); otherwise "unavailable".

**API (operator token; the evaluator token is rejected).** `GET /api/experiments/config`, `POST /api/runs/validate`,
`POST|GET /api/runs`, `GET /api/runs/{id}[/metrics|/checkpoints|/artifacts]`, `POST /api/runs/{id}/{drain|checkpoint-stop|cancel|resume|debug-fork}`,
`GET /api/checkpoints`, `POST|GET /api/evaluations[/{id}]`, `GET /api/experiments/results`, `GET /api/debug-forks`,
`GET /api/infrastructure`, `POST /api/infrastructure/{drain|cleanup}`, `GET /api/audit`, SSE `GET /api/experiments/stream?run_id=`.
Mutations need `Idempotency-Key` (replayed byte-for-byte, refusals included) and are written to `exp_audit` (actor from `X-Actor`,
default `operator`; ok and refused). Config errors are 422 with every problem in `errors`. Artifact endpoints return metadata
(path, size, sha256, manifest problems); there is no file download route.

**Streams.** Experiment events are `source_kind=experiment` (`training_metric` sampled to ≤20 per run per tick, `run_*`,
`checkpoint_written`, `evaluation_finished`). Training episode events are `source_kind=world` with the producer's own `visibility`
(default `evaluator`). `/api/events*` is now development-only; `/api/experiments/stream` is experiment-only and rate-capped at 5 Hz;
`/api/world/<perspective>/stream` is unchanged. A stale cursor gets `resync` with the current run list.

**Evaluation / replay / debug.** Evaluation submits one attacker and one defender checkpoint (hash pinned at submission, re-checked
at run time), ≤20 episodes, `frozen: true`. Compatibility (roles, `catalog_hash`, `observation_schema_hash`, cleaned artifacts) is checked
at submission (422 listing every mismatch). The injected `Evaluator` returns pair results with CI; exported episode JSON goes to
`<root>/episodes` and `Library.rescan()` makes it replayable through the existing perspective routes (evaluator truth stays in
the evaluator DTO). `debug-fork` results carry `debug=1`, label `DEBUG`, are excluded from `/api/experiments/results` and listed only
by `/api/debug-forks`.

**Cleanup.** `POST /api/infrastructure/cleanup?dry_run=true` (default). Only attempts with a DB row, in a terminal run, whose manifest
verifies are deletable; never the newest checkpoint of a resumable run, nor anything a queued/running evaluation reads; strangers in the
artifact root are reported as orphans and never touched.

**UI.** Training (validate → resolved-config preview with caps/estimate → start; drain / checkpoint-and-stop / force cancel / resume;
metric chart; live feed; checkpoints with debug fork; frozen evaluation of two checkpoints), Experiments (stored results per suite with CI,
cross-play matrix, failed-evaluation/compatibility errors, replay links), Infrastructure (jobs, heartbeat age, stale, policy versions,
cleanup dry run → delete), Artifacts (checkpoint lineage + hashes, per-attempt manifest state). Loading, empty, stale ("disconnected,
showing last known data") and partial-artifact states are explicit text; nothing animates a counter.

**Known gaps.** The current trainer does not write `metrics.jsonl`/`episodes.jsonl` (requests/agent-D.md), so a real local run shows
checkpoints and a result but "No metrics have been written". No real evaluator is wired. Fake-backend e2e proves the workflow, not training.
