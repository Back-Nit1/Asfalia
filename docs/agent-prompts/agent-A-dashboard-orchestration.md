# Agent A — Dashboard, API, and orchestration

Read `prompt (2).md` in the repo root first (sections 14, 15, 18 "vertical slices", and the repo layout). You are a specialist coding agent. The lead owns the world engine, shared schemas, `pyproject.toml`, lockfiles, and the CLI skeleton. You build the development-control-plane and the operational dashboard.

## Setup
- The lead will create a git repo with tag `base-v0` containing: `pyproject.toml`, `src/hospital_rl/__init__.py`, `src/hospital_rl/events.py` (EventEnvelope + payload DTOs), `src/hospital_rl/cli.py` (typer/argparse skeleton with stub subcommands `dashboard`, `orchestrate`).
- Work in your own worktree on branch `agent/dashboard-orchestration`, branched from `base-v0`. Do not start until that tag exists. Never commit to `main`. Commit often; the lead integrates.

## Owned paths (write only here)
- `src/hospital_rl/api/**`
- `src/hospital_rl/orchestration/**`
- `frontend/**`
- `configs/orchestrator.yaml`
- `tests/api/**`, `tests/orchestration/**`
- `docs/dashboard.md`

Do NOT edit `pyproject.toml`, `events.py`, `cli.py`, or anything else. If you need a dependency or a schema/CLI change, append a request to `docs/requests/agent-A.md` (what, why, exact proposed diff). Work around it meanwhile (e.g. optional import, local pip install in your venv).

## Contract you code against
`EventEnvelope` (pydantic v2): `event_id: str(uuid)`, `source_kind: Literal["development","experiment","world","infrastructure"]`, `event_type: str`, `run_id|task_id|episode_id: str|None`, `worker_id: str`, `sequence: int` (per source+worker, monotonic), `timestamp: datetime(UTC)`, `tick: int|None`, `visibility: Literal["attacker","defender","evaluator","operator"]`, `schema_version: int`, `payload: dict`. If `events.py` differs from this when you start, `events.py` wins.

## Deliverables
1. **Storage**: SQLite (WAL, single writer process) for tasks, runs, events, artifacts index, idempotency keys. Dedupe by `event_id`, order per `(source, worker)`, bounded retention with snapshot resync, cursor-based reads.
2. **Task orchestrator** (`orchestration/{tasks,contracts,scheduler,adapters,events,integration}.py`): Task record + state machine exactly as in section 14 (queued→ready→running→review→passed→integrated; failed/blocked/cancelled), dependencies, leases, heartbeats, retry cap 2, max concurrency (default 3), timeouts, cancellation, one-owner-per-path enforcement, git-worktree-per-task helper. `CodingAgentAdapter` ABC (submit, session id, event stream, cancel, result, capabilities) plus a `ClaudeCodeCliAdapter` using the documented `claude -p ... --output-format stream-json` CLI if the binary exists; otherwise report "coding runtime not configured". A `FakeAdapter` lives only under `tests/` and any demo using it must be visibly labeled. Store evidence: changed files, test results, redacted stdout/stderr. Never store chain-of-thought. A task is `passed` only when its acceptance commands were actually run by the orchestrator, not on the agent's word. Fixed task DAG loaded from `configs/orchestrator.yaml`.
3. **FastAPI** (`api/`): REST per section 15 "minimum APIs" (tasks, runs, checkpoints, episodes snapshot/events/replay, metrics, infrastructure, artifacts), SSE with cursors and `Last-Event-ID` resync, idempotency keys on mutations, localhost bind by default, token auth for non-local binds. Separate DTOs and endpoints for attacker, defender, and evaluator perspectives; evaluator endpoints require a distinct privileged token and are labeled privileged in the payload. Filter by `visibility` server-side. The development-task stream must be unreachable from any policy-facing endpoint. Runs and infrastructure endpoints return honest "not configured / no data" states until real producers exist.
4. **Frontend** (`frontend/`): React + TypeScript + Vite, React Flow (check current docs for the API before coding), routes `/overview /build /training /world/:episode_id /experiments /infrastructure /artifacts` per section 15. Dark restrained research-console style, accessible contrast, keyboard operable, state conveyed by labels not only color. Build `/build` and `/world/:episode_id` fully (task DAG board with heartbeat/evidence/diff view; topology grouped by zone, perspective switch hitting different endpoints, action feed, metric cards, LIVE/REPLAY/DEBUG marker, replay play/pause/step/scrub). Other routes can be thin but must render honest empty/disconnected states. Update throttling 2–5 Hz, metrics ~1 Hz, configurable. SSE reconnect and page reload must not lose state.
5. **Fixtures**: until the lead's real episode export exists, use a clearly named `fixtures/` episode JSON labeled FIXTURE in the UI. Keep the replay loader format-agnostic behind one adapter so swapping to the real format is a one-file change.

## Acceptance (run and paste outputs in your final report)
- `pytest tests/api tests/orchestration` green. Tests must cover: event dedupe and ordering, SSE reconnect with cursor, idempotent mutations, perspective filtering (attacker DTO contains no evaluator fields), evaluator endpoint rejects the operator token, task state machine illegal transitions, lease expiry, retry cap, path-ownership conflicts, cancellation.
- `cd frontend && npm run typecheck && npm run build` green. Add at least one Playwright (or equivalent) workflow check: open `/build`, open `/world/:id`, switch perspective, reload.
- No fabricated live counters. Anything not wired shows as not configured.

## Rules
- Match surrounding style; no speculative abstractions; no extra dependencies you can avoid.
- Keep provider keys out of source, prompts, and browser responses.
- Final report: branch name, commit list, files changed, commands run with results, what is fixture vs real, open requests to the lead.
