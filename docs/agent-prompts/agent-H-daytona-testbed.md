# Agent H — implement Daytona lifecycle and the optional isolated synthetic-service testbed

This is a post-core task. Implement it only after local/Modal graph training, transfer, cleanup, and artifact contracts
are stable. Use current official Daytona documentation and pin the SDK/API version used.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` sections 11 and 17, infrastructure backend/artifact contracts,
`docs/training_infra.md`, `configs/cloud_daytona.yaml`, the newer direct Daytona path in
`src/hospital_rl/infra/cloud_school.py`, and Agent E's completed lifecycle work. Treat the direct launcher as code to
audit and integrate, not proof that the `DaytonaBackend` contract is complete.

## Ownership

- `src/hospital_rl/infra/daytona_backend.py`
- New Daytona adapter/reconciler modules under `src/hospital_rl/infra/`
- Daytona-specific portions of `src/hospital_rl/infra/cloud_school.py`, preferably reduced to a thin caller of the
  common backend rather than a second lifecycle implementation
- `configs/cloud_daytona.yaml`
- Daytona tests under `tests/infra/`
- New isolated synthetic-testbed package, configs, tests, and documentation
- `docs/requests/agent-H.md` for CLI/API/dependency/shared contract changes

Do not edit PPO/engine semantics, API/frontend, shared CLI, or Modal/local backend behavior.

## Deliverables

1. Refactor or replace the direct Daytona school launcher so there is one lifecycle implementation. Remove any code
   that reads `.env`; credentials may be consumed only through the provider SDK/process environment and their values
   must never be printed. Replace shell-interpolated user inputs with validated typed arguments or safe invocation.
2. Implement the complete `JobBackend` lifecycle with pinned snapshot/image, run/task/attempt ownership labels, explicit
   caps, durable provider IDs, status, drain, checkpoint-and-stop, force cancel, bounded retries, resume lineage, TTL,
   idle stop, and orphan reconciliation.
3. Implement a bounded coding-workspace smoke: start at a pinned commit, install controlled dependencies, run a real
   test, return commit/artifact evidence, verify upload, and clean up. Do not expose host/provider credentials.
4. Implement an optional graph rollout smoke using the same resolved job/rollout contract as local/Modal. Transfer
   immutable compressed artifacts through the trusted configured transport; verify hashes and timeouts. Avoid per-step
   cross-cloud RPC.
5. Add cleanup dry-run and ownership validation. Never delete before required artifacts are verified durable.
6. Build the optional synthetic-service bundle locally first: invented PostgreSQL records, mock identity, imaging
   metadata, benign workload generators, trusted observability channel, bounded resources, internal-only network, and
   repeatable reset. Policies still emit only allowlisted typed actions—no generated SQL or shell.
7. Test graph-to-testbed adapter semantics and transition mismatch without claiming real-hospital validity. Keep audit
   storage separate from actor-accessible records and deny attacker-policy egress/provider credentials.
8. Add a capped Daytona smoke script gated by explicit opt-in and credentials. Never read/print secret values; absence
   must produce setup instructions and a working local fallback.

## Acceptance

- Fake-provider tests cover lifecycle, retry, timeout, partial upload, restart recovery, orphan detection, ownership,
  and cleanup.
- Local synthetic-testbed tests prove isolation, deterministic seed/reset, bounded data, audit separation, and typed-
  action-only access.
- A clean dry-run resolves resources and estimates without contacting Daytona.
- Run one credentialed capped smoke only when explicitly authorized; otherwise mark it unexecuted.
- No real or realistic patient identifiers are used or committed.

## Non-goals

- Do not place every graph node in a separate sandbox.
- Do not make Daytona or the database testbed a prerequisite for the core graph benchmark release.
- Do not claim HIPAA compliance or real-world remediation effectiveness.
