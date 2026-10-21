# Remaining-work execution map

Use the prompts in this directory to finish the repository from the current `main` branch. Each agent must first read
`CLAUDE.md`, `README.md`, `prompt (2).md`, and the files named in its prompt. The repository hard rule applies: commit
buildable or clearly labeled WIP at least every four minutes of active work and before long-running commands. Never
read or print `.env` or credential values.

## Rules shared by every agent

- Inspect the current implementation before editing. Preserve working behavior and unrelated user changes.
- Implement working vertical slices, not new placeholders. A component is not complete until its real caller uses it.
- Keep learned policies restricted to typed simulator actions. Never add shell, SQL, network, provider credentials, or
  evaluator truth to actor inputs.
- Use synthetic data only. Do not make clinical, security-effectiveness, novelty, or superiority claims from smoke runs.
- Run focused tests while developing and report exact commands and results. Mark credentialed or platform-specific
  paths honestly when they cannot be exercised.
- Do not edit files outside the prompt's ownership list. If a shared file must change, write the exact requested diff
  to `docs/requests/<agent-name>.md` for the lead-integration agent.
- Finish with: branch, commits, files changed, commands/results, artifacts produced, remaining risks, and deferred work.

## Order and dependencies

### Wave 1 — may run in parallel

1. `agent-C-experiment-pipeline.md`
2. `agent-D-control-plane-dashboard.md`
3. `agent-E-cloud-lifecycle.md`

Agent D should define its controller behind a small interface so it can initially use local jobs and later accept the
backend wiring from Agent E. Agent C owns experiment semantics; Agent D owns HTTP/UI presentation; Agent E owns compute
lifecycle.

### Wave 2 — after Agent C contracts are stable

4. `agent-F-evaluation-data-reporting.md`
5. `agent-G-native-inference.md`

These agents may run in parallel. Agent F consumes checkpoint/evaluation contracts from Agent C. Agent G needs at
least one compatible real checkpoint but must not commit generated model binaries.

### Wave 3 — after local and Modal paths are coherent

6. `agent-H-daytona-testbed.md`

Daytona and the synthetic-service testbed must not block the graph simulator's local end-to-end release.

### Final serialized pass

7. `agent-I-lead-integration-release.md`

This prompt owns shared CLI/config/task-graph/docs integration, resolves specialist requests, and runs the clean
end-to-end acceptance sequence. Do not run the final pass concurrently with specialists.
