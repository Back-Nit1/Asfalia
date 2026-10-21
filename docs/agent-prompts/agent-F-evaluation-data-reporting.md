# Agent F — complete benchmark dimensions, trajectory contracts, metrics, and reports

Close the gaps between the implemented simulator and the evaluation/data acceptance contract. Preserve strict actor vs
evaluator separation and make every result reproducible from recorded provenance.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` sections 5, 6, 9, 10, and 12, `docs/contracts.md`,
`src/hospital_rl/evaluation/`, `src/hospital_rl/data/trajectory.py`, `src/hospital_rl/runner.py`, scenario generation,
and all invariant/data tests.

## Ownership

- `src/hospital_rl/evaluation/` except adaptation mechanics owned by Agent C
- `src/hospital_rl/data/`
- New reporting modules/scripts under `src/hospital_rl/reporting/` or `scripts/`
- `tests/evaluation/` and focused data/invariant tests
- Generated textual/JSON validation summaries under `docs/img/` only when produced by checked-in scripts
- `docs/requests/agent-F.md` for runner, engine metrics, shared schemas, CLI, or dashboard DTO changes

Do not edit engine/env/shared schemas directly, PPO training, API/frontend, or infrastructure.

## Deliverables

1. Version benchmark strata for topology family, workload profile, attack schedule, vulnerability combination/difficulty,
   telemetry-loss level, visibility mode, and seed range. Hash the complete definition and reject test-stratum use in
   training/curriculum/opponent selection.
2. Add missing objective metrics: attacker success, recovery time with censoring, task-level success/degradation,
   visibility-shift degradation, downtime, containment censoring, detection false positives on benign episodes, and
   constraint violations. Do not substitute zero for censored values.
3. Make trajectory rows contain the exact actor input used for the decision, including encoded action mask, schema and
   normalizer versions, policy/checkpoint ID, scenario/config/split hashes, seed, rollout/episode ID, reward, and
   termination/truncation. Keep evaluator state in a separately keyed privileged artifact.
4. Strengthen trajectory validation beyond a short banned-key list: validate against role-specific DTO/schema, shapes,
   dtypes/JSON representations, masks, hashes, row ordering, and cross-file keys. Add tamper/leakage tests.
5. Wire trajectory export into real episode/evaluation workflows through a callable entry point. Do not leave it used
   only by tests.
6. Add runnable ablation definitions for no pretraining, no adaptation, no workflow cost, no domain randomization, no
   opponent league, full-observability diagnostic, and telemetry-loss levels. Expensive runs need not execute now, but
   configs and validation must be real and reject invalid comparisons.
7. Produce a versioned report from stored experiment outputs with per-seed values, mean/spread/CI, compatibility checks,
   cross-play, and security/availability/cost plots. Clearly label one-seed smoke output as insufficient.
8. Add verified MITRE tactic/technique metadata only where the abstraction genuinely corresponds. Cite authoritative
   identifiers in documentation, mark incomplete coverage, and do not imply executable attack fidelity.
9. Add a concise related-systems/limitations note covering CyberBattleSim and the limits of synthetic evidence without
   making novelty claims.

## Acceptance

- Tests prove hidden-state perturbations do not alter actor trajectories/masks absent emitted evidence.
- Tests prove test strata cannot enter training and benchmark hashes change for every stratum change.
- Trajectory round-trip and deliberate privileged-field injection tests pass.
- Report generation is deterministic from checked fixtures and never invents missing runs.
- `PYTHONPATH=src .venv/bin/pytest -q tests/evaluation tests/test_data.py tests/test_invariants.py` passes.
- Submit exact shared metric/runner changes to the lead rather than editing owned engine/schema files.

## Non-goals

- Do not execute expensive multi-seed studies without an explicit budget.
- Do not use passive public traces to claim remediation effectiveness.
- Do not add a privileged critic.
