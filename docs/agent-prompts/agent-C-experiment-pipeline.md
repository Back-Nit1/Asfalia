# Agent C — complete the training, transfer, adaptation, and comparison pipeline

You own the experiment semantics needed to move from school checkpoints to defensible shared-world results. Do not
stop after adding helper classes: ship runnable, resumable commands/modules and bounded real-environment smoke tests.

Read first: `CLAUDE.md`, `README.md`, `prompt (2).md` sections 8–12, `docs/contracts.md`,
`docs/training_infra.md`, `src/hospital_rl/training/`, `src/hospital_rl/evaluation/`, and current training tests.

## Ownership

- `src/hospital_rl/training/`
- `src/hospital_rl/evaluation/transfer.py`
- New experiment configuration files under `configs/experiments/`
- Focused tests under `tests/training/` and `tests/evaluation/`
- `docs/requests/agent-C.md` for changes needed in `cli.py`, API DTOs, shared schemas, or other lead-owned files

Do not edit `src/hospital_rl/cli.py`, frontend/API files, infrastructure backends, or shared engine/schema files.

## Deliverables

1. Implement Stage E adaptation as a real module with a `main(argv)` entry point. Require explicit trainable role,
   attacker checkpoint, defender checkpoint, shared-world config, step budget, seed, output directory, and whether the
   historical league is enabled. Never silently select a checkpoint from `runs/`.
2. Adapt exactly one role while the other is frozen. Load both through compatibility checks; freeze the opponent's
   model and normalization state. Preserve episode-boundary state reset and validation-only checkpoint selection.
3. Integrate the existing bounded `League` into training. Persist a versioned league manifest containing scripted and
   historical checkpoint IDs, train/evaluation pool membership, eviction history, and selection provenance. The
   evaluation pool must never be sampled for updates or used for checkpoint selection.
4. Add alternating adaptation only after the one-side path works. Alternate at explicit round boundaries, checkpoint
   both sides, and prevent stale/duplicate rollout consumption. Keep the first smoke configuration one-side only.
5. Strengthen Stage D evaluation: honor the complete shared-world configuration rather than only `size`; provide clear
   errors for missing files and every checkpoint incompatibility; record config/split/catalog hashes, checkpoint IDs,
   seeds, code revision, and matchup provenance.
6. Produce cross-play reports from real scripted and learned actors, not a toy callback. Emit machine-readable JSON
   with all per-pair metrics and a labeled matrix for at least exposure and availability violation.
7. Add explicit experiment configurations/runners for:
   - pretrained schools + shared-world adaptation;
   - scratch initialization + the same shared-world adaptation budget;
   - equal-total-environment-step comparison including pretraining;
   - frozen Stage D transfer before adaptation.
   Reject incomparable runs rather than averaging them.
8. Add a bounded real hospital smoke test that trains both roles for a tiny budget, performs frozen transfer, performs
   one adaptation round, and writes reports. It may be marked slow, but it must use the actual hospital environments,
   not `tests/training/toy_env.py`.
9. Make resume lineage explicit for school and adaptation runs. Never overwrite an incompatible or unrelated run.

## Acceptance

- Focused unit tests cover frozen-opponent correctness, league persistence/disjointness, one-side adaptation, resume,
  stale rollout rejection, complete config resolution, missing/incompatible checkpoints, cross-play, and budget
  accounting.
- A short CPU smoke command creates attacker and defender checkpoints, a Stage D report, an adapted checkpoint, and a
  comparison manifest under a temporary/output directory.
- No generated checkpoints are committed.
- `PYTHONPATH=src .venv/bin/pytest -q tests/training tests/evaluation` passes.
- Submit an exact CLI wiring request for `train`, `evaluate`, and `adapt`, including full help text and exit semantics.

## Non-goals

- Do not claim policy superiority from smoke runs.
- Do not implement the dashboard, Modal/Daytona, GRU, or synthetic database.
- Do not tune on the test split.
