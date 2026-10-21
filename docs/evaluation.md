# Evaluation, data and reporting (Agent F)

> **HARD RULE: commit at least every 4 minutes** (see CLAUDE.md).

| need | entry point |
|---|---|
| benchmark strata + hash, test-stratum guard | `hospital_rl.evaluation.strata` (`benchmark_hash`, `require_trainable`, `check_training_config`) |
| objective metrics (censored recovery/containment, attacker success, downtime, benign false positives, constraint violations) | `hospital_rl.evaluation.metrics`; `evaluation.benchmark.run` rows now carry `objectives` |
| trajectory export from real evaluation | `python -m hospital_rl.data.trajectory export --attacker-checkpoint A --defender-checkpoint D --out DIR` (or `evaluate_pair(..., sink=TrajectoryWriter(...))`) |
| trajectory validation | `python -m hospital_rl.data.trajectory validate DIR` |
| ablations | `python -m hospital_rl.evaluation.ablations list|write|check` (configs in `configs/experiments/ablations/`) |
| report from stored experiment dirs | `python -m hospital_rl.reporting.report --runs RUN_DIR... --out OUT` |
| evidence regeneration | `PYTHONPATH=src python scripts/eval_evidence.py` (writes `docs/img/report_fixture/`, `docs/img/trajectory_validation.json`) |
| MITRE labels, limits | `docs/mitre_mapping.md`, `docs/related_systems_and_limits.md` |

## Trajectory format v2
Per episode: `<id>.jsonl` (actor rows), `<id>.privileged.json` (evaluator truth, same `(role, step)` keys), `<id>.manifest.json`
(sha256 of both, policy/normalizer identity, scenario/config/split/benchmark/catalog hashes). Each actor row holds exactly the
encoded observation the policy received (`node_features f32`, `node_mask i8`, `adjacency i8`, `history f32`, `globals f32`), the
encoded `action_mask`, `action` index, `reward`, `reward_components`, `terminated`/`truncated`, `policy_id`, `checkpoint_id`,
`normalizer_version`/`normalizer_hash`, `observation_schema_hash`, `rollout_id`, `episode_id`, `seed`. The normalizer is applied inside
the policy, so rows hold the *raw* encoded input plus the hash of the frozen normalizer statistics. Validation is a whitelist: exact key
sets, shapes, dtypes, binary masks, action legality, padding, ordering, header constancy, manifest/file hashes, privileged cross-keys.

## Report
Input = Agent C run directories (one per seed). Mixed hashes/code revisions are refused; absent arms are listed as MISSING, never
imputed. Fewer than 5 seeds is labelled INSUFFICIENT (one seed: no spread or CI exists and is shown as `n/a`). Output has no
timestamps or absolute paths, so it is byte-reproducible. The checked fixtures in `tests/fixtures/experiments/` are real 2-seed smoke
outputs (128-step budgets, scrubbed of absolute paths), not evidence about policy quality.
