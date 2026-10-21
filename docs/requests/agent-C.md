# Requests from Agent C (experiment pipeline) to the lead

> **HARD RULE: commit at least every 4 minutes** (see CLAUDE.md).

## 1. CLI wiring (`cli.py`) — `train`, `evaluate`, `adapt`
`_DELEGATE` already maps all three to my modules and returns `main(rest) or 0`, so **exit codes already propagate**.
One defect remains in `cli.py`: the stub subparsers (`sub.add_parser(name)`) swallow `-h/--help`, so
`hospital-rl adapt --help` prints `usage: hospital-rl adapt [-h]` instead of the real help. Fix (exact):

```diff
-    for name in ("train", "evaluate", "adapt", "orchestrate", "dashboard", "cloud-plan", "cleanup",
-                 "export-policy", "benchmark-inference"):
-        sub.add_parser(name)
+    for name in ("train", "evaluate", "adapt", "orchestrate", "dashboard", "cloud-plan", "cleanup",
+                 "export-policy", "benchmark-inference"):
+        sub.add_parser(name, add_help=False)   # delegated commands print their own help
```
With `add_help=False`, `-h` falls into `parse_known_args`'s `rest` and reaches the delegate's argparse, which prints the
help below and exits 0.

Exit semantics (all three): **0** finished / report written; **2** unusable input (missing, unreadable or incompatible
checkpoint — message lists every mismatching field; invalid or missing shared-world config; `--out` already belongs to a
different run [lineage conflict]); argparse usage errors are also 2. No other codes are used.

### `hospital-rl adapt --help` (module `hospital_rl.training.adapt`, `main(argv)`)
```
usage: hospital-rl adapt [-h] --trainable-role {attacker,defender}
                         --attacker-checkpoint ATTACKER_CHECKPOINT
                         --defender-checkpoint DEFENDER_CHECKPOINT --world-config WORLD_CONFIG
                         --steps STEPS --seed SEED --out OUT --league {on,off}
                         [--init {pretrained,scratch}] [--league-checkpoint LEAGUE_CHECKPOINT]
                         [--eval-checkpoint EVAL_CHECKPOINT] [--pretrain-steps PRETRAIN_STEPS]
                         [--rounds ROUNDS] [--eval-every EVAL_EVERY] [--val-episodes VAL_EPISODES]
                         [--n-envs N_ENVS] [--lr LR] [--ent-coef ENT_COEF]

Stage E: adapt exactly one role in the shared hospital world while the other role is frozen.
Nothing is selected from runs/; every checkpoint is explicit. Exit 0: finished; exit 2:
missing/incompatible checkpoint, invalid world config, or the --out dir belongs to a different
run.
```
(options as printed by `python -m hospital_rl.training.adapt --help`; `--rounds N` switches to alternating adaptation,
`--steps` is then steps per phase.)

### `hospital-rl evaluate --help` (module `hospital_rl.evaluation.transfer`)
```
usage: hospital-rl evaluate [-h] [--config CONFIG] --attacker-checkpoint ATTACKER_CHECKPOINT
                            --defender-checkpoint DEFENDER_CHECKPOINT
                            [--extra-attacker-checkpoint EXTRA_ATTACKER_CHECKPOINT]
                            [--extra-defender-checkpoint EXTRA_DEFENDER_CHECKPOINT]
                            [--split {train,validation,test}] [--episodes EPISODES] [--out OUT]
```
`--config` now resolves every shared-world field (size, horizon, difficulty, base_logging); unknown keys are errors.
`--split test` is allowed for a final frozen report only; nothing in my runners selects or tunes on it.

### `hospital-rl train --help` (module `hospital_rl.training.schools`)
Unchanged flags. New behaviour: `--out` is bound to (role, config text, seed, n_envs, rollout_len) via `lineage.json`;
re-running resumes, a mismatch exits 2. Pre-lineage dirs (e.g. `runs/attacker`) are adopted only if `latest.pt` has the
same role and seed.

### Experiment runner (no CLI slot requested; optional `hospital-rl experiment` later)
`python -m hospital_rl.training.experiments run --config configs/experiments/<name>.yaml --out DIR`
`python -m hospital_rl.training.experiments compare --rule equal_adapt|equal_total RUN_DIR RUN_DIR ...`

## 2. `envs/factories.py::scenario_for` / `hospital_penv` ignore `base_logging`
`configs/hospital40.yaml` has `base_logging`, but `scenario_for` has no such parameter, so the old Stage D silently
dropped it. I resolve the full config in `training/world.py::world_penv` (duplicating ~4 lines of family/workload
selection). Request: add `base_logging: int = 0` to `evaluation/splits.py::scenario_for` and `envs/factories.hospital_penv`,
then `world_penv` can delegate and the duplication goes away.

## 3. Register the `slow` pytest marker (`pyproject.toml`)
`test_real_hospital_smoke_pipeline` is `@pytest.mark.slow` (≈10 s today). Add
`markers = ["slow: real-environment smoke tests"]` under `[tool.pytest.ini_options]` to silence the unknown-mark warning.

## 4. Nothing else needed
No `events.py`/`schemas.py`/engine changes. `training/train.py::TrainConfig` gained `init_checkpoint` (mine).
