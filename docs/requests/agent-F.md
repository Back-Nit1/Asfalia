# Requests from Agent F (evaluation / data / reporting) to the lead

> **HARD RULE: commit at least every 4 minutes** (see CLAUDE.md).

Agent C's experiment formats (`comparison_manifest.json`, `arms/*/adapt_manifest.json`, `stage_d/*.json`, `ExperimentConfig`)
are the active contract; nothing below changes them. Everything I own is already merged on my branch
(`agent/evaluation-data-reporting`, based on `agent/experiment-pipeline`).

## Engine / env / schema changes needed to *run* the blocked ablations (all definitions + refusals already exist)
`evaluation/ablations.py` raises `Unavailable` for these until the lead adds support; `evaluation/strata.py::ENGINE_SUPPORT`
marks the matching stratum values non-runnable.

1. **`no_workflow_cost`** — `engine/rewards.py`: reward-weight switch that zeroes the defender `critical`/`noncritical`/`latency`
   terms (metrics must still be computed). Suggested: `ScenarioSpec.reward_weights: dict[str, float] = {}` (default 1.0 per
   term; part of `ScenarioSpec.hash()`).
2. **`no_domain_randomization`** — `training/world.py::world_penv` (or `envs/factories`): option `randomize: bool = True` on `WorldConfig`
   (hashed). `False` = one fixed (family, workload, seed) draw per split.
3. **`full_observability_diagnostic`** — `envs/encoding.py::encode_obs`: `visibility="full_diagnostic"` mode for the defender that fills
   node features from truth (compromised/implant/staged). Must change `obs_schema_hash` and `Stratum.visibility_mode`; checkpoints
   trained in this mode must be refused by `load_frozen_agent` outside diagnostics.
4. **telemetry-loss levels** — `engine/telemetry.py::release(..., drop=0.1)` hardcodes the drop probability; `transitions.py:107` calls it
   without a spec value. Add `ScenarioSpec.telemetry_drop: float = 0.1` (hashed) and pass it through; then `generate(...)` /
   `world_penv` need a `telemetry_drop` parameter (`WorldConfig.telemetry_drop`). Levels used: 0.0, 0.1 (current), 0.3, 0.5.
5. **attack schedule** — `Stratum.attack_schedule` is a label (`scripted_v1`, `learned_frozen`); no engine change needed.

## Enforcing the test-stratum rule where training code builds environments
`evaluation/strata.py::require_trainable(split, use)` raises `StratumError` for `test`. Please call it in:
- `envs/curricula.py::build_env(cfg, level, split)` and `envs/factories.hospital_penv` (training/curriculum),
- `training/league.py` when choosing opponents/eval pools (use="opponent selection"),
- `training/adapt.py` / `schools.py` for checkpoint selection (use="checkpoint selection").
Also call `strata.check_training_config(cfg_dict)` on every loaded school/experiment YAML. Test: `tests/evaluation/test_strata.py`
(it already scans every checked-in YAML under `configs/`). Splits are still defined in `evaluation/splits.py::SPLITS`; the
`benchmark_hash()` in `strata.py` covers every dimension and is written into every trajectory row/manifest.

## CLI slots (`cli.py`, add to the delegated list with `add_help=False`, as in Agent C's request)
| command | module | exit codes |
|---|---|---|
| `hospital-rl export-trajectories` | `hospital_rl.data.trajectory` (`export` subcommand) | 0 ok, 2 bad input/invalid trajectory |
| `hospital-rl validate-trajectories DIR` | `hospital_rl.data.trajectory` (`validate`) | 0 ok, 2 invalid |
| `hospital-rl report --runs DIR... --out DIR` | `hospital_rl.reporting.report` | 0 ok, 2 incompatible/missing input |
| `hospital-rl ablations list|write|check` | `hospital_rl.evaluation.ablations` | 0 ok, 2 invalid/unavailable |

## Runner / metrics (`runner.py::summarize`, `engine/state.py::Metrics`)
My `evaluation/metrics.py` derives recovery, containment censoring, per-workflow failures and constraint violations from the episode
`steps` events, so no runner change is required. Optional improvements that remove approximations:
- `Metrics.wf_total_by_id` / `wf_failed_by_id` (per-workflow denominators; today I report failure-event counts only).
- Record per-tick `service_up` counts so recovery can be defined per node rather than per critical workflow failure.
- `summarize` could include `first_critical_failure_tick`; today recovery is computed from `workflow_failed` events.

## `docs/contracts.md` (lead-owned)
Replace the "Episode record" paragraph: trajectories are now written by `data.trajectory.TrajectoryWriter` (callable sink for
`evaluation.transfer.play/evaluate_pair(..., sink=)`), trajectory version 2 (see `docs/evaluation.md`). `run_episode(record=True)` rows are
raw role views for quick debugging only and are no longer the export format.

## `pyproject.toml`
Nothing new from F (the `slow` marker request from C still stands). Report/plots use only the stdlib (hand-written SVG).

## Small edit I made in Agent C's file (flagging for review)
`evaluation/transfer.py`: `play(..., sink=None)` and `evaluate_pair(..., sink=None)`. With `sink=None` behavior is unchanged; the sink
is the only wiring point for trajectory export. If you prefer, revert it and I will wrap `play` externally.
