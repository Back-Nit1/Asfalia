# Contracts (owned by the lead)

> **HARD RULE: commit at least every 4 minutes** (see CLAUDE.md).

Merge or rebase `main` to get the real implementations; everything below exists on `main`.

## Environment (`hospital_rl.envs`)
- `HospitalParallelEnv(spec_or_fn)` — PettingZoo ParallelEnv, agents `attacker`, `defender`; episodes truncate at the horizon.
- Observation (per role): `node_features (64,F) f32` (F=15 attacker, 17 defender), `node_mask (64,) int8`,
  `adjacency (64,64) int8`, `history (8,16) f32`, `globals (6,) f32`, `action_mask (N,) int8`.
- Actions: `Discrete(N)`, N = 1 + n_types*64 (attacker 449, defender 705). Index 0 = wait (always legal).
  `idx-1 = type*64 + slot`. Slots are permuted per episode. Illegal/padding/out-of-range indices raise `IllegalAction`.
- `env.metadata["catalog_hash"]`, `["observation_schema_hash"][role]`, `["tie_policy"]`.
- `info[role]["reward_components"]`. Evaluator truth is only on `env.world.truth` (never in obs/info).
- `SingleRoleEnv(penv, role, opponent)`; opponent = `callable(penv, opp_role) -> catalog index` with optional `.reset()`.
  A learned frozen opponent reads its own role's obs from `penv` (see `ScriptedOpponent`).
- `envs.curricula.build_env(cfg, level, split)` builds a school env from `configs/{attacker,defender}_school.yaml`.

## Engine semantics
Tick: defender action charged + queued as job; resolution order set by `tie_policy` (default defender_first);
workflows; recurring costs (hard budget cap; unfunded recurring controls lapse); telemetry; tick += 1.
Rewards are uncalibrated (`engine/rewards.py`). Defender reward terms are all <= 0.

## Episode record
`runner.run_episode(..., record=True)` remains a compact debugging record. The evaluation data contract is trajectory
version 2, written by `data.trajectory.TrajectoryWriter` (also accepted as the `sink` for
`evaluation.transfer.play/evaluate_pair`). Each episode has actor-visible JSONL, a separate evaluator-privileged JSON
file, and a hash manifest covering policy/normalizer/config/split identities. See `docs/evaluation.md`.
