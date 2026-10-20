# Agent B — RL training infrastructure, checkpoints, cloud backends, inference export

Read `prompt (2).md` in the repo root first (sections 3, 8, 16, 17, 18, and the Gate 3/4 text). You are a specialist coding agent. The lead owns the world engine, environments, shared schemas, `pyproject.toml`, lockfiles, and the CLI skeleton. You build everything that trains, stores, ships, and exports policies. You must not depend on the real hospital engine existing yet.

## Setup
- The lead will create a git repo with tag `base-v0` containing: `pyproject.toml` (torch, gymnasium, numpy, pydantic, pyyaml, pytest), `src/hospital_rl/__init__.py`, `src/hospital_rl/events.py` (EventEnvelope), `src/hospital_rl/cli.py` skeleton.
- Work in your own worktree on branch `agent/training-infra`, branched from `base-v0`. Do not start until that tag exists. Never commit to `main`. Commit often.

## Owned paths (write only here)
- `src/hospital_rl/policies/{base,ppo,checkpoint}.py`
- `src/hospital_rl/training/{train,league,curriculum}.py`
- `src/hospital_rl/infra/**`
- `native/**`, `benchmarks/**`
- `configs/{cloud_local,cloud_modal,cloud_daytona}.yaml`
- `tests/training/**`, `tests/infra/**`
- `docs/training_infra.md`

Do NOT edit `pyproject.toml`, `events.py`, `cli.py`, the engine, or envs. For deps, schema, or CLI changes, append to `docs/requests/agent-B.md` (what, why, exact diff). Work around meanwhile.

## Contract you code against (lead will supply the real env; build against a toy env with the same interface)
Single-role training env (Gymnasium): `action_space = Discrete(N)` where N = 1 + n_action_types * max_nodes (max_nodes 64, configurable), wait is index 0. Observation is a `Dict` with: `node_features (max_nodes, F) float32`, `node_mask (max_nodes,) bool`, `adjacency (max_nodes, max_nodes) bool`, `history (H, E) float32`, `globals (G,) float32`, `action_mask (N,) bool`. Env `info` carries `reward_components: dict[str,float]`. Env exposes `metadata["catalog_hash"]`, `["observation_schema_hash"]`, `["role"]`. Ship `tests/training/toy_env.py` (small masked-Discrete env with padded graph obs, deterministic seed) and use it for all tests. Illegal actions must be impossible to sample (mask applied to logits, `-inf`/large negative, wait always legal).

## Deliverables
1. **Masked PPO** (`policies/ppo.py`, `training/train.py`): either a verified maintained library that supports Dict obs + action masks (document and test the choice) or a minimal clean PPO (GAE, clipped objective, value loss, entropy, minibatches, grad clip). One implementation for both roles. Policy net: masked node encoder with pooled context (mean/max over valid nodes only; padding must not leak), feed-forward head; recurrent-state spec in the interface (`initial_state`, reset at episode boundary) even if GRU is added later. Vectorized envs in one process (sync). Records behavior log-probs and values per rollout; rollouts are versioned (`policy_version`) and on-policy, with stale/duplicate batch rejection.
2. **Checkpoints** (`policies/checkpoint.py`): atomic write (tmp+rename), interrupt-safe (SIGINT/SIGTERM triggers a final checkpoint), resume that restores model, optimizer, normalizer, RNG states (torch/numpy/python), step counter, rollout/policy version. Metadata JSON: role, schema versions, `catalog_hash`, `observation_schema_hash`, normalizer stats + frozen flag, max node shape, config hash, code revision if available, library versions, training steps, parent checkpoint id (lineage), seed. `load_for_role(path, expected_role, expected_catalog_hash, expected_obs_hash)` raises a readable `IncompatibleCheckpointError` naming exactly which field mismatched. Normalizer fits only during training and is frozen at evaluation.
3. **Frozen-opponent adapter**: wrap a two-role parallel env plus a frozen checkpoint into the single-role env above (the lead's real envs will plug in via a small protocol you define in `policies/base.py`; document it). Test it with the toy env.
4. **League** (`training/league.py`): bounded opponent pool (scripted + historical checkpoints), sampling distribution, separate evaluation pool never used for updates or selection, cross-play matrix computation over a callable `evaluate(attacker_id, defender_id) -> metrics`.
5. **Compute backends** (`infra/`): provider-neutral `JobBackend` and `ArtifactStore` ABCs; `LocalBackend` (subprocess, resource caps, drain / checkpoint-and-stop / force-cancel as three distinct operations); `modal_app.py` + `ModalBackend` (first mode: one bounded job, vectorized envs + learner together; Volume artifacts with unique run/attempt paths, atomic publish, manifest hashes, commit/reload per current docs). **Verify the current Modal SDK API with the docs (context7 or web) before coding.** `DaytonaBackend` as an optional stub implementing the same interface with the same rollout contract; mark untested paths DEFERRED in docs. `cloud-plan` dry-run function returning resolved config, resource caps, estimated resource-hours with provenance; estimates are labeled estimates. Missing credentials give setup instructions and fall back to local, never a fake success. Secret names documented, values never read into code or logs. Everything is tested without network via fakes; real Modal smoke run only if credentials exist and with an explicit tiny cap.
6. **Inference export**: `InferenceBackend` ABC with `PythonReferenceBackend` (batch, masks, normalization, optional recurrent state in, logits/value/new state out; define dtype/shape/contiguity semantics in the docstring). ONNX export of the feed-forward policy with metadata (schema/catalog hashes, normalizer, max shape, checkpoint id) and a parity test (logits within a documented tolerance, argmax equal except documented ties) using onnxruntime Python. Write `native/CMakeLists.txt` + `policy_runner.cpp` + pybind11 `bindings.cpp` against ONNX Runtime C++ only if the toolchain is present; verify the API in current docs. Otherwise leave a documented skeleton and state the exact missing prerequisite. Do not claim native execution that did not happen. `benchmarks/inference` script comparing backends (p50/p95, throughput, load time) on identical inputs.

## Acceptance (paste outputs in your final report)
- `pytest tests/training tests/infra` green. Tests must cover: masked logits never pick illegal actions across thousands of random states; padding invariance of the encoder (permuting/adding padded nodes does not change output); weights actually change after a train run; resume reproduces the uninterrupted run's next checkpoint bit-for-bit given same seed (or document the exact nondeterminism); incompatible checkpoint error message names the field; stale rollout rejected; ONNX parity; atomic checkpoint survives a kill mid-write.
- A tiny CPU smoke run on the toy env finishes in under a minute and shows a learning signal on a toy task designed to be learnable; report the real numbers, label as one-seed smoke, no superiority claims.
- Backend dry-run prints a plan without needing credentials.

## Rules
- Match surrounding style; no speculative abstractions; do not build a custom RL algorithm as a goal.
- Final report: branch, commit list, files changed, commands run with results, what is tested vs deferred, open requests to the lead.
