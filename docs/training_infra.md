# Training infrastructure (Agent B)

Everything here is verified by `pytest tests/training tests/infra` unless marked **DEFERRED** or **UNTESTED**.

## Layout
| Path | What |
|---|---|
| `policies/base.py` | action layout, `mask_logits`, `Policy` ABC (recurrent-state spec), `TwoRoleEnv`/`FrozenActor` protocols, `FrozenOpponentEnv` |
| `policies/ppo.py` | `GraphPolicy`, `Normalizer`, `SyncVecEnv`, `collect_rollout`, `PPO`, `Agent`/`FrozenAgent` |
| `policies/checkpoint.py` | atomic save, resume, `load_for_role`, interrupt flag |
| `training/train.py` | loop, control signals, `evaluate`, `main` |
| `training/league.py`, `curriculum.py` | bounded league + cross-play, validation-gated curriculum |
| `infra/` | `backend` (ABCs), `artifacts`, `job`, `local_backend`, `modal_app`, `daytona_backend`, `plan`, `inference` |
| `native/` | ONNX Runtime C++ runner + pybind11 bindings |
| `benchmarks/inference/run.py` | backend comparison |

## PPO
Own minimal PPO (GAE, clipped objective, value loss, entropy, minibatches, grad clip). No maintained library was
adopted: none supports Dict obs + action masks + the padding-invariance and versioned-rollout requirements without
wrapper code that would be as large as this file. One implementation serves both roles (`role` is data).

- Action layout: `0 = wait`, else `1 + action_type * max_nodes + node`. **Needs lead confirmation** (requests §2).
- Illegal actions: logits set to `-1e9` where `action_mask` is false; wait is forced legal inside `mask_logits`
  (so a malformed row can never produce NaN); the collector additionally asserts the env honours wait-legal.
- Encoder: node MLP -> one neighbour-mean round over `adjacency & valid` -> mean/max pooling over valid nodes ->
  per-node action-type heads + wait head + value head. All masking uses `torch.where`, so even `inf/NaN` in padded
  rows is inert (tested). Weights do not depend on `max_nodes`, so the same policy runs on a larger padded shape.
- Recurrent spec: `Policy.initial_state(B)`, `forward(obs, state)`, `reset_state(state, done)`; the collector calls
  them at episode boundaries. The shipped policy is feed-forward (state `None`); GRU needs sequence minibatching in
  `PPO.update` (not built).
- Rollouts: `Rollout.policy_version` + monotonically increasing `rollout_id`. `PPO.update` raises
  `StaleRolloutError` for a version mismatch or a repeated/older id. Envs are reset at the start of every rollout
  with seeds derived from `(seed, rollout_id, env_index)`, so seeds never repeat and resume needs no env snapshot.
  Cost: episodes longer than `rollout_len` are cut at the rollout boundary (bootstrapped from the value head).
- Time-limit truncation is bootstrapped from the final observation's value; termination is not.
- Normalizer: fits only inside `collect_rollout` while `frozen == False`; `FrozenAgent`/`load_frozen_agent` freeze it,
  and `evaluate()` freezes it for the duration. Stored rollout observations are the normalized ones the behaviour
  policy actually saw.

## Checkpoints
One `.pt` (loaded with `weights_only=True`) + `.pt.meta.json` sidecar. Written tmp -> fsync -> `os.replace` -> fsync dir.
Contents: model, optimizer, normalizer, torch/numpy/python RNG, counters, metadata (role, schema versions, catalog /
observation hashes, normalizer stats + frozen flag, max_nodes, policy_config, config hash, code revision, library
versions, training steps, policy version, last rollout id, parent checkpoint id, seed, checkpoint id).
`load_for_role(path, role, catalog_hash, obs_hash)` raises `IncompatibleCheckpointError` listing every mismatching
field, e.g. `checkpoint ... incompatible - catalog_hash: checkpoint has 'a', expected 'b'`.

Resume: the test trains 4 rounds straight and 2+2 with a restart; model weights, normalizer and torch RNG state are
bit-identical (`torch.equal`). Scope: CPU, single process, no concurrent training threads. Not claimed: GPU, or
different thread counts/machines. (A flaky failure was traced to a test fake that kept training in a background
thread against the shared global RNG; `train()` uses process-global RNG, so run one training per process.)

Interrupts: SIGINT/SIGTERM -> flag -> at the next safe boundary the loop stops and writes a final checkpoint
(second SIGINT hard-stops). Tested with a real subprocess + SIGTERM, with SIGKILL during repeated writes (every
published file loads), and with an injected `os.replace` failure (old checkpoint intact).

## Stop semantics (three distinct operations)
| op | mechanism | effect |
|---|---|---|
| drain | control file `drain` | finish the in-flight round (collect + update), checkpoint, exit -> `DRAINED` |
| checkpoint-and-stop | control file `stop` (or SIGTERM) | in-flight rollout discarded, last completed update checkpointed -> `STOPPED` |
| force-cancel | SIGKILL / `FunctionCall.cancel(terminate_containers=True)` | no checkpoint, recent work lost -> `CANCELLED` |

Control files live in `<root>/<run>/control/<attempt>`, separate from the learner-owned `<root>/<run>/<attempt>/`.
Timeout: at `timeout_s` the local backend requests a stop, then kills after `grace_s` -> `TIMED_OUT` (checkpoint kept).
Retry/restart = new `attempt_id` with `resume_from_attempt`; step counters live in the checkpoint, so nothing is
double counted (tested: 64 + 64 = 128, not 192). Same-attempt retries resume from that attempt's `latest.pt`.

## League
`League`: scripted entries permanent, checkpoint entries bounded FIFO; train pool and eval pool are disjoint
(adding across pools raises); `sample()` only reads the train pool (p_scripted, p_latest, else uniform older).
`cross_play(evaluate, attackers, defenders, metric)` -> matrix + full per-pair metrics. Checkpoint *selection* must
use validation outside the eval pool; the league does not select anything. `Curriculum` advances only on reported
validation success (all of the last `window` results >= threshold).

## Backends
Interfaces: `JobBackend` (`submit/status/drain/checkpoint_and_stop/force_cancel`), `ArtifactStore`
(`put` immutable atomic, hash-verified `get`, `publish_manifest` once, `verify_manifest`).
- **LocalBackend** (tested): subprocess per attempt, caps = wall timeout (+grace kill), `RLIMIT_CPU`
  (`timeout*cpu + grace`), `OMP_NUM_THREADS`, and `RLIMIT_AS` on Linux only. **macOS does not enforce memory**
  (RLIMIT_AS is ignored there); do not rely on `memory_mb` locally on macOS.
- **ModalBackend** (logic tested with a fake gateway; `RealModalGateway` smoke-tested live, see below). First mode: one
  bounded job = vectorized envs + learner. Caps from config are checked before anything launches; submit is idempotent
  per `(run_id, attempt_id)`. Volume at `/runs`, unique `run/attempt` paths, `vol.commit()` after each checkpoint and
  after the manifest, `vol.reload()` before each control poll in the container. Modal API verified against docs and
  the installed SDK (1.6.2): `App.function(serialized, cpu, memory[MiB], timeout, retries, max_containers, volumes,
  secrets)`, `Function.spawn`, `FunctionCall.get/cancel/from_id`, `Volume.commit/reload/batch_upload`, `App.run`.
  Findings: `@app.function` must be at global scope unless `serialized=True` (used, because caps come from config);
  `Volume.reload()` works only inside a running function; a *detached* app left its container alive after the
  function returned, so the gateway holds a non-detached `app.run()` and stops the app when the call ends/cancels.
  Consequence: the client process must stay alive for the job's duration (durable state is on the Volume). Not built:
  second-mode distributed rollout workers, reconciler for orphans, Modal-side retries beyond `modal.Retries`.
- **DaytonaBackend**: **DEFERRED** stub, every method raises `NotImplementedError("DEFERRED ...")`.

### Live Modal smoke (one-seed, tiny)
`python tests/infra/modal_smoke.py` — caps: 1 CPU, 1 GiB, 120 s, 1 container, 0 retries, 2048 steps, toy env.
Result (2026-10-20): state `done`, 2048 env steps, ~20-25 s wall, 4 checkpoints + manifest + result on the Volume,
run directory removed afterwards, app stopped. Not tested live: drain/stop/cancel on Modal, timeout, preemption,
resume across attempts on Modal, quota errors. Two earlier smoke run directories remain in Volume
`hospital-rl-smoke-runs` (`smoke-ab7987a6`, `smoke-fbcef79a`) from the runs before the cleanup fix.

### cloud-plan (dry run, no network, no credentials needed)
`python -m hospital_rl.infra.plan --backend modal --config configs/cloud_modal.yaml` prints resolved config, hard caps
and an **ESTIMATE**: `cpu_core_hours_upper_bound = cpu x max_containers x (1+max_retries) x timeout_s / 3600` (a cap
that cannot be exceeded) and, only if the config supplies `estimate.steps_per_sec` (a user assumption, labeled so),
an expected runtime. No prices, credits or billing are inferred. Credentials are checked by presence only
(env var names / `~/.modal.toml` existence; contents never read). Missing credentials -> setup instructions and
`effective_backend: local`; Daytona -> always local (DEFERRED).

Secret names (values never read into code or logs): Modal `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` (or `modal setup`);
Daytona `DAYTONA_API_KEY`; Modal Secret objects mounted into jobs via `secret_names` (none needed today).

## Inference export
`PythonReferenceBackend(checkpoint)` (oracle, real Normalizer) / `OnnxRuntimeBackend(onnx)` / `NativeBackend(onnx)`
share `InferenceBackend.run(batch, state=None) -> (logits (B,N) f32, value (B,) f32, state)`. Dtype/shape/contiguity
rules are in the `hospital_rl.infra.inference` docstring: strict dtypes, C-contiguous, no silent casting, raw
(un-normalized) features in, normalization baked into the graph, state must be `None` (feed-forward only).
`export_onnx(ckpt, out)` writes opset 17 with dynamic batch/history axes, metadata in ONNX `metadata_props["hospital_rl"]`
and `<out>.meta.json` (checkpoint id, role, catalog/obs hashes, max_nodes, policy_config, normalizer stats, policy
version, exporter versions). Backends refuse a model without metadata or with mismatching expected hashes.
Uses the legacy TorchScript exporter (`dynamo=False`; deprecated in torch >= 2.9 but parity-verified here).

Parity (tested): batches 1/7/64, normal and extreme-finite inputs (features ~1e6), all-non-wait-invalid masks,
garbage in padded nodes. Tolerance: logits `atol = rtol = 1e-4` on legal entries, masked entries identical (`-1e9`),
values 1e-4, argmax equal except where the top-2 legal logits are within 1e-4. GRU export: not built.

### Native build (verified on macOS arm64, Apple clang, 2026-10-20)
Prerequisites: C++17 compiler, `cmake`, `pybind11` (both `pip install cmake pybind11`), and an ONNX Runtime **C++
release** (headers + libonnxruntime; the pip wheel has no headers): e.g. `onnxruntime-osx-arm64-1.31.0.tgz` from
github.com/microsoft/onnxruntime/releases (sha256 of the one used: `1031b36d...2afc`).
```bash
cmake -S native -B native/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-arm64-1.31.0 \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DPython_EXECUTABLE=$(which python)
cmake --build native/build
pytest tests/infra/test_native.py          # skipped (not passed) if the module is not built
```
Linux/Windows builds: untested. The runner is in-process (no subprocess per action, no network), GIL released during
`run`, bad dtype/layout rejected on the C++ side too.

### Benchmark (toy model, one machine, 1 thread, CPU)
`python benchmarks/inference/run.py [--checkpoint ...]`; stored: `benchmarks/inference/results_toy_macos_arm64.json`.
Single run, toy 8-node model; no speedup target is claimed and the python backend remains the default.

## Tested vs deferred
Tested: masked PPO, padding invariance, versioned rollouts, checkpoints (atomic/kill/resume/compat), interrupts,
frozen-opponent adapter (toy), league/curriculum, local backend (done/drain/stop/cancel/timeout/resume), artifacts,
Modal logic via fake gateway, live tiny Modal smoke, cloud-plan, ONNX + native parity, benchmark script.
**DEFERRED / UNTESTED**: Daytona; Modal second mode (distributed rollout workers), live Modal control ops and
preemption; GRU policy/export; reconciler for orphaned resources; GPU; Linux/Windows native builds; the real hospital
env (only the toy env was used); hosted dashboard endpoints.

## Cloud school training (`infra/cloud_school.py`)
Runs the real school driver (curriculum + validation + best.pt) off-device, one container/sandbox per (role, seed), all parallel.
- Modal: `PYTHONPATH=src modal run -d src/hospital_rl/infra/cloud_school.py --steps 200000 --seeds 7,8,9`, then
  `modal volume get hospital-rl-school /<tag> runs/cloud-<tag>` (target dir must not exist). Detached; survives the client.
  Same tag resumes. 2 CPU / 4 GiB per run, 24 h timeout.
- Daytona: `python -m hospital_rl.infra.cloud_school daytona --steps 50000 --seeds 7,8` (client must stay up; results in `runs/daytona/<tag>`).
- Verified 2026-10-21: both smoke-tested live (defender, 4096 / 2048 steps, level 0). Not tested: long runs, resume after crash, cost.
- "Faster" = more seeds/roles in parallel, not a faster single run (SyncVecEnv is single-process).
