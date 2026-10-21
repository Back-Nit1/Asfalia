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
| `infra/` | `backend` (ABCs), `artifacts`, `job`, `local_backend`, `modal_app`, `daytona_backend`, `plan`, `inference`, `inference_cli`, `inference_bench` |
| `native/` | ONNX Runtime C++ runner + pybind11 bindings |
| `benchmarks/inference/` | stored benchmark results; `run.py` wraps `infra/inference_bench` |

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
- **ModalBackend**: same `GatewayBackend` as local, over `RealModalGateway`; see [cloud_lifecycle.md](cloud_lifecycle.md)
  for the bundle, semantics, ephemeral vs deployed modes, packaging, and limits. Not built: second-mode distributed
  rollout workers.
- **DaytonaBackend**: **DEFERRED** stub, every method raises `NotImplementedError("DEFERRED ...")`.

### Live Modal smoke (tiny, opt-in)
`ASFALIA_LIVE_MODAL=1 PYTHONPATH=src python tests/infra/modal_smoke.py` (real hospital env; done + stop + resume; <= 3 jobs).
An earlier toy-env smoke (2026-10-20: done, 2048 steps, ~20-25 s) used the pre-refactor gateway and does not validate
the current one. **The refactored gateway has not been run live.** Leftover Volume runs `smoke-ab7987a6`, `smoke-fbcef79a`
in `hospital-rl-smoke-runs` predate cleanup and are unowned by any registry (reconcile would report, never delete, them).

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

### Export / benchmark CLIs
```bash
hospital-rl export-policy --checkpoint runs/attacker/best.pt --out out/attacker.onnx --expect-role attacker \
    [--expect-catalog-hash H] [--expect-observation-hash H] [--overwrite]   # one JSON object on stdout; exit 0 / 2
hospital-rl benchmark-inference --checkpoint runs/attacker/best.pt [--backends python,python_compiled,onnxruntime,cpp] \
    [--batches 1,32,256] [--warmup 30] [--repetitions 300] [--threads 1] [--env-steps 300] [--out report.json]
```
(until `cli.py` wires `benchmark-inference`: `python -m hospital_rl.infra.inference_bench ...`; see `docs/requests/agent-G.md`.)
Export writes `<out>` and `<out>.meta.json` via temp files + rename, refuses to overwrite without `--overwrite`, and
re-loads what it wrote. Every ONNX/native load runs `check_meta`: missing metadata, unknown `export_format`, or a
role/catalog/observation-hash mismatch raises `IncompatibleCheckpointError`. GRU export remains unsupported.

Benchmark: golden corpus = real hospital observations (hub/tiered/flat/segmented x 12/24/40/60 nodes, padded to 64, random
legal play for both roles) plus an extreme-but-finite variant. Each backend must pass parity vs the Python reference
(logits 1e-4 on legal entries, argmax equal outside the 1e-4 near-tie band) on both corpora, else exit 2 and no numbers.
Timed unit = validation + normalization + inference + copy + argmax. `env_steps_per_s` = full single-role env loop, B=1.
`memory_mb` is null (peak RSS is not attributable per backend in one process). A backend whose toolchain is missing is
reported `unavailable`, not failed.

### Measured (macOS arm64, 12 cores, 1 thread, ORT 1.31.0, torch 2.14.1; one run each, real hospital checkpoints)
Checkpoints: `runs/attacker/best.pt` (55296 steps) and `runs/cloud/smoke/defender-s7/best.pt` (smoke-sized, 4096 steps).
Stored: `benchmarks/inference/results_hospital_{attacker,defender}_macos_arm64.json`. B=1 p50 (us): python 153-157,
python_compiled 120-121, onnxruntime 90, cpp 88-90. B=256 throughput (samples/s): python ~22k, others 36-40k. Env steps/s
are within ~7% across backends (the simulator dominates). **Native C++ is not faster than ONNX Runtime** (same engine,
differences are noise), so Python stays the default and ONNX the opt-in fast path; no speedup target exists.
`python_compiled` costs ~5 s on first run. Caveat: loading the pip `onnxruntime` and the C++ release dylib in one
process makes macOS print an Objective-C duplicate-class warning; benchmark numbers looked unaffected but run `cpp`
alone (`--backends python,cpp`) if in doubt.

Platforms: macOS arm64 built and tested (Apple clang, cmake 4.4.4, pybind11 3.1.0). Linux x86_64/aarch64: CMake logic
is portable (`libonnxruntime.so`, RPATH) and the instructions above apply with the `onnxruntime-linux-*` release, but
**no Linux build was run**. Without the toolchain `tests/infra/test_native.py` skips with one reason:
"native module not built (needs cmake, pybind11, ONNX Runtime C++ release)".


## Tested vs deferred
Tested: masked PPO, padding invariance, versioned rollouts, checkpoints, interrupts, league/curriculum, the common
lifecycle (local subprocess with the real hospital env: done/drain/stop/cancel/timeout/resume/kill-retry; fake
gateways: every state mapping, restart idempotence, cleanup protections), bundle validation + clean-interpreter import,
school jobs, cloud-plan, ONNX + native parity. **DEFERRED / UNEXECUTED**: Daytona (Agent H); Modal deployed mode and every
real-Modal control/timeout/preemption/quota path; Modal distributed rollout workers; learner consumption of pinned
opponent/normalizer versions; GRU; GPU; Linux/Windows native builds; hosted dashboard endpoints.

## Cloud school training (`infra/cloud_school.py`)
Role x seed school driver off-device, one container/sandbox per (role, seed), all parallel.
- Modal (common lifecycle, one `kind="school"` job per role x seed, run id `<tag>-<role>-s<seed>`):
  `PYTHONPATH=src python -m hospital_rl.infra.cloud_school modal --steps 200000 --seeds 7,8,9` (ephemeral: keep the process up),
  `--detach` (deployed app, UNEXECUTED) then `... status --tag <tag>`; fetch with `modal volume get hospital-rl-school /<run> runs/cloud-<run>`.
  Re-running a tag creates the next attempt, resuming from the previous one. 2 CPU / 4 GiB per run. **Regression to know:**
  the previous `modal run -d` fan-out survived the client; the default ephemeral mode does not (use `--detach` once verified).
- Daytona: unchanged: `python -m hospital_rl.infra.cloud_school daytona --steps 50000 --seeds 7,8`.
- The 2026-10-21 live smokes were of the pre-refactor launcher. "Faster" = more seeds/roles in parallel, not a faster single run.
