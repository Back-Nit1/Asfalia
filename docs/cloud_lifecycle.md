# Compute lifecycle (Agent E)

One lifecycle for every provider. Status legend: **TESTED** (fakes/local, in `tests/infra`), **LIVE** (executed against a
real provider and recorded), **UNEXECUTED** (code exists, never run live), **DEFERRED** (not built).

## Pieces
| Module | Role |
|---|---|
| `infra/backend.py` | `JobSpec` = the versioned job bundle, `JobStatus`, `JobBackend`, `ArtifactStore` ABCs |
| `infra/bundle.py` | `resolve(spec)` fills pins; `validate(spec, check_code=)` refuses stale/unsafe/unrunnable bundles |
| `infra/job.py` | `run_job`: what actually runs inside a container/subprocess (train or school), control, deadline, manifest |
| `infra/lifecycle.py` | `GatewayBackend` (all semantics), `Registry` (durable attempt records), `LocalGateway`, `Gateway` protocol |
| `infra/local_backend.py`, `modal_app.py`, `modal_functions.py` | thin local / Modal transports on the same backend |
| `infra/reconcile.py` | owned-resource reconcile + cleanup (dry run by default) |
| `infra/cloud_school.py` | role x seed school fan-out; its Modal path only builds `kind="school"` specs |
| `benchmarks/rollout_scaling/` | single vs vectorized vs (opt-in) Modal workers |

## The job bundle (`JobSpec`, `bundle_version = 1`)
`kind` (`train` | `school`), `run_id`, `attempt_id`, `train_config` (TrainConfig fields, or for `school`:
`role, config, steps, seed, eval_every, n_val, n_envs, rollout_len, ppo`), `scenario`, `split`, `seed_range` (the split's
`[start, stop)`), `opponent_checkpoint_ids`, `policy_version`, `normalizer_version`, `horizon`, `output_path`
(`<run>/<attempt>`), `code_hash` (sha256 of the `hospital_rl` source tree), `config_hash`, `labels`, plus resources
(`timeout_s, grace_s, cpu, memory_mb, max_containers, max_retries`) and `resume_from_attempt`.
`submit()` calls `resolve` then `validate` then the caps check, all **before** anything is spawned. Rules enforced:
env factory must be `hospital_rl.*` and importable (no test modules); ids are safe path components; `config_hash`,
`output_path`, split/seed-range and bundle version must match; inside a container `code_hash` must equal the shipped code.
`opponent_checkpoint_ids`, `policy_version`, `normalizer_version` are carried and hashed into the bundle but are not yet
*consumed* by the learner (no training entry point takes pinned opponents): **DEFERRED** until the league/rollout mode exists.

## Backend semantics (identical for local and Modal; **TESTED**)
- `submit`: idempotent per `(run_id, attempt_id)` across controller restarts (registry). Intent is recorded before the
  spawn, so a crash in between yields FAILED "spawn outcome unknown; attempt id burned" rather than a second launch.
- `status`: `PENDING/RUNNING/DONE/DRAINED/STOPPED/FAILED/CANCELLED/TIMED_OUT`; terminal states are persisted and final.
  `JobStatus.checkpoint = {checkpoint, env_steps, written_at, age_s}` is the last durable checkpoint (from
  `progress.json`, written after every checkpoint); failures add whether artifacts are partial (no manifest) and how to resume.
- `drain` / `checkpoint_and_stop` write a backend-owned control word; `force_cancel` kills (no checkpoint). No-ops once terminal.
- **Timeouts**: the job stops itself at `timeout_s` (checkpoint, `TIMED_OUT`, manifest). Hard kill at `timeout_s + grace_s`
  (local supervisor / Modal function timeout). The checkpoint therefore lands inside the grace margin whenever the learner
  reaches a boundary; a stuck learner is killed with the last published checkpoint intact.
- **Retries** (`max_retries`): same attempt, resumes from that attempt's `latest.pt` (Modal `Retries`; local supervisor).
  A retry after completion returns the stored result and never publishes a second manifest. Steps are counted in the
  checkpoint, so nothing is double counted (TESTED with a SIGKILL mid-run: final steps == cap).
- **New attempt** = `resume_from_attempt`; the source must not be live; a finished source must verify against its manifest.
- **School jobs**: drain/stop are honored at validation-chunk boundaries (`eval_every` env steps), because
  `run_school` has no control hook and training code is not ours to edit. Request R2 in `docs/requests/agent-E.md`.

## Modal gateway lifecycle
- Remote call ids live in the registry (`runs/.registry/<backend>` by default), not process memory.
- `mode: ephemeral` (default, the only mode live-verified before this work): controller holds `app.run()`; app stops with
  the last owned call or `close()`. Restarted controller: finished calls are pollable; a call it does not own that still
  reports running is returned as FAILED("lost") with the last checkpoint (cancel it, then resume with a new attempt).
  **Limitation, stated precisely:** a controller restart does not keep an ephemeral job alive; what Modal does to containers
  when the owning client dies abruptly was not verified.
- `mode: deployed` (**UNEXECUTED**): `RealModalGateway.deploy()` then `Function.from_name(app, "train_job").spawn`; calls
  outlive the controller and results are readable 7 days (Modal docs). `cloud_school modal --detach`.
- Failure mapping: function timeout -> `TIMED_OUT`; quota/startup/preemption/remote exception -> `FAILED` with the SDK
  message; expired output -> `FAILED` (read `result.json` from the Volume). Real-gateway behavior for these is
  **UNEXECUTED** (fake gateways only).
- Clean-container packaging: image = CPU torch + `DEPS` + `add_local_python_source("hospital_rl")` + `configs/`. Proven
  locally by a fresh interpreter with only `src/` on the path building the configured env and validating the bundle
  with `check_code=True` (**TESTED**); proven in a real container only by the live smoke (**UNEXECUTED**).

## Reconcile / cleanup
`python -m hospital_rl.infra.reconcile --backend local --root <store> [--apply --ttl-hours N --include-partial]`
(`--backend modal --config configs/cloud_modal.yaml`). Dry run by default. Candidates must carry `labels.owner=asfalia`
in the registry. Never deleted: unresolved attempts, the newest terminal attempt per run, a source an unresolved attempt
resumes from, attempts younger than the TTL, attempts whose manifest does not verify, partial (no-manifest) attempts
unless `--include-partial`, and anything without a registry record (reported under `orphans_not_deleted`).
The registry record of a cleaned attempt is kept (lineage). Modal calls cannot be enumerated by the SDK, so remote
calls unknown to the registry are undiscoverable. **TESTED** with fakes; the Modal delete path is **UNEXECUTED**.

## Interface Agent H (Daytona) consumes
Do not change these; implement a `Gateway` and reuse `GatewayBackend`:
```python
class Gateway(Protocol):
    def spawn(self, spec: JobSpec) -> str                      # start one attempt, return a durable remote id
    def poll(self, remote_id) -> tuple[str, dict | str]        # running|done(result dict)|timed_out|failed|lost
    def cancel(self, remote_id) -> None
    def write_control(self, run_id, attempt_id, word) -> None  # "drain" | "stop" -> <root>/<run>/control/<attempt>
    def read_file(self, run_id, attempt_id, name) -> bytes | None   # progress.json, manifest.json, result.json, ...
    def list_attempts(self) -> list[tuple[str, str]]
    def delete_attempt(self, run_id, attempt_id) -> None
```
Inside the sandbox run `python -m hospital_rl.infra.job --spec <spec.json> --root <store>` (or call `run_job`); it
validates the bundle, honors the control file, and writes `result.json`, `progress.json`, `manifest.json`. Register with
`GatewayBackend("daytona", gateway, Registry(dir), ResourceCaps(...))`.
Implemented by Agent H in `infra/daytona_backend.py` (see [daytona.md](daytona.md)).

## Live status of this work
No live cloud call was made by this work (no explicit cloud budget was given). The earlier 2026-10-20/21 live results in
`docs/training_infra.md` predate this refactor and do not validate it. To run: `ASFALIA_LIVE_MODAL=1 PYTHONPATH=src python
tests/infra/modal_smoke.py` (<= 3 jobs, 2048 steps each, 1 cpu, 120 s) and record the printed call ids.
