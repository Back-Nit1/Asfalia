# Requests from Agent D (control plane + dashboard)

## To Agent E (backend contract the controller depends on)
Controller code: `src/hospital_rl/experiments/controller.py`; it only touches `infra.backend.JobBackend` + files under `<root>/<run>/<attempt>/`.
1. **Idempotent `submit`** per `(run_id, attempt_id)` (already true for `LocalBackend` within a process). After a controller restart the
   controller re-submits a `starting` run with no recorded job id and expects the same job id back, not a second job.
2. **`status(job_id)` must raise `LookupError` for a job it does not know** (e.g. `LocalBackend` after a restart). The controller then decides
   only from durable files (`result.json` + verified `manifest.json`); otherwise the run becomes `failed: outcome unknown`. A Modal gateway that can
   recover by remote id should keep returning the real state instead. Any other exception is shown as `backend_error` and does not change state.
3. **`LocalBackend` must not discard job stderr/stdout** and the job must write progress files (below). Heartbeat today = last successful `status()`;
   a backend-side `last_seen`/`updated_at` on `JobStatus` would be better.
4. **Progress files in the attempt dir** (append-only JSONL, one object per line, flushed per round):
   `metrics.jsonl` rows `{"env_steps": int, "policy_version": int, "mean_return": float|null, "t": <unix seconds>, ...ppo stats}`; optional
   `episodes.jsonl` rows `{"event_type", "episode_id", "visibility": "attacker|defender|evaluator|operator", "payload": {...}}`.
   Without `t` no step-rate estimate is shown. `train()` already builds exactly this `row`; it only needs writing.
5. **Resume** is `resume_from_attempt` → `latest.pt` only; the controller refuses other checkpoint ids. Per-checkpoint resume would need a spec field.
6. **Timeout/stop result**: `TIMED_OUT` with a checkpoint becomes `stopped`; without one `failed`. Please keep `ckpt_*.pt` + `ckpt_*.pt.meta.json`
   naming (the reader indexes those, not `latest.pt`).
7. Optional: owned-resource listing (`backend.list_owned()` with labels/age) so the Infrastructure page can show remote orphans; today orphans =
   attempt dirs in the artifact root with no DB row.
8. **Evaluator**: the controller takes an injected `Evaluator.run(spec, {"attacker": path, "defender": path}, out_dir) -> {"pairs": [{attacker, defender, metric,
   mean, ci_low, ci_high, n}], "episode_ids": [...]}`; episodes must be written as export JSON under `out_dir`'s parent (see `api/replay.parse_episode`).
   Agent C/F own the real implementation; none is wired, so real-backend evaluations are refused with "no evaluation backend configured".

## To the lead (cli.py / schemas / events)
1. `cli.py`: `dashboard` should delegate to `hospital_rl.api.__main__.main(rest)`; new flags `--experiments-root DIR`, `--fake-backend`, `--tick-seconds`.
2. `events.py`: no change needed (`SourceKind` already has `experiment`/`infrastructure`). Event types emitted: `run_queued|starting|running|draining|checkpointing|
   stopped|failed|cancelled|completed`, `training_metric`, `checkpoint_written`, `evaluation_finished`. Please document them next to `EventEnvelope` if you want them contractual.
3. Episode export (`runner.run_episode`) still lacks the `topology` block (see agent-A.md #2); eval-exported episodes need it for the World view.
4. `pyproject.toml`: no new dependency (pydantic/fastapi/httpx already present). Playwright config now expects `.venv/bin/python` at the repo root as before.
