# Requests from Agent H to the lead

1. `pyproject.toml` (lead-owned): pin the SDK in the cloud extra: `cloud = ["modal", "daytona==0.225.0"]`; `python-dotenv` is no longer
   needed by any Daytona code (`cloud_school` was its only user in this repo section: please confirm and drop it).
2. Rollout-only job kind (`kind="rollout"`: pinned split, seed range, opponent checkpoints, policy/normalizer versions, horizon,
   output path) in `infra/bundle.py`/`job.py`. Daytona's "graph rollout smoke" currently runs a tiny `kind="train"` attempt.
3. CLI: expose `hospital-rl cloud-plan --backend daytona` (already works via `infra.plan`) and optionally `daytona-smoke` wrapping
   `scripts/daytona_smoke.py`. No other shared contract changed.
4. `infra/plan.py` and `tests/infra/test_infra.py` were edited only to turn Daytona from "DEFERRED, falls back to local" into a real
   dry run; the Modal/local paths are unchanged.
