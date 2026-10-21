# Requests from Agent E (cloud lifecycle) to the lead

R1. **CLI** (`cli.py`, lead-owned): optional subcommands `hospital-rl cloud-plan`, `hospital-rl cloud-reconcile` delegating to
`hospital_rl.infra.plan.main` / `hospital_rl.infra.reconcile.main` (both take `argv`). Today they run via `python -m`.

R2. **`training/schools.py` control hook**: `run_school(..., control=None)` forwarding `control` into its `train(...)` call and
returning early on a drain/stop. Today school jobs honor drain/stop only at `eval_every` chunk boundaries (infra drives
`run_school` one chunk at a time). Not required for correctness.

R3. **Experiment controller**: consume `GatewayBackend.status().checkpoint` (`env_steps`, `age_s`) for freshness; treat
`FAILED` details starting with "spawn outcome unknown" as "attempt id burned, resubmit with a new attempt id";
a `lost` Modal call surfaces as `FAILED` ("owning controller is gone") - cancel it before resuming. Registry dir must be
persisted by the controller (default `runs/.registry/<backend>`).

R4. **Packaging / dependencies** (`pyproject.toml`, lead-owned): `cloud` extra already lists `modal`; no new dependency was
added. The `.venv` lacks `modal`, so Modal-SDK tests skip there (they pass in an environment that has it).

R5. **Consumption of pinned versions**: `JobSpec.opponent_checkpoint_ids / policy_version / normalizer_version` are validated
and hashed but nothing in `training/train.py` consumes them; wire them when league/rollout mode lands.

R6. **`.gitignore`**: `runs/.registry/` holds attempt records; add it if runs/ is not already ignored.
