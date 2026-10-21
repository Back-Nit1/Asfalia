# Experiment configs (Agent C)

Run: `PYTHONPATH=src .venv/bin/python -m hospital_rl.training.experiments run --config configs/experiments/<name>.yaml --out <dir>`
Re-running with the same `--out` and config resumes; a different config/seed on that dir is refused (`lineage.json`).

| config | what it does |
|---|---|
| `smoke.yaml` | tiny CPU run, all arms, real envs (`smoke_world.yaml`, 12 nodes) |
| `pretrained_adapt.yaml` | schools -> frozen Stage D transfer -> adapt from school weights |
| `scratch_adapt.yaml` | same frozen opponent + same adapt budget, random init |
| `equal_total.yaml` | pretrained arm vs scratch given `adapt.steps + school steps` (equal total env steps) |
| `frozen_transfer.yaml` | Stage D only, explicit checkpoints (replace `CHANGE_ME`) |

Outputs under `--out`: `school/<role>/`, `stage_d/transfer_{before,after}_adaptation.json`, `arms/<arm>/` (with
`adapt_manifest.json`, `league.json`, `eval_pool.json`), `comparison_manifest.json`. Comparisons are rejected (recorded
with a reason, exit 2), never averaged, when world/split/catalog hash, opponent checkpoint, league flag, eval budget or
the chosen budget rule (`equal_adapt` / `equal_total`) differ. Smoke numbers are not evidence of policy quality.
