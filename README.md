# Hospital Security RL Simulator

Synthetic graph simulator + benchmark for separately trained attacker/defender policies, evaluated against
disruption of clinical workflows. Spec: `prompt (2).md`. All data is synthetic; actions are typed abstract transitions.

## Status (updated by the lead; no results are claimed beyond what is listed)

| Area | State |
| --- | --- |
| Engine, 8-node fixture, workflows, permissions, budget, telemetry | implemented, unit-tested (Gate 1) |
| PettingZoo env, masks, slot permutation, single-role adapter | implemented, `parallel_api_test` passes (Gate 2) |
| Scenario generator (4 topology families, 8-40+ nodes), splits | implemented |
| Trajectory JSONL export, fixed-intent / adaptive counterfactuals, replay | implemented, smoke-tested |
| Scripted baselines + baseline benchmark | implemented; scripted attacker only, uncalibrated |
| PPO training, checkpoints, league, Modal/Daytona, ONNX/C++ | in progress on branch `agent/training-infra` |
| Dashboard, API, orchestrator | in progress on branch `agent/dashboard-orchestration` |
| Learned-policy experiments | **none run** |

## Setup and commands
```bash
uv sync --extra dev
uv run pytest
uv run hospital-rl demo --config configs/tiny.yaml --seed 7          # writes runs/demo/episode.json
uv run hospital-rl replay --episode runs/demo/episode.json
uv run hospital-rl counterfactual --episode runs/demo/episode.json --mode fixed-intent   # or adaptive
uv run hospital-rl benchmark --split validation --episodes 20 --size 12
```
Other CLI subcommands are stubs and print "not implemented yet".

## Metrics
Records exfiltrated (unique; sensitivity-weighted exposure fraction), critical/noncritical workflow success,
added latency, spend, downtime node-ticks, containment time (`None` = censored, never 0), false-positive alerts on
benign episodes, availability violation (critical success < 0.95; a benchmark parameter, not a clinical standard).

## Limitations
Uncalibrated constants and rewards; one scripted attacker; graph abstraction, not a hospital replica or HIPAA
assessment; unpatchable devices and the 95% threshold are scenario assumptions. See `docs/contracts.md`.
