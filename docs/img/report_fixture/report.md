# Asfalia report: smoke (report v1)

**Evidence level: INSUFFICIENT: 2 seeds < 5; spread and CI are shown but are not a basis for claims**

Seeds: [1, 2]. Code revision: `f44a68938e7e`. Compatibility: checked (world/split/catalog/schema/league/validation episodes identical across runs).

## Adaptation arms (validation split, last evaluation, per-seed + mean ± std [95% CI])

### pretrained (adapt 128, pretrain 128, total 256 env steps)
- weighted_exposure_frac: 0.000 ±0.000 [0.000, 0.000] (n=2); per-seed {'1': 0.0, '2': 0.0}
- critical_success: 0.884 ±0.164 [-0.594, 2.361] (n=2); per-seed {'1': 1.0, '2': 0.7674278846153846}
- spend: 34.938 ±1.503 [21.437, 48.438] (n=2); per-seed {'1': 33.875, '2': 36.0}
- success: 0.500 ±0.707 [-5.853, 6.853] (n=2); per-seed {'1': 1.0, '2': 0.0}
- score: -0.500 ±0.707 [-6.853, 5.853] (n=2); per-seed {'1': -0.0, '2': -1.0}

### scratch_same_adapt (adapt 128, pretrain 0, total 128 env steps)
- weighted_exposure_frac: 0.000 ±0.000 [0.000, 0.000] (n=2); per-seed {'1': 0.0, '2': 0.0}
- critical_success: 0.880 ±0.170 [-0.647, 2.407] (n=2); per-seed {'1': 1.0, '2': 0.7596153846153846}
- spend: 34.938 ±1.503 [21.437, 48.438] (n=2); per-seed {'1': 33.875, '2': 36.0}
- success: 0.500 ±0.707 [-5.853, 6.853] (n=2); per-seed {'1': 1.0, '2': 0.0}
- score: -0.500 ±0.707 [-6.853, 5.853] (n=2); per-seed {'1': -0.0, '2': -1.0}

### scratch_equal_total (adapt 256, pretrain 0, total 256 env steps)
- weighted_exposure_frac: 0.000 ±0.000 [0.000, 0.000] (n=2); per-seed {'1': 0.0, '2': 0.0}
- critical_success: 0.880 ±0.170 [-0.647, 2.406] (n=2); per-seed {'1': 1.0, '2': 0.7597355769230769}
- spend: 33.875 ±3.005 [6.875, 60.875] (n=2); per-seed {'1': 31.75, '2': 36.0}
- success: 0.500 ±0.707 [-5.853, 6.853] (n=2); per-seed {'1': 1.0, '2': 0.0}
- score: -0.500 ±0.707 [-6.853, 5.853] (n=2); per-seed {'1': -0.0, '2': -1.0}

## Stage D cross-play, before adaptation (validation split, 2 episodes per pair)

**weighted_exposure_frac** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 0.000 | 0.637 | 0.000 | 0.280 |
| learned_attacker | 0.000 | 0.000 | 0.000 | 0.000 |

**critical_success** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 0.655 | 1.000 | 0.899 | 0.534 |
| learned_attacker | 0.831 | 1.000 | 0.899 | 0.534 |

**spend** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 24.450 | 0.000 | 36.000 | 36.000 |
| learned_attacker | 19.100 | 0.000 | 36.000 | 36.000 |

## Stage D cross-play, after adaptation (validation split, 2 episodes per pair)

**weighted_exposure_frac** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 0.000 | 0.637 | 0.000 | 0.162 |
| learned_attacker | 0.000 | 0.000 | 0.000 | 0.000 |

**critical_success** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 0.655 | 1.000 | 0.899 | 0.627 |
| learned_attacker | 0.831 | 1.000 | 0.899 | 0.627 |

**spend** (rows attacker, cols defender; mean over seeds)

| attacker \ defender | GreedyDefender | Noop | StaticHardening | learned_defender |
|---|---|---|---|---|
| ScriptedAttacker | 24.450 | 0.000 | 36.000 | 36.000 |
| learned_attacker | 19.100 | 0.000 | 36.000 | 36.000 |

## Figures

- `security.svg`, `availability.svg`, `cost.svg`: Stage D, learned vs scripted defenders
- `arms_security.svg`, `arms_availability.svg`, `arms_cost.svg`: adaptation arms

## Notes

- Rewards are uncalibrated; scripted opponents and one synthetic world family set are all that is tested.
- Arms were compared only where Agent C's budget-rule compare() accepted them (see rejected_comparisons).
- Cross-play cells with a learned actor aggregate different per-seed checkpoints under one label.
