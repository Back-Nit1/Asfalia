"""Stage A baseline benchmark: identical held-out scenarios for every defender baseline."""
from __future__ import annotations

from ..policies.scripted import GreedyDefender, IsolateEverything, Noop, ScriptedAttacker, StaticHardening
from ..runner import run_episode
from .counterfactual import _ci
from .splits import SPLITS, scenario_for, split_hash

DEFENDERS = {"noop": Noop, "static_hardening": StaticHardening, "greedy": GreedyDefender,
             "isolate_everything": IsolateEverything}
KEYS = ("records_exfiltrated", "weighted_exposure_frac", "critical_success", "noncritical_success", "spend",
        "added_latency")


def run(split: str, n: int, size: int = 8, difficulty: float = 0.5, horizon: int = 40) -> dict:
    seeds = list(SPLITS[split]["seeds"])[:n]
    specs = {s: scenario_for(split, s, size, difficulty, horizon) for s in seeds}
    out = {"split": split, "split_hash": split_hash(), "n_scenarios": n, "defenders": {}}
    for name, mk in DEFENDERS.items():
        runs = [run_episode(specs[s], s, ScriptedAttacker(), mk())["metrics"] for s in seeds]
        benign = [run_episode(specs[s], s, _Idle(), mk())["metrics"] for s in seeds]
        row = {k: _ci([r[k] for r in runs]) for k in KEYS}
        thr = specs[seeds[0]].critical_success_threshold
        row["availability_violation_rate"] = sum(r["availability_violation"] for r in runs) / n
        row["feasible"] = row["critical_success"]["mean"] >= thr
        row["benign_false_positive_alerts"] = _ci([r["false_positive_alerts"] for r in benign])
        contained = [r["containment_tick"] for r in runs if r["containment_tick"] is not None]
        row["containment_time"] = {"observed": contained, "censored": n - len(contained)}   # never 0 for censored
        out["defenders"][name] = row
    return out


class _Idle:
    def act(self, view, legal):
        from ..engine.transitions import WAIT
        return WAIT
