"""Simulator counterfactuals (not causal claims about hospitals). Exogenous draws are keyed, so branches share
workload/telemetry/transition randomness wherever the same (tick, action, node) key recurs."""
from __future__ import annotations

import math

from ..engine.transitions import Action
from ..runner import run_episode
from ..schemas import ScenarioSpec


class _Replay:
    """Fixed attacker intent: re-issue recorded actions; blocked ones simply fail in the new world."""

    def __init__(self, actions):
        self.actions, self.i = actions, 0

    def act(self, view, legal):
        a = self.actions[self.i]
        self.i += 1
        return a if a in legal else Action("wait")


def attacker_actions(ep: dict) -> list[Action]:
    return [Action(s["attacker"]["type"], s["attacker"]["node"]) for s in ep["steps"]]


def fixed_intent(ep: dict, defenders: dict[str, callable]) -> dict:
    spec = ScenarioSpec.model_validate(ep["spec"])
    acts = attacker_actions(ep)
    return {"estimand": "effect on THIS fixed attack sequence under different defender configurations",
            "interventions": list(defenders),
            "results": {name: run_episode(spec, ep["seed"], _Replay(acts), mk())["metrics"]
                        for name, mk in defenders.items()}}


def adaptive(spec: ScenarioSpec, seeds: list[int], attacker_factory, defenders: dict[str, callable]) -> dict:
    """Rerun a frozen adaptive attacker; the realized attack path may change (different estimand)."""
    out = {}
    for name, mk in defenders.items():
        runs = [run_episode(spec, s, attacker_factory(), mk())["metrics"] for s in seeds]
        out[name] = {k: _ci([r[k] for r in runs]) for k in ("weighted_exposure_frac", "critical_success", "spend")}
    return {"estimand": "effect of posture on a frozen ADAPTIVE attacker; attack path may differ per posture",
            "interventions": list(defenders), "n_seeds": len(seeds), "results": out}


def _ci(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) if n > 1 else float("nan")
    h = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": m, "ci95": [m - h, m + h], "n": n}
