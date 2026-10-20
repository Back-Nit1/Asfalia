"""Run one full-horizon episode between two policies and return an evaluator-side record."""
from __future__ import annotations

from dataclasses import asdict

from .engine.transitions import World
from .schemas import ScenarioSpec, catalog_hash


def run_episode(spec: ScenarioSpec, seed: int, attacker, defender) -> dict:
    w = World(spec, seed)
    total = {"attacker": 0.0, "defender": 0.0}
    steps = []
    while True:
        a = attacker.act(w.view("attacker"), w.legal_attacker())   # both choose from pre-step views
        d = defender.act(w.view("defender"), w.legal_defender())
        info = w.step(a, d)
        for k in total:
            total[k] += info["reward"][k]
        steps.append({"tick": w.st.tick - 1, "attacker": asdict(a), "defender": asdict(d),
                      "events": info["events"], "reward": info["reward"], "components": info["components"]})
        if info["done"]:
            break
    return {"scenario": spec.name, "scenario_hash": spec.hash(), "catalog_hash": catalog_hash(), "seed": seed,
            "tie_policy": spec.tie_policy, "return": total, "metrics": summarize(w), "steps": steps}


def summarize(w: World) -> dict:
    m, spec = w.st.m, w.spec
    crit = 1 - m.wf_failed["critical"] / m.wf_total["critical"] if m.wf_total["critical"] else 1.0
    non = 1 - m.wf_failed["noncritical"] / m.wf_total["noncritical"] if m.wf_total["noncritical"] else 1.0
    return {"records_accessed": m.records_accessed, "records_exfiltrated": m.records_exfiltrated,
            "weighted_exposure_frac": m.weight_exfiltrated / w.total_weight,
            "critical_success": crit, "noncritical_success": non,
            "availability_violation": crit < spec.critical_success_threshold,
            "spend": m.spend, "added_latency": m.added_latency, "downtime_node_ticks": m.downtime_node_ticks,
            "containment_tick": m.containment_tick, "contained": m.containment_tick is not None,
            "first_access_tick": m.first_access_tick, "false_positive_alerts": m.false_positive_alerts,
            "true_positive_alerts": m.true_positive_alerts, "emergency_overrides_used": m.emergency_overrides_used}
