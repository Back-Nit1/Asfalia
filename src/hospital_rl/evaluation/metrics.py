"""Objective metrics from an evaluator-side episode record (``runner.run_episode`` output).

Censored quantities (recovery, containment) are never reported as 0: each is ``{"observed": [...], "censored": [...]}``
where a censored entry is the *observed lower bound* (time the episode ran without the event).
"""
from __future__ import annotations

from .counterfactual import _ci


def _runs(ticks: list[int], horizon: int) -> tuple[list[int], list[int]]:
    """Maximal consecutive-tick outages -> (recovered durations, right-censored lower bounds)."""
    done, cens, i = [], [], 0
    t = sorted(set(ticks))
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] == t[j] + 1:
            j += 1
        (cens if t[j] == horizon - 1 else done).append(t[j] - t[i] + 1)
        i = j + 1
    return done, cens


def episode_objectives(ep: dict, benign: bool = False) -> dict:
    spec, m, horizon = ep["spec"], ep["metrics"], len(ep["steps"])
    critical = {w["id"] for w in spec["workflows"] if w["critical"]}
    ev = [e for s in ep["steps"] for e in s["events"]]
    fails = [e for e in ev if e["type"] == "workflow_failed"]
    per_wf: dict[str, int] = {}
    for e in fails:
        per_wf[e["workflow"]] = per_wf.get(e["workflow"], 0) + 1
    rec, rec_cens = _runs([e["tick"] for e in fails if e["workflow"] in critical], horizon)
    truth = [e for e in ev if e["type"] == "attacker_truth"]
    lapsed = sum(e["type"] == "control_lapsed" for e in ev)
    return {
        "benign": benign,
        "attacker_success": (not benign) and m["records_exfiltrated"] > 0,
        "weighted_exposure_frac": m["weighted_exposure_frac"],
        "attacker_actions_succeeded": sum(e["success"] for e in truth),
        "recovery": {"observed": rec, "censored": rec_cens},
        "workflow_failures": {"by_workflow": per_wf, "critical_success": m["critical_success"],
                              "noncritical_success": m["noncritical_success"],
                              "note": "failure-event counts; success fractions are the denominator-aware summary"},
        "downtime_node_ticks": m["downtime_node_ticks"],
        "containment": ({"observed": [], "censored": []} if m["first_access_tick"] is None else
                        {"observed": [m["containment_tick"] - m["first_access_tick"]], "censored": []}
                        if m["contained"] else {"observed": [], "censored": [horizon - m["first_access_tick"]]}),
        "false_positive_alerts": m["false_positive_alerts"],
        "constraint_violations": {"availability_threshold": bool(m["availability_violation"]),
                                  "controls_lapsed": lapsed, "emergency_overrides_used": m["emergency_overrides_used"],
                                  "total": int(m["availability_violation"]) + lapsed},
    }


def _cens(parts: list[dict]) -> dict:
    obs = [x for p in parts for x in p["observed"]]
    cen = [x for p in parts for x in p["censored"]]
    n = len(obs) + len(cen)
    return {"observed": obs, "censored": cen, "n_events": n,
            "observed_mean": sum(obs) / len(obs) if obs else None,    # None, not 0: nothing observed
            "restricted_mean_lower_bound": (sum(obs) + sum(cen)) / n if n else None}


def aggregate(objs: list[dict]) -> dict:
    """Aggregate attack episodes and benign episodes separately; benign episodes only feed false positives."""
    att, ben = [o for o in objs if not o["benign"]], [o for o in objs if o["benign"]]
    out: dict = {"n_attack": len(att), "n_benign": len(ben)}
    if att:
        out |= {"attacker_success_rate": sum(o["attacker_success"] for o in att) / len(att),
                "weighted_exposure_frac": _ci([o["weighted_exposure_frac"] for o in att]),
                "downtime_node_ticks": _ci([o["downtime_node_ticks"] for o in att]),
                "recovery": _cens([o["recovery"] for o in att]),
                "containment": _cens([o["containment"] for o in att]),
                "critical_success": _ci([o["workflow_failures"]["critical_success"] for o in att]),
                "noncritical_success": _ci([o["workflow_failures"]["noncritical_success"] for o in att]),
                "constraint_violations": {k: sum(o["constraint_violations"][k] for o in att)
                                          for k in ("availability_threshold", "controls_lapsed", "total")}}
    if ben:
        out["benign_false_positive_alerts"] = _ci([o["false_positive_alerts"] for o in ben])
    return out


def visibility_shift_degradation(partial: dict[int, float], full: dict[int, float]) -> dict:
    """Paired by seed: metric(partial) - metric(full_diagnostic). Mismatched seeds or empty input is an error."""
    if not partial or set(partial) != set(full):
        raise ValueError("visibility-shift degradation needs identical, non-empty seed sets for both modes")
    d = [partial[s] - full[s] for s in sorted(partial)]
    return {"paired_seeds": sorted(partial), "per_seed": d, **_ci(d)}
