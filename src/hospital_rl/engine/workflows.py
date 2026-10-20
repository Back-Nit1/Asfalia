"""Benign clinical workloads. Success/failure is computed from simulated permissions and service state."""
from __future__ import annotations

from ..schemas import ScenarioSpec
from .permissions import Topology, edge_open, role_ok, service_up
from .rng import draw
from .state import WorldState


def run_workflows(topo: Topology, st: WorldState, spec: ScenarioSpec, seed: int, events: list):
    """Returns (crit_fail_frac, noncrit_fail_frac, added_latency_frac) for this tick and logs events."""
    tot = {"critical": 0, "noncritical": 0}
    fail = {"critical": 0, "noncritical": 0}
    added = 0.0
    n_req_all = 0
    for w in spec.workflows:
        n_req = int(w.rate) + (1 if draw(seed, "wf_count", st.tick, w.id) < w.rate - int(w.rate) else 0)
        cls = "critical" if w.critical else "noncritical"
        for k in range(n_req):
            tot[cls] += 1
            n_req_all += 1
            ok, why, lat = _request(topo, st, spec, w)
            base = len(w.path) - 1
            if not ok and why == "role_or_deadline" and w.emergency and st.overrides_left > 0:
                st.overrides_left -= 1
                st.m.emergency_overrides_used += 1
                events.append({"type": "emergency_override", "workflow": w.id, "tick": st.tick, "visible": "defender"})
                ok, lat = True, base
            if ok:
                added += max(0, lat - base)
            else:
                fail[cls] += 1
                events.append({"type": "workflow_failed", "workflow": w.id, "reason": why, "tick": st.tick,
                               "visible": "defender"})
            if ok and lat > base:
                events.append({"type": "workflow_delayed", "workflow": w.id, "added": lat - base, "tick": st.tick,
                               "visible": "defender"})
    for c in tot:
        st.m.wf_total[c] += tot[c]
        st.m.wf_failed[c] += fail[c]
    st.m.added_latency += added
    cf = fail["critical"] / tot["critical"] if tot["critical"] else 0.0
    nf = fail["noncritical"] / tot["noncritical"] if tot["noncritical"] else 0.0
    lf = added / (max(n_req_all, 1) * spec.latency_norm)
    return cf, nf, min(lf, 1.0)


def _request(topo, st, spec, w):
    for n in w.path:
        if not service_up(topo, st, n):
            return False, "service_down", 0
    for a, b in zip(w.path, w.path[1:]):
        if b not in topo.out[a]["reach"] or not edge_open(st, a, b):
            return False, "unreachable", 0
    for n in w.path:
        if not role_ok(st, n, w.role):
            return False, "role_or_deadline", 0
    lat = (len(w.path) - 1) + spec.mfa_latency * sum(1 for n in w.path if st.nodes[n].mfa)
    if lat > w.deadline:
        return False, "role_or_deadline", lat
    return True, "", lat
