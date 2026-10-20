"""Defender telemetry: noisy, delayed, or missing evidence derived from true events."""
from __future__ import annotations

from ..schemas import ScenarioSpec
from .rng import draw
from .state import Alert, WorldState

KIND = {"attempt_initial_access": "auth_anomaly", "move": "auth_anomaly",
        "attempt_privilege_change": "process_anomaly", "collect": "flow_anomaly",
        "exfiltrate": "flow_anomaly", "discover": "flow_anomaly", "probe": "flow_anomaly"}
WEIGHT = {"discover": 0.2, "probe": 0.5}   # recon is quieter than action-on-objective


def emit_for_true_event(st: WorldState, spec: ScenarioSpec, seed: int, kind: str, node: str):
    p = spec.detect_by_logging[st.nodes[node].logging] * WEIGHT.get(kind, 1.0)
    if draw(seed, "detect", st.tick, kind, node) < p:
        delay = 1 + (draw(seed, "delay", st.tick, kind, node) < 0.3)
        st.pending_alerts.append(Alert(st.tick + delay, node, KIND[kind], True))


def benign_noise(st: WorldState, spec: ScenarioSpec, seed: int):
    for n, ns in st.nodes.items():
        if draw(seed, "fp", st.tick, n) < spec.false_positive_by_logging[ns.logging]:
            st.pending_alerts.append(Alert(st.tick + 1, n, "benign_noise", False))


def release(st: WorldState, drop_prob_stream_seed: int, drop: float = 0.1) -> list[Alert]:
    """Move due alerts to the defender; a fraction is lost (missing telemetry)."""
    due = [a for a in st.pending_alerts if a.visible_tick <= st.tick]
    st.pending_alerts = [a for a in st.pending_alerts if a.visible_tick > st.tick]
    out = [a for i, a in enumerate(due) if draw(drop_prob_stream_seed, "drop", st.tick, a.node, a.kind, i) >= drop]
    for a in out:
        st.m.true_positive_alerts += a.true_positive
        st.m.false_positive_alerts += (not a.true_positive)
    st.visible_alerts.extend(out)
    return out
