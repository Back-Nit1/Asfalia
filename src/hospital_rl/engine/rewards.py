"""Illustrative, uncalibrated reward weights. All defender terms are <= 0, so cycling controls cannot profit."""
from __future__ import annotations

DEFENDER_W = {"exfil": 5.0, "critical": 2.0, "noncritical": 1.0, "spend": 0.2, "latency": 0.1}
ATTACKER_W = {"exposure": 5.0, "time": 0.01, "detection": 0.1}  # first try (0.5 detection) collapsed the attacker to wait


def defender_reward(exfil_frac: float, crit_fail: float, noncrit_fail: float,
                    spend_norm: float, latency_norm: float) -> tuple[float, dict]:
    c = {"exfil": -DEFENDER_W["exfil"] * exfil_frac,
         "critical": -DEFENDER_W["critical"] * crit_fail,
         "noncritical": -DEFENDER_W["noncritical"] * noncrit_fail,
         "spend": -DEFENDER_W["spend"] * spend_norm,
         "latency": -DEFENDER_W["latency"] * latency_norm}
    return sum(c.values()), c


def attacker_reward(exfil_frac: float, true_pos_alerts: int) -> tuple[float, dict]:
    c = {"exposure": ATTACKER_W["exposure"] * exfil_frac, "time": -ATTACKER_W["time"],
         "detection": -ATTACKER_W["detection"] * true_pos_alerts}
    return sum(c.values()), c
