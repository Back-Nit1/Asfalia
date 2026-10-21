"""MITRE ATT&CK labels for the abstract attacker actions. Labels only: nothing here executes or emulates a technique.

IDs/names/tactics were read from attack.mitre.org technique pages (see docs/mitre_mapping.md). ``fit`` says how closely
the simulator abstraction corresponds: ``partial`` = shares the objective but omits the mechanism.
"""
from __future__ import annotations

ATTACK_PAGE = "https://attack.mitre.org/techniques/{}/"
TECHNIQUES = {   # id -> (name, tactics as listed on the page)
    "T1595": ("Active Scanning", ("Reconnaissance TA0043",)),
    "T1190": ("Exploit Public-Facing Application", ("Initial Access TA0001",)),
    "T1133": ("External Remote Services", ("Persistence TA0003", "Initial Access TA0001")),
    "T1078": ("Valid Accounts", ("Stealth", "Persistence", "Privilege Escalation", "Initial Access")),
    "T1068": ("Exploitation for Privilege Escalation", ("Privilege Escalation TA0004",)),
    "T1021": ("Remote Services", ("Lateral Movement TA0008",)),
    "T1074": ("Data Staged", ("Collection TA0009",)),
    "T1041": ("Exfiltration Over C2 Channel", ("Exfiltration TA0010",)),
}
# simulator action -> (technique ids, fit, what is abstracted away)
ATTACKER_MAP = {
    "discover": (("T1595",), "partial", "reveals reachable nodes; no scan traffic, ports or tooling"),
    "probe": (("T1595",), "partial", "reveals vulnerability flags on a known node; no scanner behavior"),
    "attempt_initial_access": (("T1190", "T1133"), "partial",
                               "succeeds on an `initial_access` flag at an external node; no exploit or VPN protocol"),
    "attempt_privilege_change": (("T1068",), "partial", "gated by a `priv_esc` flag; no exploit mechanics"),
    "move": (("T1021", "T1078"), "partial", "uses reach/trust edges or a held credential; no protocol-level detail"),
    "collect": (("T1074",), "partial", "marks synthetic records as staged on a node"),
    "exfiltrate": (("T1041",), "partial", "moves staged synthetic records out via an egress node; no C2 channel"),
}
UNMAPPED = {"defender actions (no verified D3FEND/ATT&CK mitigation mapping yet)", "benign workload and telemetry noise"}
COVERAGE = "incomplete: 8 techniques cover 7 abstract attacker actions; no sub-techniques, no defender mapping"
