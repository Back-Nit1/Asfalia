# MITRE ATT&CK labels (partial, descriptive only)

Asfalia's attacker actions are abstract, probabilistic state transitions on a synthetic hospital graph. The labels below
say which ATT&CK technique an action is *loosely analogous to*. They do **not** mean the simulator reproduces the
technique, its telemetry, or its detection behavior, and no result here is evidence about real-world attacks.

Coverage is **incomplete**: 8 techniques, no sub-techniques, no defender-side mapping (no verified D3FEND/mitigation IDs
were checked), nothing for benign workload. Source of truth in code: `src/hospital_rl/evaluation/attack_mapping.py`.

| simulator action | technique (page title) | tactic(s) per ATT&CK | fit | abstracted away |
|---|---|---|---|---|
| `discover` | [T1595](https://attack.mitre.org/techniques/T1595/) Active Scanning | Reconnaissance (TA0043) | partial | no scan traffic or tooling |
| `probe` | [T1595](https://attack.mitre.org/techniques/T1595/) Active Scanning | Reconnaissance (TA0043) | partial | vulnerability flag lookup only |
| `attempt_initial_access` | [T1190](https://attack.mitre.org/techniques/T1190/) Exploit Public-Facing Application; [T1133](https://attack.mitre.org/techniques/T1133/) External Remote Services | Initial Access (TA0001); T1133 also Persistence (TA0003) | partial | no exploit or VPN protocol |
| `attempt_privilege_change` | [T1068](https://attack.mitre.org/techniques/T1068/) Exploitation for Privilege Escalation | Privilege Escalation (TA0004) | partial | flag-gated coin flip |
| `move` | [T1021](https://attack.mitre.org/techniques/T1021/) Remote Services; [T1078](https://attack.mitre.org/techniques/T1078/) Valid Accounts | Lateral Movement (TA0008); T1078 spans several tactics | partial | edge/credential rules only |
| `collect` | [T1074](https://attack.mitre.org/techniques/T1074/) Data Staged | Collection (TA0009) | partial | counters on synthetic records |
| `exfiltrate` | [T1041](https://attack.mitre.org/techniques/T1041/) Exfiltration Over C2 Channel | Exfiltration (TA0010) | partial | no C2 channel |

Verification: each ID, name and tactic was read from its attack.mitre.org technique page (ATT&CK v20 content; page
"last modified" dates between 2025-10-24 and 2026-09-11). Re-check against the current ATT&CK release before citing.
Reference: MITRE ATT&CK, <https://attack.mitre.org/>.
