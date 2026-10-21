# Related systems and limits of the evidence

## Related systems
- **CyberBattleSim** (Microsoft) is an open-source, graph-based, abstract attacker-vs-defender simulation for
  reinforcement learning. Asfalia is the same broad kind of tool: abstract state transitions, no real exploit
  execution. Differences that matter here: a hospital-specific asset/workflow model with clinical-availability
  constraints, a trainable defender alongside the attacker, typed role-local observations with action masks, and a
  split between actor-visible trajectories and evaluator truth. We make no claim that these are new in the field.
- ATT&CK labels for attacker actions are in `docs/mitre_mapping.md`; they are descriptive tags, not fidelity claims.

## Limits
- All data is synthetic. Topologies, vulnerabilities, workloads, alert rates and costs are generated or hand-set, and
  rewards are uncalibrated. Nothing is validated against a real hospital or real incident data.
- Scripted attackers and a small family of generated worlds define the test; strong results do not imply robustness to
  an unseen adaptive adversary or a different environment.
- Several benchmark dimensions (telemetry-loss level, full-observability diagnostic) are defined but not yet runnable
  in the engine; see `docs/requests/agent-F.md`.
- Smoke and few-seed outputs are labeled INSUFFICIENT by the report generator; confidence intervals with n < 5 seeds
  are shown for transparency, not as a basis for claims.
- Passive public traces must not be used to claim remediation effectiveness: they contain no counterfactual.
