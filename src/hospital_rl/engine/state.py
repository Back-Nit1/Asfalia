"""Mutable world state. Everything here is evaluator truth; role views are built in transitions.view()."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import ScenarioSpec


@dataclass
class NodeState:
    up: bool = True
    quarantined: bool = False
    isolated: bool = False
    down_jobs: int = 0                     # running jobs that take the service down
    logging: int = 0
    mfa: bool = False
    patched: set[str] = field(default_factory=set)
    allowed_roles: set[str] = field(default_factory=set)
    implant: bool = False                  # attacker persistence; survives patching and session revocation
    staged: set[str] = field(default_factory=set)       # collected, not yet exfiltrated
    taken: set[str] = field(default_factory=set)        # all records ever collected


@dataclass
class Session:
    node: str
    role: str
    cred: str | None
    revoked: bool = False


@dataclass
class Job:
    action: str
    node: str
    end_tick: int          # resolves in the resolution phase of tick end_tick - 1
    downtime: bool
    recurring: float


@dataclass
class Alert:
    visible_tick: int
    node: str
    kind: str              # auth_anomaly | flow_anomaly | process_anomaly | benign_noise
    true_positive: bool    # evaluator-only label


@dataclass
class Knowledge:
    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    vulns: dict[str, set[str]] = field(default_factory=dict)
    creds: set[str] = field(default_factory=set)
    access: set[str] = field(default_factory=set)      # believed access (updated on success / observed auth error)
    staged: dict[str, int] = field(default_factory=dict)


@dataclass
class Metrics:
    records_accessed: int = 0
    records_exfiltrated: int = 0
    weight_exfiltrated: float = 0.0
    spend: float = 0.0
    wf_total: dict[str, int] = field(default_factory=lambda: {"critical": 0, "noncritical": 0})
    wf_failed: dict[str, int] = field(default_factory=lambda: {"critical": 0, "noncritical": 0})
    added_latency: float = 0.0
    downtime_node_ticks: int = 0
    false_positive_alerts: int = 0
    true_positive_alerts: int = 0
    first_access_tick: int | None = None
    containment_tick: int | None = None    # None = censored / not contained
    emergency_overrides_used: int = 0


@dataclass
class WorldState:
    tick: int = 0
    nodes: dict[str, NodeState] = field(default_factory=dict)
    sessions: list[Session] = field(default_factory=list)
    cred_valid: dict[str, bool] = field(default_factory=dict)
    jobs: list[Job] = field(default_factory=list)
    pending_alerts: list[Alert] = field(default_factory=list)
    visible_alerts: list[Alert] = field(default_factory=list)   # released to defender, newest last
    know: Knowledge = field(default_factory=Knowledge)
    budget_left: float = 0.0
    overrides_left: int = 0
    m: Metrics = field(default_factory=Metrics)


def build_state(spec: ScenarioSpec) -> WorldState:
    s = WorldState(budget_left=spec.budget, overrides_left=spec.emergency_overrides)
    for n in spec.nodes:
        s.nodes[n.id] = NodeState(logging=n.logging, mfa=n.mfa, allowed_roles=set(n.allowed_roles))
    s.cred_valid = {c.id: True for c in spec.credentials}
    s.know.nodes = {n.id for n in spec.nodes if n.external}
    return s
