"""Versioned static scenario schema and the typed action catalog."""
from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

import yaml
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
CATALOG_VERSION = 1
MAX_NODES = 64


class AttackerAction(StrEnum):
    WAIT = "wait"
    DISCOVER = "discover"
    PROBE = "probe"
    INITIAL_ACCESS = "attempt_initial_access"
    PRIVILEGE_CHANGE = "attempt_privilege_change"
    MOVE = "move"
    COLLECT = "collect"
    EXFILTRATE = "exfiltrate"


class DefenderAction(StrEnum):
    WAIT = "wait"
    ENABLE_MFA = "enable_mfa"
    RESTRICT_ROLE = "restrict_role"
    REVOKE_SESSION = "revoke_session"
    ROTATE_CREDENTIAL = "rotate_credential"
    PATCH = "patch"
    ISOLATE = "isolate"
    RESTORE_CONNECTIVITY = "restore_connectivity"
    INCREASE_LOGGING = "increase_logging"
    QUARANTINE = "quarantine"
    REIMAGE = "reimage"
    RESTORE_SERVICE = "restore_service"


ATTACKER_TYPES = [a for a in AttackerAction if a is not AttackerAction.WAIT]
DEFENDER_TYPES = [a for a in DefenderAction if a is not DefenderAction.WAIT]


def catalog_hash() -> str:
    blob = json.dumps({"v": CATALOG_VERSION, "a": [a.value for a in ATTACKER_TYPES],
                       "d": [d.value for d in DEFENDER_TYPES], "max_nodes": MAX_NODES})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class NodeSpec(BaseModel):
    id: str
    asset_type: Literal["ehr", "pacs", "workstation", "billing", "idp", "vpn", "k8s_service", "device"]
    zone: str
    records: int = 0                       # synthetic record count
    sensitivity: float = 0.0               # per-record weight
    patchable: bool = True
    vulns: list[str] = Field(default_factory=list)  # abstract flags: initial_access, priv_esc, lateral
    logging: int = 0                       # 0..2
    encrypted: bool = False                # at rest; does NOT stop an authorized session
    mfa: bool = False
    credentials: list[str] = Field(default_factory=list)  # credential ids stored on node
    allowed_roles: list[str] = Field(default_factory=list)  # empty = no role check
    restrictable_roles: list[str] = Field(default_factory=list)
    external: bool = False                 # attacker-reachable entry / egress point


class EdgeSpec(BaseModel):
    src: str
    dst: str
    kind: Literal["reach", "trust", "depends"]


class IdentitySpec(BaseModel):
    id: str
    role: str


class CredentialSpec(BaseModel):
    id: str
    identity: str
    scope: list[str]                       # node ids where it authenticates


class WorkflowSpec(BaseModel):
    id: str
    kind: Literal["chart_lookup", "image_retrieval", "device_telemetry", "billing_sync", "emergency_access"]
    rate: float                            # expected requests per tick
    path: list[str]                        # client ... service
    role: str
    deadline: int                          # max latency ticks
    critical: bool = True
    emergency: bool = False


class ScenarioSpec(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    identities: list[IdentitySpec] = Field(default_factory=list)
    credentials: list[CredentialSpec] = Field(default_factory=list)
    workflows: list[WorkflowSpec] = Field(default_factory=list)
    horizon: int = 40
    budget: float = 30.0
    action_capacity: int = 2
    maintenance_windows: list[tuple[int, int]] = Field(default_factory=lambda: [(0, 10 ** 9)])
    tie_policy: Literal["defender_first", "attacker_first"] = "defender_first"
    emergency_overrides: int = 3
    mfa_latency: int = 2
    exfil_per_tick: int = 3
    collect_per_tick: int = 3
    critical_success_threshold: float = 0.95
    latency_norm: float = 4.0
    # defender action tables: type -> (spend, duration, recurring_per_tick)
    costs: dict[str, tuple[float, int, float]] = Field(default_factory=lambda: {
        "enable_mfa": (3, 1, 0.05), "restrict_role": (2, 1, 0.0), "revoke_session": (1, 1, 0.0),
        "rotate_credential": (2, 2, 0.0), "patch": (4, 3, 0.0), "isolate": (1, 1, 0.1),
        "restore_connectivity": (1, 1, 0.0), "increase_logging": (2, 1, 0.25),
        "quarantine": (2, 1, 0.1), "reimage": (6, 5, 0.0), "restore_service": (2, 2, 0.0)})
    # attacker success probabilities (abstract, uncalibrated)
    p: dict[str, float] = Field(default_factory=lambda: {
        "discover": 0.9, "probe": 0.8, "initial_access": 0.7, "priv": 0.6, "move_vuln": 0.6,
        "move_trust": 0.8, "move_cred": 0.9, "mfa_bypass": 0.1})
    detect_by_logging: tuple[float, float, float] = (0.1, 0.4, 0.8)
    false_positive_by_logging: tuple[float, float, float] = (0.005, 0.02, 0.05)

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def load_scenario(path: str) -> ScenarioSpec:
    with open(path) as f:
        return ScenarioSpec.model_validate(yaml.safe_load(f))
