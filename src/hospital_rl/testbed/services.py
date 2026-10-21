"""Synthetic services, typed-action surface, audit log and isolation policy."""
from __future__ import annotations

import random
import re
import sqlite3
from dataclasses import dataclass, field

from hospital_rl.schemas import AttackerAction, DefenderAction, ScenarioSpec

MAX_RECORDS = 200  # per node: hard bound on generated rows
SCHEMA = """
CREATE TABLE patient (id TEXT PRIMARY KEY, node TEXT NOT NULL, dept TEXT NOT NULL);
CREATE TABLE imaging_study (id TEXT PRIMARY KEY, node TEXT NOT NULL, modality TEXT NOT NULL, size_kb INTEGER NOT NULL);
CREATE TABLE identity (account TEXT PRIMARY KEY, node TEXT NOT NULL, mfa INTEGER NOT NULL, revoked INTEGER NOT NULL);
"""
DEPTS, MODALITIES = ("cardiology", "oncology", "radiology", "emergency"), ("CT", "MR", "XR", "US")
TYPES = {"attacker": {a.value for a in AttackerAction}, "defender": {a.value for a in DefenderAction}}
SECRET_ENV = re.compile(r"DAYTONA|MODAL|API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AWS_|GITHUB|ANTHROPIC|OPENAI", re.I)
INTERNAL_HOSTS = frozenset({"testbed-db", "testbed-idp", "testbed-pacs"})


class ActionRejected(ValueError):
    pass


@dataclass(frozen=True)
class IsolationPolicy:
    """Declared sandbox posture for the testbed bundle; ``check`` is what tests and the deploy smoke assert."""
    internal_only: bool = True
    cpu: float = 1.0
    memory_mb: int = 1024
    max_records: int = MAX_RECORDS
    deny_provider_credentials: bool = True

    def check(self) -> None:
        if not (self.internal_only and self.deny_provider_credentials):
            raise ValueError("testbed must be internal-only and deny provider credentials")
        if self.max_records > MAX_RECORDS:
            raise ValueError(f"max_records {self.max_records} > {MAX_RECORDS}")

    @staticmethod
    def egress(host: str) -> None:
        if host not in INTERNAL_HOSTS:
            raise PermissionError(f"egress to {host!r} denied (internal-only testbed)")

    @staticmethod
    def actor_env(environ: dict[str, str]) -> dict[str, str]:
        """The environment a policy process may see: no provider credentials or secrets."""
        return {k: v for k, v in environ.items() if not SECRET_ENV.search(k)}


class AuditLog:
    """Evaluator-side storage in a separate database; the actor API holds no reference to it."""

    def __init__(self):
        self._db = sqlite3.connect(":memory:")
        self._db.execute("CREATE TABLE audit (seq INTEGER PRIMARY KEY, tick INTEGER, source TEXT, kind TEXT, node TEXT)")

    def write(self, tick: int, source: str, kind: str, node: str | None) -> None:
        self._db.execute("INSERT INTO audit (tick, source, kind, node) VALUES (?,?,?,?)", (tick, source, kind, node))

    def rows(self) -> list[tuple]:
        return self._db.execute("SELECT tick, source, kind, node FROM audit ORDER BY seq").fetchall()


@dataclass
class Svc:
    up: bool = True
    isolated: bool = False
    quarantined: bool = False
    mfa: bool = False
    logging: int = 0
    patched: bool = False
    restricted: bool = False
    staged: int = 0
    taken: int = 0


class SyntheticServices:
    """Deterministic for (spec, seed): same seed -> same rows; ``reset`` rebuilds everything from scratch."""

    def __init__(self, spec: ScenarioSpec, seed: int, audit: AuditLog | None = None):
        self.spec, self.seed = spec, seed
        self.audit = audit or AuditLog()
        self.reset()

    def reset(self) -> None:
        rng = random.Random(self.seed)
        self.db = sqlite3.connect(":memory:")
        self.db.executescript(SCHEMA)
        self.svc = {n.id: Svc(mfa=n.mfa, logging=n.logging) for n in self.spec.nodes}
        self.tick, self.exfiltrated = 0, 0
        for n in self.spec.nodes:
            for i in range(min(n.records, MAX_RECORDS)):
                if n.asset_type == "ehr":
                    self.db.execute("INSERT INTO patient VALUES (?,?,?)", (f"SYN-{n.id}-{i:04d}", n.id, rng.choice(DEPTS)))
                elif n.asset_type == "pacs":
                    self.db.execute("INSERT INTO imaging_study VALUES (?,?,?,?)", (f"SYN-IMG-{n.id}-{i:04d}", n.id, rng.choice(MODALITIES), rng.randrange(10, 500)))
            self.db.execute("INSERT INTO identity VALUES (?,?,?,0)", (f"svc-{n.id}", n.id, int(n.mfa)))
        self.audit.write(0, "testbed", "reset", None)

    # --- evaluator-side views -------------------------------------------------------------------------------
    def row_counts(self) -> dict[str, int]:
        return {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("patient", "imaging_study", "identity")}

    def records_at(self, node: str) -> int:
        q = "SELECT (SELECT COUNT(*) FROM patient WHERE node=?) + (SELECT COUNT(*) FROM imaging_study WHERE node=?)"
        return self.db.execute(q, (node, node)).fetchone()[0]

    def workload(self) -> int:
        """Benign, bounded, deterministic background activity: reads that only appear in the audit log."""
        rng = random.Random(self.seed * 1000 + self.tick)
        up = [n for n, s in self.svc.items() if s.up and not s.isolated and not s.quarantined]
        reads = rng.randrange(1, 4) if up else 0
        for _ in range(reads):
            self.audit.write(self.tick, "benign", "read", rng.choice(up))
        return reads

    # --- the only mutation path: allowlisted typed actions -----------------------------------------------------
    def apply(self, role: str, type: str, node: str | None = None) -> bool:
        if type not in TYPES.get(role, ()):
            raise ActionRejected(f"{role} action {type!r} is not in the allowlist")
        if type != "wait" and node not in self.svc:
            raise ActionRejected(f"unknown node {node!r}")
        s, ok = self.svc.get(node), True
        if role == "defender":
            if type == "enable_mfa":
                s.mfa = True
                self.db.execute("UPDATE identity SET mfa=1 WHERE node=?", (node,))
            elif type == "restrict_role":
                s.restricted = True
            elif type == "revoke_session":
                self.db.execute("UPDATE identity SET revoked=1 WHERE node=?", (node,))
            elif type == "patch":
                s.patched = True
            elif type == "isolate":
                s.isolated = True
            elif type == "restore_connectivity":
                s.isolated = False
            elif type == "increase_logging":
                s.logging = min(2, s.logging + 1)
            elif type == "quarantine":
                s.quarantined = True
            elif type == "reimage":
                s.staged = 0
            elif type == "restore_service":
                s.quarantined, s.up = False, True
        elif type == "collect":
            ok = s.up and not s.isolated and not s.quarantined
            if ok:
                s.staged = min(self.records_at(node) - s.taken, s.staged + self.spec.collect_per_tick)
                s.taken = max(s.taken, s.staged)
        elif type == "exfiltrate":
            ok = s.up and not s.isolated and s.staged > 0
            if ok:
                n = min(s.staged, self.spec.exfil_per_tick)
                s.staged -= n
                self.exfiltrated += n
        self.audit.write(self.tick, role, type + ("" if ok else ":failed"), node)
        return ok

    def step(self) -> None:
        self.workload()
        self.tick += 1


class ActorAPI:
    """What a policy process is handed: typed actions in, a small role view out. No SQL, shell, network, audit access."""

    def __init__(self, services: SyntheticServices, role: str):
        self.__svc, self.__role = services, role

    def act(self, type: str, node: str | None = None) -> bool:
        return self.__svc.apply(self.__role, type, node)

    def observe(self) -> dict:
        return {"tick": self.__svc.tick,
                "services": {n: {"up": s.up and not s.isolated and not s.quarantined} for n, s in self.__svc.svc.items()}}
