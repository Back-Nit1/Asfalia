"""Reachability, effective availability, and role/credential checks. Network access != authorization."""
from __future__ import annotations

from ..schemas import ScenarioSpec
from .state import WorldState


class Topology:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.node = {n.id: n for n in spec.nodes}
        self.cred = {c.id: c for c in spec.credentials}
        self.role_of = {i.id: i.role for i in spec.identities}
        self.out = {n.id: {"reach": set(), "trust": set(), "depends": set()} for n in spec.nodes}
        for e in spec.edges:
            self.out[e.src][e.kind].add(e.dst)


def blocked(st: WorldState, n: str) -> bool:
    ns = st.nodes[n]
    return ns.isolated or ns.quarantined


def edge_open(st: WorldState, src: str, dst: str) -> bool:
    return not blocked(st, src) and not blocked(st, dst)


def service_up(topo: Topology, st: WorldState, n: str, _seen: frozenset = frozenset()) -> bool:
    if n in _seen:
        return True  # dependency cycle: do not recurse
    ns = st.nodes[n]
    if not ns.up or ns.quarantined or ns.down_jobs > 0:
        return False
    return all(service_up(topo, st, d, _seen | {n}) for d in topo.out[n]["depends"])


def reachable_from_external(topo: Topology, st: WorldState, n: str) -> bool:
    """Egress/ingress: BFS over open reach edges to any external node."""
    seen, stack = {n}, [n]
    while stack:
        cur = stack.pop()
        if topo.node[cur].external and not blocked(st, cur):
            return True
        for nxt in topo.out[cur]["reach"]:
            if nxt not in seen and edge_open(st, cur, nxt):
                seen.add(nxt)
                stack.append(nxt)
    return False


def role_ok(st: WorldState, n: str, role: str) -> bool:
    allowed = st.nodes[n].allowed_roles
    return not allowed or role in allowed


def active_sessions(st: WorldState, n: str):
    return [s for s in st.sessions if s.node == n and not s.revoked and
            (s.cred is None or st.cred_valid[s.cred])]


def attacker_has_access(st: WorldState, n: str) -> bool:
    return not blocked(st, n) and (st.nodes[n].implant or bool(active_sessions(st, n)))
