"""Single transition engine for all three environments.

Tick order: (1) defender action validated, charged, queued as a job; (2) resolution phase, ordered by
spec.tie_policy: completing defender jobs vs the attacker action; (3) workflows; (4) recurring costs;
(5) telemetry; (6) tick += 1. Both roles choose from pre-step views; neither sees the other's current action.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import ATTACKER_TYPES, AttackerAction as A, DefenderAction as D, ScenarioSpec
from . import rewards, telemetry
from .permissions import (Topology, active_sessions, attacker_has_access, blocked, edge_open, reachable_from_external,
                          role_ok, service_up)
from .rng import draw
from .state import Job, Session, WorldState, build_state
from .workflows import run_workflows


@dataclass(frozen=True)
class Action:
    type: str
    node: str | None = None


WAIT = Action("wait")


class IllegalAction(ValueError):
    pass


class World:
    def __init__(self, spec: ScenarioSpec, seed: int):
        self.spec, self.seed = spec, seed
        self.topo = Topology(spec)
        self.st: WorldState = build_state(spec)
        self.total_weight = sum(n.records * n.sensitivity for n in spec.nodes) or 1.0
        self.truth: list[dict] = []        # evaluator-only log
        self.events: list[dict] = []       # role-tagged events of the latest step

    # ------------------------------------------------------------------ legality (role-observable only)
    def in_window(self) -> bool:
        return any(a <= self.st.tick < b for a, b in self.spec.maintenance_windows)

    def legal_defender(self) -> list[Action]:
        st, out = self.st, [WAIT]
        busy = {j.node for j in st.jobs}
        if len(st.jobs) >= self.spec.action_capacity:
            return out
        for nid, node in self.topo.node.items():
            ns = st.nodes[nid]
            if nid in busy:
                continue
            ok = {
                D.ENABLE_MFA: not ns.mfa,
                D.RESTRICT_ROLE: bool(set(node.restrictable_roles) & ns.allowed_roles),
                D.REVOKE_SESSION: True,
                D.ROTATE_CREDENTIAL: bool(node.credentials),
                D.PATCH: node.patchable and self.in_window() and len(ns.patched) < len(node.vulns) + 1,
                D.ISOLATE: not ns.isolated,
                D.RESTORE_CONNECTIVITY: ns.isolated,
                D.INCREASE_LOGGING: ns.logging < 2,
                D.QUARANTINE: not ns.quarantined,
                D.REIMAGE: True,
                D.RESTORE_SERVICE: ns.quarantined or not ns.up,
            }
            for t, allowed in ok.items():
                if allowed and self.spec.costs[t.value][0] <= st.budget_left:
                    out.append(Action(t.value, nid))
        return out

    def legal_attacker(self) -> list[Action]:
        """Depends only on discovered nodes. Whether an action's hidden prerequisites hold is not revealed."""
        out = [WAIT]
        for n in sorted(self.st.know.nodes):
            out += [Action(t.value, n) for t in ATTACKER_TYPES]
        return out

    # ------------------------------------------------------------------ step
    def step(self, att: Action, dfn: Action) -> dict:
        st, spec = self.st, self.spec
        if att not in self.legal_attacker():
            raise IllegalAction(f"attacker action {att} not legal")
        if dfn not in self.legal_defender():
            raise IllegalAction(f"defender action {dfn} not legal")
        self.events = []
        before_w, before_acc = st.m.weight_exfiltrated, st.m.records_accessed
        tp_before = st.m.true_positive_alerts
        spend = self._start_job(dfn)

        resolve = [self._resolve_jobs, lambda: self._attacker(att)]
        if spec.tie_policy == "attacker_first":
            resolve.reverse()
        for r in resolve:
            r()

        cf, nf, lf = run_workflows(self.topo, st, spec, self.seed, self.events)
        spend += self._recurring()
        spend = min(spend, st.budget_left)             # hard cap: never overspend
        st.m.spend += spend
        st.budget_left -= spend
        if st.budget_left <= 1e-9:
            self._lapse_recurring()
        st.m.downtime_node_ticks += sum(1 for n in self.topo.node if not service_up(self.topo, st, n))
        telemetry.benign_noise(st, spec, self.seed)
        released = telemetry.release(st, self.seed)
        for a in released:
            self.events.append({"type": "alert", "node": a.node, "kind": a.kind, "tick": st.tick, "visible": "defender"})
        self._update_containment()

        frac = (st.m.weight_exfiltrated - before_w) / self.total_weight
        r_d, comp_d = rewards.defender_reward(frac, cf, nf, spend / spec.budget, lf)
        r_a, comp_a = rewards.attacker_reward(frac, st.m.true_positive_alerts - tp_before)
        st.tick += 1
        done = st.tick >= spec.horizon
        info = {"reward": {"attacker": r_a, "defender": r_d},
                "components": {"attacker": comp_a, "defender": comp_d},
                "events": self.events, "done": done, "spend": spend}
        self.truth.append({"tick": st.tick - 1, "att": att.__dict__, "dfn": dfn.__dict__, **{k: v for k, v in info.items()
                           if k != "events"}, "events": list(self.events)})
        return info

    # ------------------------------------------------------------------ defender
    def _start_job(self, a: Action) -> float:
        if a.type == "wait":
            return 0.0
        spend, dur, rec = self.spec.costs[a.type]
        downtime = a.type in ("reimage", "patch")
        if downtime:
            self.st.nodes[a.node].down_jobs += 1
        self.st.jobs.append(Job(a.type, a.node, self.st.tick + dur, downtime, rec))
        self.events.append({"type": "defender_action_started", "action": a.type, "node": a.node,
                            "duration": dur, "tick": self.st.tick, "visible": "defender"})
        return spend

    def _recurring(self) -> float:
        # logging/mfa/isolation costs are charged while the control is active, after the one-time job spend
        total = 0.0
        for n, ns in self.st.nodes.items():
            total += self.spec.costs["increase_logging"][2] * max(0, ns.logging - self.topo.node[n].logging)
            total += self.spec.costs["enable_mfa"][2] * (ns.mfa and not self.topo.node[n].mfa)
            total += self.spec.costs["isolate"][2] * ns.isolated
            total += self.spec.costs["quarantine"][2] * ns.quarantined
        return total

    def _lapse_recurring(self):
        """Unfunded recurring controls (extra logging, added MFA) revert to scenario baseline."""
        for n, ns in self.st.nodes.items():
            base = self.topo.node[n]
            if ns.logging > base.logging or (ns.mfa and not base.mfa):
                ns.logging, ns.mfa = min(ns.logging, base.logging), base.mfa
                self.events.append({"type": "control_lapsed", "node": n, "tick": self.st.tick, "visible": "defender"})

    def _resolve_jobs(self):
        st = self.st
        done = [j for j in st.jobs if j.end_tick - 1 <= st.tick]
        st.jobs = [j for j in st.jobs if j not in done]
        for j in done:
            ns = st.nodes[j.node]
            if j.downtime:
                ns.down_jobs -= 1
            self._apply_defense(j, ns)
            self.events.append({"type": "defender_action_done", "action": j.action, "node": j.node,
                                "tick": st.tick, "visible": "defender"})

    def _apply_defense(self, j: Job, ns):
        st, n, t = self.st, j.node, j.action
        node = self.topo.node[n]
        if t == D.ENABLE_MFA:
            ns.mfa = True
        elif t == D.RESTRICT_ROLE:
            ns.allowed_roles -= set(node.restrictable_roles)
        elif t == D.REVOKE_SESSION:
            for s in st.sessions:
                if s.node == n:
                    s.revoked = True           # implant is NOT removed
        elif t == D.ROTATE_CREDENTIAL:
            for c in node.credentials:
                st.cred_valid[c] = False       # sessions using c die via active_sessions()
        elif t == D.PATCH:
            if node.patchable:
                ns.patched |= set(node.vulns) | {"_all"}   # future transitions only; existing access stays
        elif t == D.ISOLATE:
            ns.isolated = True
        elif t == D.RESTORE_CONNECTIVITY:
            ns.isolated = False
        elif t == D.INCREASE_LOGGING:
            ns.logging = min(2, ns.logging + 1)
        elif t == D.QUARANTINE:
            ns.quarantined = True              # compromise persists but is inert while quarantined
        elif t == D.REIMAGE:
            ns.implant = False                 # residual: held credentials and vulns remain
            for s in st.sessions:
                if s.node == n:
                    s.revoked = True
            ns.staged.clear()
        elif t == D.RESTORE_SERVICE:
            ns.quarantined = False
            ns.up = True
            if ns.implant:
                self.events.append({"type": "restore_with_residual_risk", "node": n, "tick": st.tick,
                                    "visible": "evaluator"})

    # ------------------------------------------------------------------ attacker
    def _unpatched(self, n: str, flag: str) -> bool:
        return flag in self.topo.node[n].vulns and flag not in self.st.nodes[n].patched

    def _hit(self, stream: str, p: float, a: Action) -> bool:
        return draw(self.seed, stream, self.st.tick, a.type, a.node) < p

    def _attacker(self, a: Action):
        if a.type == "wait":
            return
        st, spec, n = self.st, self.spec, a.node
        ok = False
        if a.type == A.DISCOVER:
            ok = (attacker_has_access(st, n) or self.topo.node[n].external) and not blocked(st, n) and \
                self._hit("hit", spec.p["discover"], a)
            if ok:
                for m in self.topo.out[n]["reach"] | self.topo.out[n]["trust"]:
                    st.know.nodes.add(m)
                    st.know.edges.add((n, m))
        elif a.type == A.PROBE:
            ok = self._attack_reachable(n) and self._hit("hit", spec.p["probe"], a)
            if ok:
                st.know.vulns.setdefault(n, set()).update(
                    v for v in self.topo.node[n].vulns if v not in st.nodes[n].patched)
        elif a.type == A.INITIAL_ACCESS:
            ok = self._attack_reachable(n) and self._unpatched(n, "initial_access") and \
                self._hit("hit", spec.p["initial_access"], a)
            if ok:
                st.sessions.append(Session(n, "service", None))
                st.know.access.add(n)
        elif a.type == A.PRIVILEGE_CHANGE:
            ok = attacker_has_access(st, n) and self._unpatched(n, "priv_esc") and self._hit("hit", spec.p["priv"], a)
            if ok:
                st.nodes[n].implant = True
                for c in self.topo.node[n].credentials:
                    if st.cred_valid[c]:
                        st.know.creds.add(c)
        elif a.type == A.MOVE:
            ok = self._move(a)
        elif a.type == A.COLLECT:
            ok = self._collect(a)
        elif a.type == A.EXFILTRATE:
            ok = self._exfil(a)
        if ok and a.type == A.MOVE:
            st.know.access.add(n)
        if ok and a.type == A.COLLECT:
            st.know.staged[n] = len(st.nodes[n].staged)
        if ok and a.type == A.EXFILTRATE:
            st.know.staged[n] = len(st.nodes[n].staged)
        if not ok and a.type in (A.COLLECT, A.EXFILTRATE, A.PRIVILEGE_CHANGE) and not attacker_has_access(st, n):
            st.know.access.discard(n)              # auth error observed on use
            st.know.staged.pop(n, None)
        self.truth_event(a, ok)
        if ok and a.type in telemetry.KIND:
            telemetry.emit_for_true_event(st, spec, self.seed, a.type, n)
        if ok and st.m.first_access_tick is None and a.type in (A.INITIAL_ACCESS, A.MOVE):
            st.m.first_access_tick = st.tick
        self.events.append({"type": "attacker_outcome", "action": a.type, "node": n, "success": ok,
                            "tick": st.tick, "visible": "attacker"})

    def truth_event(self, a: Action, ok: bool):
        self.events.append({"type": "attacker_truth", "action": a.type, "node": a.node, "success": ok,
                            "tick": self.st.tick, "visible": "evaluator"})

    def _attack_reachable(self, n: str) -> bool:
        st = self.st
        if blocked(st, n):
            return False
        if self.topo.node[n].external:
            return True
        return any(attacker_has_access(st, s) and n in self.topo.out[s]["reach"] and edge_open(st, s, n)
                   for s in self.topo.node)

    def _move(self, a: Action) -> bool:
        st, n, spec = self.st, a.node, self.spec
        if blocked(st, n):
            return False
        sources = [s for s in self.topo.node if attacker_has_access(st, s) and n in self.topo.out[s]["reach"]
                   and edge_open(st, s, n)]
        if not sources:
            return False
        for cid in sorted(st.know.creds):          # credentialed lateral movement: needs authorization, not just reach
            c = self.topo.cred[cid]
            role = self.topo.role_of[c.identity]
            if st.cred_valid[cid] and n in c.scope and role_ok(st, n, role):
                p = spec.p["mfa_bypass"] if st.nodes[n].mfa else spec.p["move_cred"]
                if self._hit("cred", p, a):
                    st.sessions.append(Session(n, role, cid))
                    return True
                return False
        if any(n in self.topo.out[s]["trust"] for s in sources) and self._hit("trust", spec.p["move_trust"], a):
            st.sessions.append(Session(n, "service", None))
            return True
        if self._unpatched(n, "lateral") and self._hit("lat", spec.p["move_vuln"], a):
            st.sessions.append(Session(n, "service", None))
            return True
        return False

    def _readable(self, n: str) -> bool:
        st = self.st
        if st.nodes[n].implant:
            return True                            # admin on host: encryption at rest does not help
        return any(role_ok(st, n, s.role) for s in active_sessions(st, n))

    def _collect(self, a: Action) -> bool:
        st, n = self.st, a.node
        if not attacker_has_access(st, n) or not service_up(self.topo, st, n) or not self._readable(n):
            return False
        node, ns = self.topo.node[n], st.nodes[n]
        todo = [f"{n}:r{i}" for i in range(node.records) if f"{n}:r{i}" not in ns.taken][:self.spec.collect_per_tick]
        ns.taken.update(todo)
        ns.staged.update(todo)
        st.m.records_accessed += len(todo)
        return bool(todo)

    def _exfil(self, a: Action) -> bool:
        st, n = self.st, a.node
        ns = st.nodes[n]
        if not attacker_has_access(st, n) or not ns.staged or not reachable_from_external(self.topo, st, n):
            return False
        batch = sorted(ns.staged)[:self.spec.exfil_per_tick]
        ns.staged.difference_update(batch)         # a record leaves staging once; unique by construction
        st.m.records_exfiltrated += len(batch)
        st.m.weight_exfiltrated += len(batch) * self.topo.node[n].sensitivity
        return True

    def _update_containment(self):
        st = self.st
        live = any(attacker_has_access(st, n) for n in self.topo.node)
        if st.m.first_access_tick is not None and not live and st.m.containment_tick is None:
            st.m.containment_tick = st.tick

    # ------------------------------------------------------------------ role views (no hidden state)
    def view(self, role: str) -> dict:
        st = self.st
        if role == "attacker":
            k = st.know
            return {"tick": st.tick, "known_nodes": sorted(k.nodes), "known_edges": sorted(k.edges),
                    "known_vulns": {n: sorted(v) for n, v in k.vulns.items()}, "held_creds": sorted(k.creds),
                    "access": sorted(k.access), "staged": {n: c for n, c in k.staged.items() if c},
                    "node_types": {n: self.topo.node[n].asset_type for n in k.nodes}}
        return {"tick": st.tick, "budget_left": st.budget_left, "in_window": self.in_window(),
                "jobs": [(j.action, j.node, j.end_tick) for j in st.jobs],
                "overrides_left": st.overrides_left,
                "nodes": {n: {"asset_type": nd.asset_type, "zone": nd.zone, "patchable": nd.patchable,
                              "logging": st.nodes[n].logging, "mfa": st.nodes[n].mfa,
                              "isolated": st.nodes[n].isolated, "quarantined": st.nodes[n].quarantined,
                              "service_up": service_up(self.topo, st, n)} for n, nd in self.topo.node.items()},
                "alerts": [(a.visible_tick, a.node, a.kind) for a in st.visible_alerts[-20:]]}
