"""Scripted/baseline policies operating only on role views and legal action lists."""
from __future__ import annotations

import random

from ..engine.transitions import WAIT, Action
from ..schemas import AttackerAction as A, DefenderAction as D


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.r = random.Random(seed)

    def act(self, view, legal):
        return self.r.choice(legal)


class Noop:
    def act(self, view, legal):
        return WAIT


class ScriptedAttacker:
    """Kill chain with retries. Stateful per episode (create a fresh instance per episode)."""

    def __init__(self, target_types=("ehr", "pacs", "billing")):
        self.targets = target_types
        self.probed: set[str] = set()

    def act(self, view, legal):
        have = {(a.type, a.node) for a in legal}
        acc, known = view["access"], view["known_nodes"]
        edges = view["known_edges"]
        mk = lambda t, n: Action(t.value, n)
        for n in acc:
            if view["staged"].get(n) and (A.EXFILTRATE, n) in have:
                return mk(A.EXFILTRATE, n)
        for n in acc:
            if view["node_types"].get(n) in self.targets and (A.COLLECT, n) in have:
                return mk(A.COLLECT, n)
        for n in acc:
            if n not in self.probed and (A.PROBE, n) in have:
                self.probed.add(n)
                return mk(A.PROBE, n)
        for n in acc:
            if "priv_esc" in view["known_vulns"].get(n, ()) and not view["held_creds"]:
                return mk(A.PRIVILEGE_CHANGE, n)
        for n in acc:
            if (A.DISCOVER, n) in have and not any(e[0] == n for e in edges):
                return mk(A.DISCOVER, n)
        goals = [n for n in known if view["node_types"].get(n) in self.targets and n not in acc]
        nbrs = [n for (s, n) in edges if s in acc and n not in acc]
        for n in goals + nbrs:
            if (A.MOVE, n) in have:
                return mk(A.MOVE, n)
        for n in known:
            if not acc and (A.DISCOVER, n) in have and not any(e[0] == n for e in edges):
                return mk(A.DISCOVER, n)
            if not acc and (A.INITIAL_ACCESS, n) in have:
                return mk(A.INITIAL_ACCESS, n)
        return WAIT


class StaticHardening:
    """Preventive posture: MFA and logging on critical assets, then patch exposed/patchable nodes in priority order."""
    PRIORITY = ("vpn", "workstation", "ehr", "pacs", "k8s_service", "billing", "idp")

    def act(self, view, legal):
        by = {(a.type, a.node): a for a in legal}
        crit = [n for n, v in view["nodes"].items() if v["asset_type"] in ("ehr", "pacs", "idp")]
        done = getattr(self, "_patched", set())
        self._patched = done
        for kind in self.PRIORITY[:2]:                                  # close the entry points first
            for n, v in sorted(view["nodes"].items()):
                if v["asset_type"] == kind and n not in done and (D.PATCH.value, n) in by:
                    done.add(n)
                    return by[(D.PATCH.value, n)]
        for t in (D.INCREASE_LOGGING, D.ENABLE_MFA):
            for n in crit:
                if (t.value, n) in by and not (t == D.ENABLE_MFA and view["nodes"][n]["asset_type"] == "ehr"):
                    return by[(t.value, n)]
        for kind in self.PRIORITY:
            for n, v in sorted(view["nodes"].items()):
                if v["asset_type"] == kind and n not in done and (D.PATCH.value, n) in by:
                    done.add(n)
                    return by[(D.PATCH.value, n)]
        return WAIT


class GreedyDefender:
    """Reactive, budget-aware: escalate on an alerted node (quarantine -> reimage -> restore). Alerts include
    false positives, so over-reaction costs care availability."""

    def act(self, view, legal):
        by = {(a.type, a.node): a for a in legal}
        busy = {j[1] for j in view["jobs"]}
        for n, v in sorted(view["nodes"].items()):                      # finish recovery first
            if v["quarantined"] and (D.REIMAGE.value, n) in by and n in getattr(self, "q", set()):
                self.q.discard(n)
                self.r = getattr(self, "r", set()) | {n}
                return by[(D.REIMAGE.value, n)]
            if n in getattr(self, "r", set()) and n not in busy and (D.RESTORE_SERVICE.value, n) in by:
                self.r.discard(n)
                return by[(D.RESTORE_SERVICE.value, n)]
        counts: dict[str, int] = {}
        for t, node, _kind in view["alerts"]:
            if t >= view["tick"] - 6:
                counts[node] = counts.get(node, 0) + 1
        for node, c in sorted(counts.items(), key=lambda x: -x[1]):         # act on repeated evidence only
            if c >= 2 and (D.QUARANTINE.value, node) in by and not view["nodes"][node]["quarantined"]:
                self.q = getattr(self, "q", set()) | {node}
                return by[(D.QUARANTINE.value, node)]
        for n, v in sorted(view["nodes"].items()):
            if v["asset_type"] in ("ehr", "pacs") and (D.INCREASE_LOGGING.value, n) in by:
                return by[(D.INCREASE_LOGGING.value, n)]
        return WAIT


class IsolateEverything:
    def act(self, view, legal):
        for a in legal:
            if a.type == D.ISOLATE:
                return a
        return WAIT
