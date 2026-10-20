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
    """Spend early on MFA/logging for critical nodes, then idle."""

    def act(self, view, legal):
        for t in (D.INCREASE_LOGGING, D.ENABLE_MFA):
            for a in legal:
                if a.type == t and view["nodes"][a.node]["asset_type"] in ("ehr", "pacs", "idp"):
                    return a
        return WAIT


class GreedyDefender:
    """Budget-aware: respond to the latest alert on a node, else raise logging on sensitive assets."""

    def act(self, view, legal):
        by = {(a.type, a.node): a for a in legal}
        for _, node, kind in reversed(view["alerts"]):
            if kind == "benign_noise":
                continue
            for t in (D.REVOKE_SESSION, D.ROTATE_CREDENTIAL, D.ISOLATE):
                if (t.value, node) in by and not view["nodes"][node]["isolated"]:
                    return by[(t.value, node)]
        for n, v in view["nodes"].items():
            if v["asset_type"] in ("ehr", "pacs") and (D.INCREASE_LOGGING.value, n) in by:
                return by[(D.INCREASE_LOGGING.value, n)]
        return WAIT


class IsolateEverything:
    def act(self, view, legal):
        for a in legal:
            if a.type == D.ISOLATE:
                return a
        return WAIT
