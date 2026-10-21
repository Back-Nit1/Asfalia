"""Graph -> testbed adapter: replay a graph episode's action sequence into the synthetic services and report where the
two disagree. This checks the adapter's *semantics* only; agreement says nothing about real-hospital validity."""
from __future__ import annotations

from hospital_rl.engine.transitions import Action, World
from hospital_rl.policies.scripted import GreedyDefender, ScriptedAttacker
from hospital_rl.schemas import ScenarioSpec
from hospital_rl.testbed.services import SyntheticServices

FLAGS = ("isolated", "quarantined", "mfa", "logging")
CAVEAT = "adapter semantics only: not evidence of real-hospital validity or remediation effectiveness"


def graph_episode(spec: ScenarioSpec, seed: int, ticks: int | None = None) -> tuple[World, list[tuple[Action, Action]]]:
    w, att, dfn, acts = World(spec, seed), ScriptedAttacker(), GreedyDefender(), []
    for _ in range(ticks or spec.horizon):
        a = att.act(w.view("attacker"), w.legal_attacker())
        d = dfn.act(w.view("defender"), w.legal_defender())
        acts.append((a, d))
        if w.step(a, d)["done"]:
            break
    return w, acts


def compare(world: World, services: SyntheticServices) -> list[dict]:
    """Per-node flag disagreements. A defender job still running in the graph (duration > 1) is reported as
    ``pending_job`` rather than hidden: the testbed applies actions instantly, the graph after ``costs[type][1]`` ticks."""
    pending = {j.node for j in world.st.jobs}
    out = []
    for n, ns in world.st.nodes.items():
        for f in FLAGS:
            g, t = getattr(ns, f), getattr(services.svc[n], f)
            if g != t:
                out.append({"node": n, "flag": f, "graph": g, "testbed": t, "cause": "pending_job" if n in pending else "mismatch"})
    return out


def replay(spec: ScenarioSpec, seed: int, ticks: int | None = None) -> dict:
    world, acts = graph_episode(spec, seed, ticks)
    sv = SyntheticServices(spec, seed)
    for a, d in acts:
        sv.apply("attacker", a.type, a.node)
        sv.apply("defender", d.type, d.node)
        sv.step()
    mism = compare(world, sv)
    return {"ticks": len(acts), "mismatches": mism, "unexplained": [m for m in mism if m["cause"] == "mismatch"], "caveat": CAVEAT}
