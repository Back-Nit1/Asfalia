"""Episode loading and per-perspective projection.

`parse_episode` is the single adapter between the lead's episode export (`runner.run_episode` output, plus an optional
`topology` block) and the API. When the real export format changes, only this function changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Perspective = Literal["attacker", "defender", "evaluator"]
# which `event["visible"]` scopes each perspective may receive
ALLOWED: dict[str, set[str]] = {"attacker": {"attacker"}, "defender": {"defender"}, "evaluator": {"attacker", "defender", "evaluator"}}
RECON, FOOTHOLD = {"discover", "probe"}, {"attempt_initial_access", "move", "attempt_privilege_change"}
ISOLATING, UNISOLATING = {"isolate", "quarantine"}, {"restore_connectivity", "restore_service"}
DEGRADING = {"patch", "reimage"}


@dataclass
class Episode:
    id: str
    label: str                 # "FIXTURE" or "RECORDED"
    scenario: str
    seed: int | None
    nodes: list[dict]          # {id, asset_type, zone}
    edges: list[dict]          # {src, dst, kind}
    steps: list[dict]
    final_metrics: dict        # evaluator-only
    has_topology: bool


def parse_episode(raw: dict, episode_id: str) -> Episode:
    topo = raw.get("topology") or {}
    return Episode(id=episode_id, label="DEBUG" if raw.get("debug") else "FIXTURE" if raw.get("fixture") else "RECORDED", scenario=raw.get("scenario", ""),
                   seed=raw.get("seed"), nodes=topo.get("nodes", []), edges=topo.get("edges", []),
                   steps=raw["steps"], final_metrics=raw.get("metrics", {}), has_topology=bool(topo.get("nodes")))


class Library:
    def __init__(self, *dirs: Path):
        self.episodes: dict[str, Episode] = {}
        self._frames: dict[tuple[str, str], list[dict]] = {}
        self.dirs = [Path(d) for d in dirs]
        self.rescan()

    def rescan(self) -> None:
        """Pick up episode exports written after startup (completed evaluations). Known ids are never replaced."""
        for d in self.dirs:
            for p in sorted(d.rglob("*.json")):
                if p.stem in self.episodes:
                    continue
                try:
                    self.episodes[p.stem] = parse_episode(json.loads(p.read_text()), p.stem)
                except (ValueError, KeyError, OSError):
                    continue    # not an episode export (or still being written)

    def frames(self, ep: Episode, perspective: str) -> list[dict]:
        key = (ep.id, perspective)
        if key not in self._frames:
            self._frames[key] = build_frames(ep, perspective)
        return self._frames[key]


def build_frames(ep: Episode, perspective: str) -> list[dict]:
    """One frame per tick holding only what `perspective` may see: filtered events, node states folded from those
    events alone, and that role's own reward. Nothing is filtered after the fact."""
    allowed = ALLOWED[perspective]
    roles = ["attacker", "defender"] if perspective == "evaluator" else [perspective]
    by_id = {n["id"]: n for n in ep.nodes}
    state: dict[str, dict] = {}
    reward = {r: 0.0 for r in roles}
    components: dict[str, dict[str, float]] = {r: {} for r in roles}
    frames, idx = [], 0
    for step in ep.steps:
        visible = [e for e in step["events"] if e.get("visible") in allowed]
        for e in visible:
            _fold(state, e, perspective)
        for r in roles:
            reward[r] += step["reward"][r]
            for k, v in step["components"][r].items():
                components[r][k] = round(components[r].get(k, 0.0) + v, 6)
        feed = []
        for e in visible:
            feed.append({"idx": idx, "tick": e.get("tick", step["tick"]), **_feed_entry(e)})
            idx += 1
        shown = [n for n in state if state[n]["status"] != "unknown"] if perspective == "attacker" else list(by_id)
        nodes = [{"id": n, "asset_type": by_id.get(n, {}).get("asset_type"), "zone": by_id.get(n, {}).get("zone"),
                  **_node_view(state.get(n))} for n in shown]
        frames.append({"tick": step["tick"], "nodes": nodes, "feed": feed,
                       "reward": {r: step["reward"][r] for r in roles},
                       "cumulative": {"reward": {r: round(reward[r], 6) for r in roles},
                                      "components": {r: dict(c) for r, c in components.items()}}})
    return frames


def _node_view(st: dict | None) -> dict:
    return {"status": st["status"], "tags": sorted(st["tags"])} if st else {"status": "reachable", "tags": []}


def _fold(state: dict, e: dict, perspective: str) -> None:
    node, kind, act = e.get("node"), e.get("type"), e.get("action")
    if node is None:
        return
    s = state.setdefault(node, {"status": "unknown" if perspective == "attacker" else "reachable", "tags": set()})
    if kind == "attacker_outcome" and e.get("success"):
        s["status"] = "reachable" if s["status"] == "unknown" else s["status"]
        s["tags"].add("attacker_access" if act in FOOTHOLD else "discovered")
    elif kind == "attacker_truth" and e.get("success") and act in FOOTHOLD:
        s["tags"].add("compromised")
    elif kind == "alert":
        s["tags"].add("alerting")
    elif kind == "defender_action_done":
        if act in ISOLATING:
            s["status"] = "isolated"
        elif act in UNISOLATING:
            s["status"] = "reachable"
        elif act in DEGRADING:
            s["status"] = "reachable"
        if act in ("revoke_session", "reimage"):
            s["tags"] -= {"compromised", "attacker_access"} if perspective == "evaluator" else set()
    elif kind == "defender_action_started" and act in DEGRADING:
        s["status"] = "degraded"
    elif kind == "control_lapsed":
        s["tags"].add("control_lapsed")


def _feed_entry(e: dict) -> dict:
    t = e["type"]
    actor = "attacker" if t.startswith("attacker") else "defender" if t.startswith("defender") or t in ("alert", "control_lapsed") else "system"
    outcome = ("success" if e["success"] else "failed") if "success" in e else \
        {"defender_action_started": "started", "defender_action_done": "done", "alert": f"alert: {e.get('kind')}",
         "control_lapsed": "control lapsed"}.get(t, t)
    return {"type": t, "actor": actor, "action": e.get("action"), "target": e.get("node"), "outcome": outcome,
            "duration": e.get("duration"), "scope": e.get("visible")}
