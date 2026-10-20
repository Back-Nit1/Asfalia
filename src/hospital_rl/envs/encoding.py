"""Role-local observation encoding. Reads only World.view(role) and legal-action lists, never true state."""
from __future__ import annotations

import hashlib

import numpy as np

from ..engine.transitions import WAIT, Action, IllegalAction, World
from ..schemas import ATTACKER_TYPES, DEFENDER_TYPES, MAX_NODES, catalog_hash

ASSETS = ["ehr", "pacs", "workstation", "billing", "idp", "vpn", "k8s_service", "device"]
ALERT_KINDS = ["auth_anomaly", "flow_anomaly", "process_anomaly", "benign_noise"]
VULNS = ["initial_access", "priv_esc", "lateral"]
HIST, E_DIM, G_DIM = 8, 16, 6
F_DIM = {"attacker": len(ASSETS) + 2 + len(VULNS) + 2, "defender": len(ASSETS) + 9}
TYPES = {"attacker": ATTACKER_TYPES, "defender": DEFENDER_TYPES}
OBS_VERSION = 1


def n_actions(role: str) -> int:
    return 1 + len(TYPES[role]) * MAX_NODES


def obs_schema_hash(role: str) -> str:
    blob = f"{OBS_VERSION}|{role}|{MAX_NODES}|{F_DIM[role]}|{HIST}|{E_DIM}|{G_DIM}|{catalog_hash()}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Codec:
    """Per-episode slot permutation + action index <-> Action mapping for both roles."""

    def __init__(self, node_ids: list[str], seed: int):
        order = np.random.default_rng(seed).permutation(len(node_ids))
        self.slot_node = [node_ids[i] for i in order]
        self.node_slot = {n: s for s, n in enumerate(self.slot_node)}

    def encode(self, role: str, a: Action) -> int:
        if a.type == "wait":
            return 0
        t = [x.value for x in TYPES[role]].index(a.type)
        return 1 + t * MAX_NODES + self.node_slot[a.node]

    def decode(self, role: str, idx: int) -> Action:
        if not (0 <= idx < n_actions(role)):
            raise IllegalAction(f"{role} action index {idx} outside catalog [0,{n_actions(role)})")
        if idx == 0:
            return WAIT
        t, slot = divmod(idx - 1, MAX_NODES)
        if slot >= len(self.slot_node):
            raise IllegalAction(f"{role} action index {idx} targets padding slot {slot}")
        return Action(TYPES[role][t].value, self.slot_node[slot])

    def mask(self, role: str, legal: list[Action]) -> np.ndarray:
        m = np.zeros(n_actions(role), dtype=np.int8)
        for a in legal:
            m[self.encode(role, a)] = 1
        m[0] = 1
        return m


def encode_obs(w: World, codec: Codec, role: str, hist: list[np.ndarray]) -> dict:
    v = w.view(role)
    n = len(codec.slot_node)
    F = np.zeros((MAX_NODES, F_DIM[role]), np.float32)
    A = np.zeros((MAX_NODES, MAX_NODES), np.int8)
    mask = np.zeros(MAX_NODES, np.int8)
    mask[:n] = 1
    if role == "attacker":
        for nid in v["known_nodes"]:
            s = codec.node_slot[nid]
            F[s, ASSETS.index(v["node_types"][nid])] = 1
            F[s, len(ASSETS)] = 1                                   # known
            F[s, len(ASSETS) + 1] = nid in v["access"]
            for i, f in enumerate(VULNS):
                F[s, len(ASSETS) + 2 + i] = f in v["known_vulns"].get(nid, ())
            F[s, len(ASSETS) + 5] = nid in v["known_vulns"]          # probed
            F[s, len(ASSETS) + 6] = min(v["staged"].get(nid, 0) / 10, 1.0)
        for a, b in v["known_edges"]:
            A[codec.node_slot[a], codec.node_slot[b]] = 1
        g = [v["tick"] / w.spec.horizon, len(v["held_creds"]) / 4, len(v["access"]) / MAX_NODES, 0, 0, 0]
    else:
        recent = {}
        for tick, nid, kind in v["alerts"]:
            if tick >= v["tick"] - 5:
                recent[nid] = recent.get(nid, 0) + 1
        busy = {j[1] for j in v["jobs"]}
        for nid, nv in v["nodes"].items():
            s = codec.node_slot[nid]
            node = w.topo.node[nid]                                  # static inventory: authorized for defender
            F[s, ASSETS.index(nv["asset_type"])] = 1
            o = len(ASSETS)
            F[s, o:o + 9] = [nv["logging"] / 2, nv["mfa"], nv["isolated"], nv["quarantined"], nv["service_up"],
                             nv["patchable"], nid in busy, min(recent.get(nid, 0) / 3, 1.0),
                             node.records * node.sensitivity / 10]
        for src, outs in w.topo.out.items():
            for dst in outs["reach"]:
                A[codec.node_slot[src], codec.node_slot[dst]] = 1
        g = [v["tick"] / w.spec.horizon, v["budget_left"] / w.spec.budget, float(v["in_window"]),
             len(v["jobs"]) / max(w.spec.action_capacity, 1), v["overrides_left"] / max(w.spec.emergency_overrides, 1), 0]
    H = np.zeros((HIST, E_DIM), np.float32)
    for i, h in enumerate(hist[-HIST:]):
        H[HIST - len(hist[-HIST:]) + i] = h
    legal = w.legal_attacker() if role == "attacker" else w.legal_defender()
    return {"node_features": F, "node_mask": mask, "adjacency": A, "history": H,
            "globals": np.array(g, np.float32), "action_mask": codec.mask(role, legal)}


def step_record(w: World, codec: Codec, role: str, events: list[dict], own: Action) -> np.ndarray:
    """One history row from role-visible events + the role's own last action."""
    h = np.zeros(E_DIM, np.float32)
    kinds = [x.value for x in TYPES[role]]
    if own.type != "wait":
        h[kinds.index(own.type)] = 1
    if role == "attacker":
        for e in events:
            if e["type"] == "attacker_outcome" and e["action"] == own.type and e["node"] == own.node:
                h[12] = e["success"]
                h[13] = codec.node_slot[e["node"]] / MAX_NODES
    else:
        for e in events:
            if e["type"] == "alert":
                h[12 + ALERT_KINDS.index(e["kind"])] += 1
    return h

ROLES_OTHER = {"attacker": "defender", "defender": "attacker"}
