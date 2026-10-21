"""Procedural scenario generator. Topology family, workload profile, and difficulty come from the split."""
from __future__ import annotations

import numpy as np

from .schemas import CredentialSpec, EdgeSpec, IdentitySpec, NodeSpec, ScenarioSpec, WorkflowSpec

FAMILIES = ("hub", "tiered", "flat", "segmented")
WORKLOADS = {"day": 1.0, "night": 0.4, "surge": 1.8}     # multiplier on request rates


def generate(family: str, size: int, seed: int, difficulty: float = 0.5, workload: str = "day",
             horizon: int = 40, base_logging: int = 0) -> ScenarioSpec:
    """size = approximate node count (>= 8). difficulty in [0,1] controls vuln density and cred reuse."""
    rng = np.random.default_rng(seed)
    n_ehr = 1 + (size >= 24)
    extra_pacs = size >= 32
    rest = size - (4 + n_ehr + 1 + extra_pacs)           # nodes left for workstations + devices
    n_dev = max(1, rest // 4)
    n_ws = max(1, rest - n_dev)
    nodes: list[NodeSpec] = []
    N = lambda **kw: nodes.append(NodeSpec(**kw))
    p_vuln = 0.4 + 0.5 * difficulty

    N(id="vpn", asset_type="vpn", zone="dmz", external=True, vulns=["initial_access"], logging=base_logging)
    N(id="idp", asset_type="idp", zone="core", credentials=["c_admin"], logging=base_logging)
    N(id="api", asset_type="k8s_service", zone="core", logging=base_logging)
    N(id="billing", asset_type="billing", zone="business", records=4, sensitivity=0.3,
      allowed_roles=["billing_ops", "support"], logging=base_logging)
    ehrs = [f"ehr{i}" for i in range(n_ehr)]
    for e in ehrs:
        N(id=e, asset_type="ehr", zone="core", records=int(rng.integers(8, 13)), sensitivity=1.0, encrypted=True,
          vulns=["lateral"] if rng.random() < p_vuln else [], allowed_roles=["clinician", "support"],
          restrictable_roles=["support"], logging=base_logging)
    pacs = ["pacs"] + (["pacs1"] if extra_pacs else [])
    for p in pacs:
        N(id=p, asset_type="pacs", zone="clinical", records=int(rng.integers(4, 9)), sensitivity=0.6,
          vulns=["lateral"] if rng.random() < p_vuln else [], allowed_roles=["clinician", "device_gw"],
          logging=base_logging)
    wss = [f"ws{i}" for i in range(n_ws)]
    for i, w in enumerate(wss):
        N(id=w, asset_type="workstation", zone="clinical",
          vulns=(["priv_esc"] if (i == 0 or rng.random() < p_vuln) else []) + (["lateral"] if rng.random() < difficulty else []),
          credentials=["c_clin"] if i == 0 else [], allowed_roles=["clinician", "device_gw"], logging=base_logging)
    devs = [f"dev{i}" for i in range(n_dev)]
    for dv in devs:
        N(id=dv, asset_type="device", zone="device", patchable=False, vulns=["lateral"], logging=base_logging)

    E: list[EdgeSpec] = []
    both = lambda a, b, k="reach": E.extend([EdgeSpec(src=a, dst=b, kind=k), EdgeSpec(src=b, dst=a, kind=k)])
    if family == "hub":
        for w in wss:
            both("vpn", w)
        hub = wss[0]
        for x in ehrs + pacs + ["idp"]:
            both(hub, x)
        for w in wss[1:]:
            both(hub, w)
    elif family == "tiered":
        both("vpn", wss[0])
        for a, b in zip(wss, wss[1:]):
            both(a, b)
        for i, w in enumerate(wss):
            both(w, ehrs[i % n_ehr])
            both(w, pacs[i % len(pacs)])
        both(wss[0], "idp")
    elif family == "flat":
        flat = wss + ehrs + pacs + ["idp"]
        both("vpn", wss[0])
        for i, a in enumerate(flat):
            for b in flat[i + 1:]:
                if rng.random() < 0.5 or b == flat[i + 1]:
                    both(a, b)
        for x in (ehrs[0], pacs[0]):
            both(wss[0], x)                              # clinical workflows must stay satisfiable
    else:  # segmented: ehr only reachable via api tier
        both("vpn", wss[0])
        for w in wss[1:]:
            both(wss[0], w)
        both(wss[0], "idp")
        for p in pacs:
            both(wss[0], p)
        both(wss[0], "api")
        for e in ehrs:
            both("api", e)
        for w in wss:
            both(w, "api")
    if family != "segmented":
        for e in ehrs:
            both(e, "api")
    both("billing", "api")
    for i, dv in enumerate(devs):
        both(dv, pacs[i % len(pacs)])
    E.append(EdgeSpec(src="vpn", dst=wss[0], kind="trust"))
    for e in ehrs:
        E.append(EdgeSpec(src=e, dst="idp", kind="depends"))
    E.append(EdgeSpec(src="billing", dst="api", kind="depends"))

    reach_scope = wss + ehrs + pacs if difficulty > 0.3 else wss[:1] + ehrs[:1]
    creds = [CredentialSpec(id="c_clin", identity="dr_a", scope=reach_scope),
             CredentialSpec(id="c_admin", identity="sup_a", scope=ehrs + ["billing"])]
    ids = [IdentitySpec(id="dr_a", role="clinician"), IdentitySpec(id="sup_a", role="support")]

    k = WORKLOADS[workload]
    via = ["api"] if family == "segmented" else []
    wf = [WorkflowSpec(id="chart", kind="chart_lookup", rate=2 * k, path=[wss[0], *via, ehrs[0]], role="clinician", deadline=4),
          WorkflowSpec(id="imaging", kind="image_retrieval", rate=1 * k, path=[wss[0], pacs[0]], role="clinician", deadline=3),
          WorkflowSpec(id="bill", kind="billing_sync", rate=1 * k, path=["billing", "api"], role="billing_ops", deadline=3, critical=False),
          WorkflowSpec(id="er", kind="emergency_access", rate=0.5 * k, path=[wss[0], *via, ehrs[0]], role="clinician", deadline=3, emergency=True)]
    for i, dv in enumerate(devs):
        wf.append(WorkflowSpec(id=f"tel_{dv}", kind="device_telemetry", rate=0.5 * k, path=[dv, pacs[i % len(pacs)]],
                               role="device_gw", deadline=3, critical=False))
    return ScenarioSpec(name=f"{family}-{size}-s{seed}-{workload}", nodes=nodes, edges=E, identities=ids,
                        credentials=creds, workflows=wf, horizon=horizon, budget=30.0 + size * 0.5)
