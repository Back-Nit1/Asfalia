import copy

import pytest

from hospital_rl.engine.permissions import attacker_has_access
from hospital_rl.engine.state import Session
from hospital_rl.engine.transitions import WAIT, Action, IllegalAction, World
from hospital_rl.policies.scripted import GreedyDefender, IsolateEverything, Noop, ScriptedAttacker
from hospital_rl.runner import run_episode
from hospital_rl.schemas import load_scenario

SPEC = load_scenario("configs/tiny.yaml")


def world(**over):
    spec = SPEC.model_copy(update=over) if over else SPEC
    return World(spec, seed=1)


def d(t, n=None):
    return Action(t, n)


def step_d(w, *acts):
    for a in acts:
        w.step(WAIT, a)


def test_scripted_episode_deterministic_and_complete():
    r1 = run_episode(SPEC, 7, ScriptedAttacker(), GreedyDefender())
    r2 = run_episode(SPEC, 7, ScriptedAttacker(), GreedyDefender())
    assert r1 == r2 and len(r1["steps"]) == SPEC.horizon
    assert r1["metrics"]["records_exfiltrated"] <= 20


def test_records_counted_once():
    r = run_episode(SPEC, 7, ScriptedAttacker(), Noop())
    assert r["metrics"]["records_exfiltrated"] == 10          # ehr only; never above its inventory
    assert r["metrics"]["records_exfiltrated"] <= r["metrics"]["records_accessed"]


def test_unpatchable_device_not_patch_legal():
    w = world()
    assert d("patch", "pump") not in w.legal_defender()
    assert d("patch", "pacs") in w.legal_defender()


def test_patch_respects_maintenance_window():
    w = world(maintenance_windows=[(5, 10)])
    assert d("patch", "pacs") not in w.legal_defender()


def test_patch_does_not_remove_existing_compromise():
    w = world()
    w.st.sessions.append(Session("ehr", "service", None))
    w.st.nodes["ehr"].implant = True
    step_d(w, d("patch", "ehr"))
    for _ in range(3):
        w.step(WAIT, WAIT)
    assert "lateral" in w.st.nodes["ehr"].patched
    assert attacker_has_access(w.st, "ehr")                    # persistence after patching


def test_revoke_session_keeps_implant():
    w = world()
    w.st.sessions.append(Session("ws1", "service", None))
    w.st.nodes["ws1"].implant = True
    step_d(w, d("revoke_session", "ws1"))
    assert all(s.revoked for s in w.st.sessions) and attacker_has_access(w.st, "ws1")
    step_d(w, d("reimage", "ws1"))
    for _ in range(6):
        w.step(WAIT, WAIT)
    assert not attacker_has_access(w.st, "ws1")


def test_rotate_credential_kills_credentialed_session():
    w = world()
    w.st.sessions.append(Session("ehr", "clinician", "c_clin"))
    assert attacker_has_access(w.st, "ehr")
    step_d(w, d("rotate_credential", "ws1"))
    w.step(WAIT, WAIT)
    assert not attacker_has_access(w.st, "ehr")


def test_isolation_disrupts_and_restore_recovers_workflow():
    w = world()
    base = w.step(WAIT, WAIT)["reward"]["defender"]
    w.step(WAIT, d("isolate", "ehr"))
    fails = [e for e in w.events if e["type"] == "workflow_failed"]
    assert fails
    w.step(WAIT, d("restore_connectivity", "ehr"))
    assert not [e for e in w.events if e["type"] == "workflow_failed" and e["workflow"] == "chart"]


def test_isolate_everything_violates_availability():
    r = run_episode(SPEC, 3, ScriptedAttacker(), IsolateEverything())
    assert r["metrics"]["availability_violation"]


def test_isolation_does_not_erase_compromise():
    w = world()
    w.st.nodes["ws1"].implant = True
    step_d(w, d("isolate", "ws1"))
    assert w.st.nodes["ws1"].implant and not attacker_has_access(w.st, "ws1")   # inert while blocked


def test_budget_hard_cap():
    r = run_episode(SPEC.model_copy(update={"budget": 5.0}), 3, Noop(), __import__("hospital_rl.policies.scripted", fromlist=["x"]).StaticHardening())
    assert r["metrics"]["spend"] <= 5.0 + 1e-9


def test_illegal_actions_raise():
    w = world()
    with pytest.raises(IllegalAction):
        w.step(Action("exfiltrate", "ehr"), WAIT)
    with pytest.raises(IllegalAction):
        w.step(WAIT, d("restore_connectivity", "ehr"))


def test_wait_always_legal():
    w = world()
    assert WAIT in w.legal_attacker() and WAIT in w.legal_defender()


def test_masks_do_not_reveal_hidden_state():
    w1, w2 = world(), world()
    w2.st.nodes["pacs"].implant = True                          # hidden compromise only
    w2.st.nodes["billing"].isolated = False
    assert w1.legal_defender() == w2.legal_defender()
    assert w1.view("defender") == w2.view("defender")
    assert w1.view("attacker") == w2.view("attacker")
    w2.topo.node["ehr"].vulns.append("priv_esc")               # hidden vulnerability
    assert w1.legal_attacker() == w2.legal_attacker()


def _race(policy):
    """Defender isolates ehr on the same tick the attacker exfiltrates from it."""
    w = World(SPEC.model_copy(update={"tie_policy": policy}), seed=1)
    w.st.nodes["ehr"].implant = True
    w.st.nodes["ehr"].staged = {"ehr:r0", "ehr:r1"}
    w.st.nodes["ehr"].taken = {"ehr:r0", "ehr:r1"}
    w.st.know.nodes.add("ehr")
    w.step(Action("exfiltrate", "ehr"), d("isolate", "ehr"))
    return w.st.m.records_exfiltrated


def test_tie_policy_simultaneous_containment_and_exfil():
    assert _race("defender_first") == 0
    assert _race("attacker_first") == 2


def test_patch_blocks_future_access_only():
    w = world()
    step_d(w, d("patch", "vpn"))
    for _ in range(3):
        w.step(WAIT, WAIT)
    assert not w._unpatched("vpn", "initial_access")


def test_emergency_override_bounded_and_audited():
    w = world(emergency_overrides=1)
    w.step(WAIT, d("enable_mfa", "ehr"))                       # mfa latency exceeds emergency deadline
    n = sum(e["type"] == "emergency_override" for e in w.events)
    for _ in range(40):
        w.step(WAIT, WAIT)
        n += sum(e["type"] == "emergency_override" for e in w.events)
        if w.st.tick >= 39:
            break
    assert n == 1 and w.st.overrides_left == 0
