"""Local synthetic-testbed tests: isolation, determinism/reset, bounded data, audit separation, typed actions only."""
import re

import pytest

from hospital_rl.evaluation.splits import scenario_for
from hospital_rl.testbed.adapter import compare, graph_episode, replay
from hospital_rl.testbed.services import (MAX_RECORDS, ActionRejected, ActorAPI, AuditLog, IsolationPolicy,
                                          SyntheticServices)

SPEC = scenario_for("train", 3, 8, 0.5, 20)


def dump(sv):
    return sv.db.execute("SELECT * FROM patient UNION ALL SELECT id, node, modality FROM imaging_study ORDER BY 1").fetchall()


def test_seed_is_deterministic_and_reset_restores_exactly():
    a, b = SyntheticServices(SPEC, 7), SyntheticServices(SPEC, 7)
    assert dump(a) == dump(b) and dump(a) != dump(SyntheticServices(SPEC, 8))
    a.apply("defender", "isolate", "ehr0" if "ehr0" in a.svc else next(iter(a.svc)))
    a.step()
    a.reset()
    assert dump(a) == dump(b) and not any(s.isolated for s in a.svc.values()) and a.tick == 0


def test_data_is_bounded_and_invented():
    sv = SyntheticServices(SPEC, 1)
    assert all(sv.records_at(n.id) <= MAX_RECORDS for n in SPEC.nodes)
    ids = [r[0] for r in dump(sv)]
    assert ids and all(re.fullmatch(r"SYN-[A-Za-z0-9_-]+", i) for i in ids)
    cols = [c[1] for t in ("patient", "imaging_study", "identity") for c in sv.db.execute(f"PRAGMA table_info({t})")]
    assert not {"name", "dob", "ssn", "address", "phone", "mrn"} & set(cols)


def test_only_allowlisted_typed_actions_and_known_nodes():
    sv = SyntheticServices(SPEC, 1)
    node = next(iter(sv.svc))
    for bad in ("DROP TABLE patient", "sh -c id", "select * from patient", "exfiltrate"):
        with pytest.raises(ActionRejected):
            sv.apply("defender", bad, node)  # exfiltrate is attacker-only
    with pytest.raises(ActionRejected):
        sv.apply("defender", "isolate", "x'; DROP TABLE patient;--")
    with pytest.raises(ActionRejected):
        sv.apply("root", "isolate", node)
    assert sv.row_counts()["patient"] > 0 or sv.row_counts()["imaging_study"] > 0


def test_actor_api_exposes_actions_and_a_view_only():
    api = ActorAPI(SyntheticServices(SPEC, 1), "defender")
    assert {n for n in dir(api) if not n.startswith("_")} == {"act", "observe"}
    obs = api.observe()
    assert set(obs) == {"tick", "services"} and "audit" not in str(obs)


def test_audit_is_separate_from_actor_accessible_records():
    audit = AuditLog()
    sv = SyntheticServices(SPEC, 1, audit)
    node = next(iter(sv.svc))
    ActorAPI(sv, "defender").act("increase_logging", node)
    sv.step()
    assert any(r[2] == "increase_logging" for r in audit.rows()) and any(r[1] == "benign" for r in audit.rows())
    tables = {r[0] for r in sv.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "audit" not in tables and audit._db is not sv.db


def test_isolation_policy_denies_egress_and_provider_credentials():
    IsolationPolicy().check()
    with pytest.raises(ValueError):
        IsolationPolicy(internal_only=False).check()
    with pytest.raises(PermissionError):
        IsolationPolicy.egress("api.daytona.io")
    IsolationPolicy.egress("testbed-db")
    env = IsolationPolicy.actor_env({"PATH": "/bin", "DAYTONA_API_KEY": "x", "MODAL_TOKEN_ID": "y", "HOME": "/h", "MY_SECRET": "z"})
    assert env == {"PATH": "/bin", "HOME": "/h"}


def test_workload_is_deterministic_and_bounded():
    a, b = SyntheticServices(SPEC, 5), SyntheticServices(SPEC, 5)
    assert [a.workload() for _ in range(5)] == [b.workload() for _ in range(5)]
    assert all(0 <= a.workload() <= 3 for _ in range(20))


def test_adapter_replay_reports_mismatch_honestly():
    r = replay(SPEC, 3)
    assert r["ticks"] > 0 and "validity" in r["caveat"] and all(m["cause"] in ("pending_job", "mismatch") for m in r["mismatches"])
    # inject a divergence: the comparison must flag it as an unexplained mismatch
    w, acts = graph_episode(SPEC, 3)
    sv = SyntheticServices(SPEC, 3)
    for a, d in acts:
        sv.apply("defender", d.type, d.node)
    node = next(iter(sv.svc))
    sv.svc[node].isolated = not w.st.nodes[node].isolated
    assert any(m["node"] == node and m["flag"] == "isolated" for m in compare(w, sv))
