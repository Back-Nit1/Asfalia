import json

from hospital_rl.api.auth import AuthConfig

from .conftest import EV, OP, ev, sse

EPS = "fixture_tiny_episode"
ATTACKER_FORBIDDEN = {"final_metrics", "edges", "notice", "defender_action_started", "defender_action_done", "alert",
                      "attacker_truth", "control_lapsed"}
DEFENDER_FORBIDDEN = {"final_metrics", "attacker_outcome", "attacker_truth", "compromised", "attacker_access", "discovered"}


def words(obj) -> set[str]:
    """every key and string value in a JSON body"""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                out.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            out.add(o)
    walk(obj)
    return out


def test_attacker_dto_contains_no_evaluator_or_defender_fields(client):
    for path in ("snapshot", "replay", "events"):
        body = client.get(f"/api/world/attacker/episodes/{EPS}/{path}").json()
        assert body["privileged"] is False and not (words(body) & ATTACKER_FORBIDDEN), path
    snap = client.get(f"/api/world/attacker/episodes/{EPS}/snapshot?tick=0").json()
    assert [n["id"] for n in snap["nodes"]] == ["vpn"]     # undiscovered hosts are not even named


def test_defender_dto_contains_no_attacker_or_evaluator_fields(client):
    for path in ("snapshot", "replay", "events"):
        body = client.get(f"/api/world/defender/episodes/{EPS}/{path}").json()
        assert body["privileged"] is False and not (words(body) & DEFENDER_FORBIDDEN), path


def test_filtering_happens_server_side_per_endpoint(client):
    a = client.get(f"/api/world/attacker/episodes/{EPS}/events").json()["events"]
    d = client.get(f"/api/world/defender/episodes/{EPS}/events").json()["events"]
    assert a and d and {e["scope"] for e in a} == {"attacker"} and {e["scope"] for e in d} == {"defender"}


def test_evaluator_requires_its_own_token(client):
    url = f"/api/world/evaluator/episodes/{EPS}/snapshot"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"Authorization": f"Bearer {OP}"}).status_code == 403
    ok = client.get(url, headers={"Authorization": f"Bearer {EV}"})
    body = ok.json()
    assert ok.status_code == 200 and body["privileged"] is True and "PRIVILEGED" in body["notice"] and "final_metrics" in body
    assert "attacker_truth" in words(body)


def test_evaluator_unconfigured_is_503(make_client):
    c = make_client(AuthConfig(OP, None))
    assert c.get(f"/api/world/evaluator/episodes/{EPS}/snapshot", headers={"Authorization": f"Bearer {OP}"}).status_code == 503


def test_non_local_requires_operator_token(make_client):
    c = make_client(AuthConfig(OP, EV, require_operator=True))
    assert c.get("/api/tasks").status_code == 401
    assert c.get(f"/api/world/attacker/episodes/{EPS}/snapshot").status_code == 401
    assert c.get("/api/tasks", headers={"Authorization": f"Bearer {OP}"}).status_code == 200
    assert c.get("/api/health").status_code == 200


def test_for_bind_refuses_open_non_local():
    import pytest
    with pytest.raises(SystemExit):
        AuthConfig.for_bind("0.0.0.0", None, None)
    assert AuthConfig.for_bind("127.0.0.1", None, None).require_operator is False


def test_development_stream_unreachable_from_policy_endpoints(client, store):
    store.append_event(ev(0, event_type="task.state_changed", task_id="t1"))
    store.append_event(ev(1, source="world", worker="env", visibility="attacker", event_type="w.atk", episode_id="e"))
    store.append_event(ev(2, source="world", worker="env", visibility="defender", event_type="w.def", episode_id="e"))
    store.append_event(ev(3, source="world", worker="env", visibility="evaluator", event_type="w.eval", episode_id="e"))
    # a dev event that (wrongly) claims attacker visibility still must not leak: it is not a world-source event
    store.append_event(ev(0, worker="rogue", visibility="attacker", event_type="dev.leak"))

    def types(path, **h):
        return [d["event_type"] for *_, d in sse(client.get(path, headers=h))]

    assert types("/api/world/attacker/stream?follow=false") == ["w.atk"]
    assert types("/api/world/defender/stream?follow=false") == ["w.def"]
    auth = {"Authorization": f"Bearer {EV}"}
    assert types("/api/world/evaluator/stream?follow=false", **auth) == ["w.atk", "w.def", "w.eval"]
    assert types("/api/events/stream?follow=false") == ["task.state_changed"]       # operator stream: dev plane only
    assert client.get("/api/world/evaluator/stream?follow=false").status_code == 401
    assert client.get("/api/world/attacker/tasks").status_code == 404           # no dev routes under policy prefixes


def test_resync_snapshot_on_world_stream_has_no_tasks(client, store):
    for i in range(6):
        store.append_event(ev(i, source="world", worker="env", visibility="attacker"))
    store.prune_events(2)
    got = sse(client.get("/api/world/attacker/stream?follow=false", headers={"Last-Event-ID": "1"}))
    assert got[0][1] == "resync" and "tasks" not in json.dumps(got[0][2])
