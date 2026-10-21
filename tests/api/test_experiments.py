"""Control-plane HTTP behaviour against the deterministic fake backend (no training, no cloud)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hospital_rl.api.app import FIXTURES, create_app
from hospital_rl.api.auth import AuthConfig
from hospital_rl.api.replay import Library
from hospital_rl.api.storage import Store
from hospital_rl.experiments.controller import Controller
from hospital_rl.experiments.fakes import FakeBackend, FakeEvaluator
from hospital_rl.experiments.reader import FsReader
from hospital_rl.orchestration.contracts import PlanConfig
from hospital_rl.orchestration.scheduler import Scheduler

from .conftest import EV, OP, sse

RUN = {"preset": "attacker-school-smoke", "total_steps": 512}


class Env:
    def __init__(self, tmp, auth=None, db=None):
        self.store = Store(db or tmp / "x.sqlite")
        self.ctl = Controller(self.store, FakeBackend(tmp / "jobs"), FsReader(tmp / "jobs"), evaluator=FakeEvaluator(FIXTURES / "fixture_tiny_episode.json"),
                              episodes_dir=str(tmp / "eps"))
        self.lib = Library(FIXTURES, tmp / "eps")
        self.c = TestClient(create_app(self.store, Scheduler(self.store, PlanConfig(), []), auth or AuthConfig(OP, EV), self.lib, controller=self.ctl))
        self.h = {"Authorization": f"Bearer {OP}"}
        self.n = 0

    def post(self, path, json=None, key=None, **kw):
        self.n += 1
        return self.c.post(path, json=json, headers={**self.h, "Idempotency-Key": key or f"k{self.n}"}, **kw)

    def get(self, path, **kw):
        return self.c.get(path, headers=self.h, **kw)

    def tick(self, n=1):
        for _ in range(n):
            self.ctl.tick()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_validate_has_no_side_effects_and_reports_every_error(env):
    ok = env.post("/api/runs/validate", RUN).json()
    assert ok["caps"]["total_steps"] == 512 and ok["estimate"]["basis"] == "unavailable"
    bad = env.post("/api/runs/validate", {"preset": "attacker-school-smoke", "cpu": 99, "total_steps": 10**9})
    assert bad.status_code == 422 and len(bad.json()["errors"]) == 2 and "cpu" in bad.json()["detail"] and "total_steps" in bad.json()["detail"]
    assert env.get("/api/runs").json()["runs"] == []


def test_create_returns_immediately_with_durable_id_and_is_idempotent(env):
    a = env.post("/api/runs", RUN, key="same")
    b = env.post("/api/runs", RUN, key="same")
    assert a.status_code == 200 and a.json()["state"] == "queued" and a.json()["id"] == "run-0001"
    assert b.json() == a.json() and b.headers["idempotent-replay"] == "true"
    assert len(env.get("/api/runs").json()["runs"]) == 1 and env.ctl.backend.submits == []     # request never touched the backend
    assert env.post("/api/runs", RUN).status_code == 409                                       # second distinct request hits the cap
    assert env.c.post("/api/runs", json=RUN, headers=env.h).status_code == 400                  # no key


def test_auth_evaluator_token_is_not_an_operator_token(tmp_path):
    e = Env(tmp_path, AuthConfig(OP, EV, require_operator=True))
    assert e.c.get("/api/runs").status_code == 401
    assert e.c.get("/api/runs", headers={"Authorization": f"Bearer {EV}"}).status_code == 401
    assert e.c.post("/api/runs", json=RUN, headers={"Authorization": f"Bearer {EV}", "Idempotency-Key": "z"}).status_code == 401
    assert e.get("/api/runs").status_code == 200
    assert e.ctl.list() == []


def test_lifecycle_cancellation_modes_and_legal_transitions(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    assert env.post(f"/api/runs/{rid}/drain").status_code == 409            # queued: nothing to drain
    env.tick(3)
    assert env.post(f"/api/runs/{rid}/drain").json()["state"] == "draining"
    env.tick()
    assert env.get(f"/api/runs/{rid}").json()["state"] == "stopped"
    assert env.post(f"/api/runs/{rid}/cancel").status_code == 409              # stopped is not cancellable
    assert env.post(f"/api/runs/{rid}/checkpoint-stop").status_code == 409
    rid2 = env.post(f"/api/runs/{rid}/resume").json()["id"]
    env.tick(2)
    c = env.post(f"/api/runs/{rid2}/cancel").json()
    assert c["state"] == "cancelled" and "lost" in c["error"]
    assert env.post("/api/runs/nope/cancel").status_code == 404


def test_checkpoint_stop_resume_lineage_and_metrics_over_http(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    env.tick(4)
    assert env.post(f"/api/runs/{rid}/checkpoint-stop").json()["state"] == "checkpointing"
    env.tick()
    cps = env.get(f"/api/runs/{rid}/checkpoints").json()["checkpoints"]
    assert env.post(f"/api/runs/{rid}/resume?checkpoint_id=bogus").status_code == 422
    r = env.post(f"/api/runs/{rid}/resume", key="res").json()
    assert r["state"] == "queued" and r["attempts"][1]["resume_checkpoint"] == cps[-1]["id"]
    assert env.post(f"/api/runs/{rid}/resume", key="res").json() == r          # replay, no third attempt
    env.tick(14)
    run = env.get(f"/api/runs/{rid}").json()
    assert run["state"] == "completed" and [a["resume_from"] for a in run["attempts"]] == [None, "a1"]
    m = env.get(f"/api/runs/{rid}/metrics").json()["metrics"]
    assert m[-1]["env_steps"] == 512 and len(m) == 8
    assert env.get(f"/api/runs/{rid}/metrics?after=5").json()["metrics"][0]["seq"] in (0, 1, 2, 3, 4, 5, 6, 7)


def test_restart_recovery_keeps_state_and_does_not_manufacture_completion(tmp_path):
    e1 = Env(tmp_path)
    rid = e1.post("/api/runs", RUN).json()["id"]
    e1.tick(3)
    e2 = Env(tmp_path)                                  # new process, same DB + artifacts, backend memory gone
    assert e2.get(f"/api/runs/{rid}").json()["state"] == "running"       # persisted, not reset
    e2.tick()
    r = e2.get(f"/api/runs/{rid}").json()
    assert r["state"] == "failed" and "outcome unknown" in r["error"] and r["resumable"]


def test_stale_heartbeat_visible_over_http(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    env.tick(2)
    env.ctl.clock = lambda c=env.ctl.clock: c() + 120
    r = env.get(f"/api/runs/{rid}").json()
    assert r["stale"] is True and r["state"] == "running" and r["heartbeat_age_s"] > 100
    assert env.get("/api/overview").json()["runs"]["stale"] == [rid]


def test_artifact_metadata_and_cleanup_endpoints(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    env.tick(12)
    art = env.get(f"/api/runs/{rid}/artifacts").json()["attempts"][0]
    names = {f["name"] for f in art["files"]}
    assert {"manifest.json", "result.json", "latest.pt"} <= names and art["partial"] is False
    ck = env.get("/api/checkpoints").json()["checkpoints"][0]
    assert len(ck["sha256"]) == 64 and ck["role"] == "attacker" and ck["available"] == 1
    dry = env.post("/api/infrastructure/cleanup").json()
    assert dry["dry_run"] and dry["plan"][0]["action"] == "delete"
    assert (env.ctl.reader.root / rid / "a1").exists()                        # dry run deleted nothing
    env.post("/api/infrastructure/cleanup?dry_run=false")
    assert not (env.ctl.reader.root / rid / "a1").exists()
    inf = env.get("/api/infrastructure").json()
    assert inf["configured"] and inf["backend"] == "fake" and inf["jobs"][0]["cleaned"]


def _ckpts(env):
    env.post("/api/runs", {"preset": "attacker-school-smoke", "total_steps": 128})
    env.tick(6)
    env.post("/api/runs", {"preset": "defender-school-smoke", "total_steps": 128})
    env.tick(6)
    cps = env.get("/api/checkpoints").json()["checkpoints"]
    return [c["id"] for c in cps if c["role"] == "attacker"][-1], [c["id"] for c in cps if c["role"] == "defender"][-1]


def test_evaluation_results_replay_and_debug_exclusion(env):
    atk, dfn = _ckpts(env)
    bad = env.post("/api/evaluations", {"attacker_checkpoint_id": dfn, "defender_checkpoint_id": atk})
    assert bad.status_code == 422 and len(bad.json()["errors"]) == 2
    ev = env.post("/api/evaluations", {"attacker_checkpoint_id": atk, "defender_checkpoint_id": dfn, "episodes": 4}).json()
    assert ev["state"] == "queued" and env.get("/api/experiments/results").json()["results"] == []      # nothing shown before it is real
    env.tick()
    res = env.get("/api/experiments/results").json()["results"]
    assert [r["id"] for r in res] == [ev["id"]] and res[0]["result"]["pairs"][0]["metric"] == "defender_win_rate"
    fork = env.post("/api/runs/run-0001/debug-fork", params={"checkpoint_id": atk, "tick": 2}).json()
    env.tick()
    assert fork["label"] == "DEBUG"
    assert [r["id"] for r in env.get("/api/experiments/results").json()["results"]] == [ev["id"]]      # DEBUG excluded from benchmarks
    assert [f["id"] for f in env.get("/api/debug-forks").json()["forks"]] == [fork["id"]]
    eps = {e["id"]: e["label"] for e in env.get("/api/world/episodes").json()["episodes"]}
    eid = res[0]["result"]["episode_ids"][0]
    assert eps[eid] == "FIXTURE" and eps[fork_ep(env, fork)] == "DEBUG"                                # exports reach the replay library
    assert env.get(f"/api/world/defender/episodes/{eid}/replay").status_code == 200


def fork_ep(env, fork):
    return env.get(f"/api/evaluations/{fork['id']}").json()["result"]["episode_ids"][0]


def test_evaluator_truth_stays_out_of_policy_views_of_exported_episodes(env):
    atk, dfn = _ckpts(env)
    env.post("/api/evaluations", {"attacker_checkpoint_id": atk, "defender_checkpoint_id": dfn})
    env.tick()
    eid = env.get("/api/experiments/results").json()["results"][0]["result"]["episode_ids"][0]
    for p in ("attacker", "defender"):
        body = env.get(f"/api/world/{p}/episodes/{eid}/replay").json()
        assert "final_metrics" not in body and body["privileged"] is False
    assert env.get(f"/api/world/evaluator/episodes/{eid}/replay").status_code == 403          # operator token is refused


def test_experiment_events_never_reach_dev_or_policy_streams(env):
    env.post("/api/runs", RUN)
    env.tick(3)
    assert env.store.events_after(0, source_kinds=("experiment",))
    dev = sse(env.get("/api/events/stream?follow=false"))
    assert all(d["source_kind"] == "development" for _, ev, d in dev if ev == "envelope")
    for p in ("attacker", "defender"):
        world = sse(env.get(f"/api/world/{p}/stream?follow=false"))
        assert all(d["source_kind"] == "world" and d["visibility"] == p for _, ev, d in world if ev == "envelope")
    ex = [d for _, ev, d in sse(env.get("/api/experiments/stream?follow=false")) if ev == "envelope"]
    assert ex and {d["source_kind"] for d in ex} == {"experiment"} and "training_metric" in {d["event_type"] for d in ex}


def test_experiment_sse_resync_after_pruning_and_last_event_id(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    env.tick(6)
    first = [i for i, ev, _ in sse(env.get(f"/api/experiments/stream?follow=false&run_id={rid}")) if ev == "envelope"]
    resumed = sse(env.c.get(f"/api/experiments/stream?follow=false&run_id={rid}", headers={**env.h, "Last-Event-ID": str(first[1])}))
    assert [i for i, ev, _ in resumed if ev == "envelope"] == first[2:]
    env.store.prune_events(2)
    ev = sse(env.get("/api/experiments/stream?follow=false&after=1"))
    assert ev[0][1] == "resync" and ev[0][2]["runs"][0]["id"] == rid


def test_audit_log_covers_ok_and_refused_mutations(env):
    rid = env.post("/api/runs", RUN).json()["id"]
    env.post(f"/api/runs/{rid}/drain")
    log = env.get("/api/audit").json()["audit"]
    assert [(a["action"], a["outcome"]) for a in reversed(log)] == [("create_run", "ok"), ("drain", "refused")]
    assert log[0]["actor"] == "operator"


def test_no_controller_still_reports_not_configured(client):
    assert client.get("/api/runs").json()["configured"] is False
    assert client.post("/api/runs", headers={"Idempotency-Key": "x"}).status_code == 501


def test_overview_counts_runs(env):
    env.post("/api/runs", RUN)
    o = env.get("/api/overview").json()
    assert o["runs"]["count"] == 1 and o["runs"]["by_state"] == {"queued": 1} and o["step_rate"] is None
