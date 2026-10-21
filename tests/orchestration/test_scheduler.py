import subprocess

from hospital_rl.orchestration.tasks import Status

from .conftest import FakeAdapter, exists_cmd, mk


def drive(s, n=3):
    for _ in range(n):
        s.tick()


def test_acceptance_is_run_by_orchestrator_not_trusted(make):
    ok = FakeAdapter({"t": {"src/a/out.txt": "1"}})
    s = make([mk("t", cmds=[exists_cmd("src/a/out.txt")])], ok)
    drive(s)
    t = s.get("t")
    assert t.status is Status.passed
    assert t.evidence.test_results[0].exit_code == 0 and t.evidence.changed_files == ["src/a/out.txt"]
    # agent claims success but the acceptance command fails -> never passed
    liar = FakeAdapter({"t": {}})
    s2 = make([mk("t", cmds=[exists_cmd("src/a/missing.txt")])], liar)
    drive(s2)
    assert s2.get("t").status is not Status.passed and s2.get("t").evidence.test_results[0].exit_code == 1


def test_no_acceptance_commands_never_passes(make):
    s = make([mk("t")], FakeAdapter())
    drive(s)
    assert s.get("t").status is not Status.passed


def test_out_of_scope_changes_fail(make):
    s = make([mk("t", cmds=[exists_cmd("README")])], FakeAdapter({"t": {"elsewhere/x.txt": "1"}}))
    drive(s)
    t = s.get("t")
    assert t.status is not Status.passed and t.evidence.out_of_scope_files == ["elsewhere/x.txt"]


def test_lease_expiry_then_retry_cap_blocks(make, clock):
    ad = FakeAdapter(auto_finish=False)
    s = make([mk("t", cmds=["true"])], ad, lease_seconds=10)
    s.tick()
    for attempt in (1, 2, 3):
        t = s.get("t")
        assert t.status is Status.running and t.attempt == attempt
        clock.t += 11
        s.tick()   # lease expires -> failed -> ready -> restarted in the same tick, unless the cap is hit
    t = s.get("t")
    assert t.status is Status.blocked and "retry cap" in t.outcome and len(ad.cancelled) == 3
    assert len(ad.submitted) == 3   # original + 2 retries, no more
    s.tick()
    assert len(ad.submitted) == 3


def test_heartbeat_extends_lease(make, clock):
    s = make([mk("h", owner="human", cmds=["true"])], None, lease_seconds=10)
    s.tick()
    s.claim("h")
    clock.t += 8
    s.heartbeat("h")
    clock.t += 8
    s.tick()
    assert s.get("h").status is Status.running
    clock.t += 5
    s.tick()
    assert s.get("h").status is Status.ready and "lease expired" in s.get("h").outcome


def test_timeout(make, clock):
    ad = FakeAdapter(auto_finish=False)
    s = make([mk("t")], ad, lease_seconds=10_000, task_timeout_seconds=50)
    s.tick()
    clock.t += 60
    s.tick()
    outcomes = [e.payload["outcome"] for _, e in s.store.events_after(0, task_id="t")]
    assert any("timed out" in o for o in outcomes) and s.get("t").attempt == 2


def test_path_ownership_conflict_serializes(make):
    ad = FakeAdapter(auto_finish=False)
    s = make([mk("a", paths=["pyproject.toml"]), mk("b", paths=["pyproject.toml"]), mk("c", paths=["other"])], ad)
    s.tick()
    assert {t.id for t in s.all() if t.status is Status.running} == {"a", "c"}
    assert s.get("b").status is Status.ready and s.path_conflicts(s.get("b")) == ["a"]


def test_max_concurrency(make):
    ad = FakeAdapter(auto_finish=False)
    s = make([mk(str(i), paths=[f"p{i}"]) for i in range(5)], ad, max_concurrency=3)
    s.tick()
    assert sum(t.status is Status.running for t in s.all()) == 3


def test_dependencies_gate_on_integration(make):
    s = make([mk("a", paths=["pa"]), mk("b", deps=["a"], paths=["pb"])], FakeAdapter())
    s.tick()
    assert s.get("b").status is Status.queued


def test_cancel_running_stops_session_and_blocks_dependents(make):
    ad = FakeAdapter(auto_finish=False)
    s = make([mk("a", paths=["pa"]), mk("b", deps=["a"], paths=["pb"])], ad)
    s.tick()
    s.cancel("a")
    assert s.get("a").status is Status.cancelled and len(ad.cancelled) == 1
    assert s.get("b").status is Status.blocked


def test_no_runtime_reports_not_configured_and_does_not_fake_progress(make):
    s = make([mk("a")], None)
    drive(s)
    assert s.get("a").status is Status.ready
    assert s.runtime_status() == {"configured": False, "name": None, "detail": "coding runtime not configured"}


def test_integrate_runs_combined_suite_and_reverts_on_failure(make, repo):
    wt_cmd = exists_cmd("src/a/out.txt")
    s = make([mk("a", cmds=[wt_cmd])], FakeAdapter({"a": {"src/a/out.txt": "1"}}),
             integration_repo=str(repo), integration_branch="main", combined_suite=["false"])
    drive(s)
    t = s.get("a")
    subprocess.run(["git", "add", "-A"], cwd=t.worktree, check=True)
    subprocess.run(["git", "commit", "-qm", "work"], cwd=t.worktree, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    s.integrate("a")
    assert s.get("a").status is not Status.integrated
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True) == head   # merge reverted
    s.cfg.combined_suite = ["true"]
    assert s.get("a").status is Status.ready   # failed merge consumed an attempt and queued a retry
    s.tick()   # re-runs the agent in same worktree -> review -> passed
    s.tick()
    s.integrate("a")
    assert s.get("a").status is Status.integrated


def test_restart_keeps_state(make):
    s = make([mk("t", paths=["p"])], FakeAdapter(auto_finish=False))
    s.tick()
    s2 = type(s)(s.store, s.cfg, [mk("t", paths=["p"])], None, s.clock)
    assert s2.get("t").status is Status.running   # no manufactured completion, ownership retained
