"""python -m hospital_rl.api --host 127.0.0.1 --port 8000 [--plan configs/orchestrator.yaml --run-scheduler]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from ..orchestration.adapters import default_adapter
from ..orchestration.contracts import PlanConfig
from ..orchestration.scheduler import Scheduler
from ..orchestration.tasks import load_plan
from .app import FIXTURES, create_app
from .auth import AuthConfig
from .replay import Library
from .storage import Store


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hospital_rl.api")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default="runs/dashboard.sqlite")
    p.add_argument("--plan", default=None, help="orchestrator YAML (task DAG)")
    p.add_argument("--episodes-dir", action="append", default=[], help="extra directory of episode JSON exports")
    p.add_argument("--run-scheduler", action="store_true", help="tick the scheduler in this process")
    p.add_argument("--no-agent", action="store_true", help="never start the coding runtime (human-owned tasks only)")
    p.add_argument("--stream-interval", type=float, default=0.25, help="SSE poll seconds (0.2-0.5 => 2-5 Hz)")
    p.add_argument("--experiments-root", default=None, help="enable the experiment controller; job artifacts live here")
    p.add_argument("--fake-backend", action="store_true", help="with --experiments-root: deterministic FAKE backend+evaluator (no training)")
    p.add_argument("--world-config", default="configs/hospital40.yaml", help="shared-world config for real frozen evaluations")
    p.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    p.add_argument("--tick-seconds", type=float, default=2.0)
    p.add_argument("--static-dir", default="frontend/dist")
    a = p.parse_args(argv)
    auth = AuthConfig.for_bind(a.host, os.environ.get("HOSPITAL_RL_OPERATOR_TOKEN"), os.environ.get("HOSPITAL_RL_EVALUATOR_TOKEN"))
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    store = Store(a.db)
    cfg, tasks = load_plan(a.plan) if a.plan else (PlanConfig(), [])
    sched = Scheduler(store, cfg, tasks, None if a.no_agent else default_adapter())
    controller, eps = None, list(map(Path, a.episodes_dir))
    if a.experiments_root:
        from ..experiments.controller import Controller
        from ..experiments.reader import FsReader
        root = Path(a.experiments_root)
        eps.append(root / "episodes")
        if a.fake_backend:
            from ..experiments.fakes import FakeBackend, FakeEvaluator
            backend, evaluator = FakeBackend(root / "jobs"), FakeEvaluator(FIXTURES / "fixture_tiny_episode.json")
        else:
            from ..experiments.evaluator import FrozenEvaluator
            from ..infra.local_backend import LocalBackend
            backend = LocalBackend(root / "jobs", registry=root / ".registry" / "local")
            evaluator = FrozenEvaluator(a.world_config, a.evaluation_split)
        controller = Controller(store, backend, FsReader(root / "jobs"), evaluator=evaluator, episodes_dir=str(root / "episodes"))
    app = create_app(store, sched, auth, Library(FIXTURES, *eps), a.stream_interval, a.static_dir, a.run_scheduler,
                     tick_seconds=a.tick_seconds, controller=controller)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
