"""python -m hospital_rl.orchestration --plan configs/orchestrator.yaml --dry-run"""
from __future__ import annotations

import argparse

from .adapters import default_adapter
from .tasks import load_plan, paths_overlap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.orchestration")
    p.add_argument("--plan", default="configs/orchestrator.yaml")
    p.add_argument("--dry-run", action="store_true", required=True, help="print the DAG and runtime status; live runs go through the dashboard process (--run-scheduler)")
    a = p.parse_args(argv)
    cfg, tasks = load_plan(a.plan)
    adapter = default_adapter()
    print(f"max_concurrency={cfg.max_concurrency} retry_cap={cfg.retry_cap} base={cfg.base_commit}")
    print("coding runtime:", adapter.capabilities().name if adapter else "coding runtime not configured")
    for t in tasks:
        clash = [o.id for o in tasks if o.id != t.id and paths_overlap(t.assigned_paths, o.assigned_paths)]
        print(f"- {t.id} [{t.owner}] deps={t.dependencies or '-'} paths={len(t.assigned_paths)}" + (f" OVERLAPS {clash}" if clash else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
