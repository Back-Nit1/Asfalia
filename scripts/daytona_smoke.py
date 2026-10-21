"""Capped Daytona smoke. Three outcomes, always labelled:  local (no opt-in / no credentials)  |  live (opted in).

    python scripts/daytona_smoke.py                      # dry run: caps, estimate, credential presence by NAME, setup help
    python scripts/daytona_smoke.py --local              # tiny local job on the same JobSpec contract (the fallback)
    ASFALIA_DAYTONA_SMOKE=1 python scripts/daytona_smoke.py --live --max-minutes 5 --max-cpu 1 --max-steps 64 --yes

--live needs ALL of: ASFALIA_DAYTONA_SMOKE=1, --yes, explicit caps, DAYTONA_API_KEY present in the environment. One
sandbox per step (graph job, then workspace test), both deleted on exit. It never reads .env or prints secret values.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from hospital_rl.infra.plan import SECRETS, cloud_plan, credentials_present

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cloud_daytona.yaml"


def tiny_spec(steps: int, minutes: int, cpu: float):
    from hospital_rl.infra.backend import JobSpec

    tc = dict(role="defender", total_steps=steps, n_envs=2, rollout_len=16, hidden=16, checkpoint_every=1,
              env_factory="hospital_rl.envs.factories:hospital_single", ppo={"epochs": 1, "minibatches": 1},
              env_kwargs=dict(role="defender", size=8, split="train", difficulty=0.5, horizon=20, opponent="scripted"))
    return JobSpec(run_id="daytona-smoke", attempt_id=f"a{int(__import__('time').time())}", train_config=tc,
                   timeout_s=minutes * 60, grace_s=30, cpu=cpu, memory_mb=2048, max_retries=0)


def local_run(steps: int) -> dict:
    from hospital_rl.infra.local_backend import LocalBackend

    with tempfile.TemporaryDirectory() as d:
        be = LocalBackend(Path(d) / "runs")
        st = be.wait(be.submit(tiny_spec(steps, 5, 1)), 300)
        return {"executed": "local", "state": st.state.value, "result": st.result}


def live_run(a) -> dict:
    from hospital_rl.infra.daytona_backend import DaytonaBackend, DaytonaConfig, DaytonaGateway, SdkClient
    from hospital_rl.infra.daytona_workspace import workspace_smoke
    from hospital_rl.infra.lifecycle import ResourceCaps
    from hospital_rl.infra.reconcile import reconcile

    cfg = DaytonaConfig(caps=ResourceCaps(cpu=a.max_cpu, memory_mb=2048, timeout_s=a.max_minutes * 60, grace_s=30,
                                          max_containers=1, max_retries=0, max_steps=a.max_steps))
    client = SdkClient(cfg)
    be = DaytonaBackend(cfg, DaytonaGateway(cfg, client))
    out: dict = {"executed": "live"}
    try:
        st = be.wait(be.submit(tiny_spec(a.max_steps, a.max_minutes, a.max_cpu)), a.max_minutes * 60 + 60)
        out["graph_job"] = {"state": st.state.value, "result": st.result, "detail": st.detail}
        out["workspace"] = workspace_smoke(client, ROOT, "HEAD", "tests/test_events.py", ROOT / "runs/daytona/workspace",
                                           timeout_s=a.max_minutes * 60, cpu=int(a.max_cpu))
    finally:
        out["cleanup"] = reconcile(be, dry_run=True)["orphans_not_deleted"]  # leftover owned sandboxes, if any
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--live", action="store_true")
    p.add_argument("--local", action="store_true")
    p.add_argument("--yes", action="store_true", help="explicit authorization for the capped live run")
    p.add_argument("--max-minutes", type=int)
    p.add_argument("--max-cpu", type=float)
    p.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    plan = cloud_plan("daytona", CONFIG)
    if not (a.live and a.yes and os.environ.get("ASFALIA_DAYTONA_SMOKE") == "1" and None not in (a.max_minutes, a.max_cpu, a.max_steps)):
        print(json.dumps({"executed": "none", "live_run": "NOT EXECUTED (needs --live --yes, ASFALIA_DAYTONA_SMOKE=1 and --max-minutes/--max-cpu/--max-steps)",
                          "credentials_present": credentials_present("daytona"), "setup": SECRETS["daytona"]["setup"],
                          "hard_caps": plan.hard_caps, "estimate": plan.estimate}, indent=1))
        if a.local:
            print(json.dumps(local_run(64), indent=1))
        return 0
    if not credentials_present("daytona"):
        print(f"DAYTONA_API_KEY missing. Setup: {SECRETS['daytona']['setup']}\nRunning the local fallback instead:")
        print(json.dumps(local_run(a.max_steps), indent=1))
        return 0
    print(json.dumps(live_run(a), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
