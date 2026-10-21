"""Run school training (role x seed grid) off-device, one container/sandbox per run, all in parallel.

Modal runs through the common job lifecycle (``lifecycle.GatewayBackend`` + ``modal_app``): every role x seed is one
``kind="school"`` job with caps, attempt ids, a manifest and drain / stop / cancel / resume like any other job.
    python -m hospital_rl.infra.cloud_school modal --steps 200000 --seeds 7,8,9        # ephemeral: keep this process up
    python -m hospital_rl.infra.cloud_school modal --detach ...                         # deployed app; submit and exit
    python -m hospital_rl.infra.cloud_school status --tag <tag>                         # re-attach from the registry
    modal volume get hospital-rl-school /<tag>-<role>-s<seed> runs/cloud-<tag>-<role>-s<seed>
Re-running a tag starts a new attempt that resumes from the previous attempt (checkpoint + school_state.json).
Daytona (common backend; client must stay up, verified artifacts are mirrored into runs/daytona/store/<run>/<attempt>):
    python -m hospital_rl.infra.cloud_school daytona --steps 50000 --seeds 7,8
Secrets: Modal auth via ~/.modal.toml; Daytona via DAYTONA_API_KEY in the process environment (no .env is read; never printed).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hospital_rl.infra.backend import JobSpec

CONFIG = {"attacker": "configs/attacker_school.yaml", "defender": "configs/defender_school.yaml"}
APP = "hospital-rl-school"


def school_specs(a, backend) -> list[JobSpec]:
    """One spec per role x seed. A re-used tag gets the next attempt id, resuming from the previous terminal attempt."""
    tag, specs = a.tag or time.strftime("%Y%m%d-%H%M%S"), []
    for role in a.roles.split(","):
        for seed in map(int, a.seeds.split(",")):
            run = f"{tag}-{role}-s{seed}"
            prev = sorted((r for r in backend.reg.all() if r["run_id"] == run), key=lambda r: r["submitted_at"])
            specs.append(JobSpec(
                run_id=run, attempt_id=f"a{len(prev) + 1}", kind="school", resume_from_attempt=prev[-1]["attempt_id"] if prev else None,
                train_config=dict(role=role, config=CONFIG[role], steps=a.steps, seed=seed, eval_every=a.eval_every,
                                  ppo={"lr": a.lr, "ent_coef": a.ent_coef}),
                timeout_s=a.timeout, grace_s=120, cpu=2, memory_mb=4096))
    return specs


def modal_main(a) -> None:
    from hospital_rl.infra.lifecycle import ResourceCaps
    from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, RealModalGateway

    n = len(a.roles.split(",")) * len(a.seeds.split(","))
    # ponytail: 2 cpu, SyncVecEnv is single-process; scale by more seeds, not bigger boxes.
    cfg = ModalConfig(app_name=APP, volume_name=APP, mode="deployed" if a.detach else "ephemeral", registry_dir="runs/.registry/school",
                      caps=ResourceCaps(cpu=2, memory_mb=4096, timeout_s=a.timeout, grace_s=120, max_containers=n, max_steps=a.steps))
    gw = RealModalGateway(cfg)
    if a.detach and not a.no_deploy:
        gw.deploy()
    be = ModalBackend(cfg, gw)
    ids = [be.submit(sp) for sp in school_specs(a, be)]
    for j in ids:
        print(f"submitted {j} call={be._rec(j)['remote_id']}")
    if a.detach:
        print("detached: poll with `cloud_school status --tag <tag>`")
        return
    try:
        while not all((st := be.status(j)).state.terminal for j in ids):
            time.sleep(10)
        for j in ids:
            st = be.status(j)
            print(f"{j} {st.state.value} {st.result.get('env_steps')} steps {st.detail}")
    finally:
        gw.close()  # stop the ephemeral app (and its containers) when this controller leaves


def status_main(a) -> None:
    from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, RealModalGateway

    cfg = ModalConfig(app_name=APP, volume_name=APP, mode="deployed" if a.detach else "ephemeral", registry_dir="runs/.registry/school")
    be = ModalBackend(cfg, RealModalGateway(cfg))
    for r in be.reg.all():
        if r["run_id"].startswith(a.tag):
            st = be.status(r["job_id"])
            print(f"{r['job_id']} {st.state.value} ckpt={st.checkpoint} {st.detail}")


def daytona_main(a) -> None:
    """Thin caller of the common Daytona backend: same caps, attempt ids, mirror verification and resume as Modal.
    Credentials come from the process environment only (the SDK reads DAYTONA_API_KEY); no .env is read."""
    from hospital_rl.infra.daytona_backend import DaytonaBackend, DaytonaConfig, DaytonaGateway, SdkClient
    from hospital_rl.infra.lifecycle import ResourceCaps

    n = len(a.roles.split(",")) * len(a.seeds.split(","))
    cfg = DaytonaConfig(registry_dir="runs/.registry/school-daytona", mirror_dir="runs/daytona/store",
                        caps=ResourceCaps(cpu=2, memory_mb=4096, timeout_s=a.timeout, grace_s=120, max_containers=n, max_steps=a.steps))
    be = DaytonaBackend(cfg, DaytonaGateway(cfg, SdkClient(cfg)))
    ids = [be.submit(sp) for sp in school_specs(a, be)]
    while not all((st := be.status(j)).state.terminal for j in ids):
        time.sleep(10)
    for j in ids:
        st = be.status(j)
        print(f"{j} {st.state.value} {st.result.get('env_steps')} steps {st.detail}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["daytona", "modal", "status"])
    p.add_argument("--detach", action="store_true", help="modal: use a deployed app so jobs outlive this process (NOT live-verified)")
    p.add_argument("--no-deploy", action="store_true", help="modal --detach: attach to the already deployed app")
    p.add_argument("--eval-every", type=int, default=8192)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seeds", default="7")
    p.add_argument("--roles", default="attacker,defender")
    p.add_argument("--tag", default="")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--timeout", type=int, default=6 * 3600)
    args = p.parse_args()
    {"daytona": daytona_main, "modal": modal_main, "status": status_main}[args.backend](args)
