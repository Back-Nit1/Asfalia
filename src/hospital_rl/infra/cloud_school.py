"""Run school training (role x seed grid) off-device, one container/sandbox per run, all in parallel.

Modal runs through the common job lifecycle (``lifecycle.GatewayBackend`` + ``modal_app``): every role x seed is one
``kind="school"`` job with caps, attempt ids, a manifest and drain / stop / cancel / resume like any other job.
    python -m hospital_rl.infra.cloud_school modal --steps 200000 --seeds 7,8,9        # ephemeral: keep this process up
    python -m hospital_rl.infra.cloud_school modal --detach ...                         # deployed app; submit and exit
    python -m hospital_rl.infra.cloud_school status --tag <tag>                         # re-attach from the registry
    modal volume get hospital-rl-school /<tag>-<role>-s<seed> runs/cloud-<tag>-<role>-s<seed>
Re-running a tag starts a new attempt that resumes from the previous attempt (checkpoint + school_state.json).
Daytona (sandboxes; client must stay up, results are downloaded into runs/daytona/<tag>):
    python -m hospital_rl.infra.cloud_school daytona --steps 50000 --seeds 7,8
Secrets: Modal auth via ~/.modal.toml; DAYTONA_API_KEY/DAYTONA_API_URL read from env or .env (never printed).
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.bundle import DEPS, TORCH_CPU  # light imports only: Daytona-only installs have no torch

ROOT = Path(__file__).resolve().parents[min(3, len(Path(__file__).resolve().parents) - 1)]  # container: /root/x.py
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


def _bundle() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(ROOT / "src/hospital_rl", "src/hospital_rl", filter=lambda i: None if "__pycache__" in i.name else i)
        t.add(ROOT / "configs", "configs")
    return buf.getvalue()


def _daytona_job(client, bundle: bytes, j: dict, dest: Path) -> str:
    from daytona import CreateSandboxFromImageParams, Image, Resources

    sb = client.create(CreateSandboxFromImageParams(
        image=Image.debian_slim("3.12"), resources=Resources(cpu=4, memory=8, disk=10),
        auto_delete_interval=0, labels={"app": "hospital-rl"}), timeout=300)
    try:
        sb.fs.upload_file(bundle, "/home/daytona/b.tgz")
        sb.process.exec(f"tar xzf b.tgz && pip install -q torch --index-url {TORCH_CPU} && pip install -q {' '.join(DEPS)}",
                        cwd="/home/daytona", timeout=1800)
        run = sb.process.exec(
            f"PYTHONPATH=src python -m hospital_rl.training.schools --role {j['role']} --config {CONFIG[j['role']]} "
            f"--steps {j['steps']} --seed {j['seed']} --lr {j['lr']} --ent-coef {j['ent_coef']} --out out",
            cwd="/home/daytona", timeout=j["timeout"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "stdout.txt").write_text(run.result)
        for f in ("best.pt", "best.pt.meta.json", "metrics.jsonl", "school_state.json"):
            if data := sb.fs.download_file(f"/home/daytona/out/{f}"):
                (dest / f).write_bytes(data)
        return f"{dest} exit={run.exit_code}"
    finally:
        sb.delete()


def daytona_main(a) -> None:
    from daytona import Daytona

    if not os.environ.get("DAYTONA_API_KEY"):
        from dotenv import dotenv_values
        os.environ.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if k.startswith("DAYTONA") and v})
    client, bundle, tag = Daytona(), _bundle(), a.tag or time.strftime("%Y%m%d-%H%M%S")
    jobs = [dict(role=r, seed=int(s), steps=a.steps, lr=a.lr, ent_coef=a.ent_coef, timeout=a.timeout)
            for r in a.roles.split(",") for s in a.seeds.split(",")]
    with ThreadPoolExecutor(len(jobs)) as ex:
        futs = [ex.submit(_daytona_job, client, bundle, j, ROOT / "runs/daytona" / tag / f"{j['role']}-s{j['seed']}")
                for j in jobs]
        for f in futs:
            print(f.result())


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
