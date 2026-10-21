"""Run school training (role x seed grid) off-device, one container/sandbox per run, all in parallel.

Modal (primary; results land on Volume ``hospital-rl-school`` and survive the client):
    modal run -d src/hospital_rl/infra/cloud_school.py --steps 200000 --seeds 7,8,9
    modal volume get hospital-rl-school /<tag> runs/cloud-<tag>        # tag printed at launch
Daytona (sandboxes; client must stay up, results are downloaded into runs/daytona/<tag>):
    python -m hospital_rl.infra.cloud_school daytona --steps 50000 --seeds 7,8
Re-running with the same tag resumes from the Volume (run_school resumes from school_state.json).
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

ROOT = Path(__file__).resolve().parents[min(3, len(Path(__file__).resolve().parents) - 1)]  # container: /root/x.py
CONFIG = {"attacker": "configs/attacker_school.yaml", "defender": "configs/defender_school.yaml"}
DEPS = ["numpy", "networkx", "gymnasium", "pettingzoo", "pydantic>=2", "pyyaml"]
TORCH_CPU = "https://download.pytorch.org/whl/cpu"  # default linux torch wheel drags in ~3GB of CUDA

try:
    import modal
except ImportError:  # daytona-only installs
    modal = None

if modal:
    app = modal.App("hospital-rl-school")
    vol = modal.Volume.from_name("hospital-rl-school", create_if_missing=True)
    image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch", index_url=TORCH_CPU)
             .pip_install(*DEPS).add_local_dir(ROOT / "configs", "/root/configs")
             .add_local_python_source("hospital_rl"))

    # ponytail: 2 cpu, SyncVecEnv is single-process; scale by more seeds, not bigger boxes.
    @app.function(image=image, volumes={"/runs": vol}, cpu=2, memory=4096, timeout=24 * 3600)
    def school(job: dict) -> dict:
        from hospital_rl.training.schools import run_school

        out = Path("/runs") / job["tag"] / f"{job['role']}-s{job['seed']}"
        try:
            return run_school(job["role"], CONFIG[job["role"]], job["steps"], job["seed"], out, job["eval_every"],
                              ppo={"lr": job["lr"], "ent_coef": job["ent_coef"]}, log=lambda s: print(s, flush=True))
        finally:
            vol.commit()  # keep partial progress so a rerun resumes

    @app.local_entrypoint()
    def main(steps: int = 200_000, seeds: str = "7", roles: str = "attacker,defender", tag: str = "",
             lr: float = 1e-3, ent_coef: float = 0.01, eval_every: int = 8192):
        tag = tag or time.strftime("%Y%m%d-%H%M%S")
        jobs = [dict(role=r, seed=int(s), steps=steps, tag=tag, lr=lr, ent_coef=ent_coef, eval_every=eval_every)
                for r in roles.split(",") for s in seeds.split(",")]
        for j in jobs:
            print(f"spawned {j['role']} seed={j['seed']} call={school.spawn(j).object_id}")
        print(f"tag={tag}  fetch: modal volume get hospital-rl-school /{tag} runs/cloud-{tag}")


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
    p.add_argument("backend", choices=["daytona"])
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--seeds", default="7")
    p.add_argument("--roles", default="attacker,defender")
    p.add_argument("--tag", default="")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--timeout", type=int, default=6 * 3600)
    daytona_main(p.parse_args())
