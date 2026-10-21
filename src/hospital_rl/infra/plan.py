"""``cloud-plan``: dry run. Resolves config, shows hard caps and an *estimate*, checks credentials by
presence only (never reads values), and falls back to local with setup instructions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.bundle import resolve, validate

SECRETS = {  # names only; values come from the environment / provider config files at runtime
    "modal": {"env": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"], "file": ".modal.toml",
              "setup": "pip install modal && modal setup   (or export MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"},
    "daytona": {"env": ["DAYTONA_API_KEY"], "file": None,
                "setup": "create an API key in the Daytona dashboard and export DAYTONA_API_KEY"},
    "local": {"env": [], "file": None, "setup": ""},
}


class CloudPlan(BaseModel):
    requested_backend: str
    effective_backend: str
    config_path: str
    resolved: dict
    hard_caps: dict
    estimate: dict
    credentials: dict
    notes: list[str]


def credentials_present(backend: str, env=os.environ, home: Path | None = None) -> bool:
    s = SECRETS[backend]
    if not s["env"]:
        return True
    home = home or Path.home()
    return all(k in env for k in s["env"]) or bool(s["file"] and (home / s["file"]).exists())


def cloud_plan(backend: str, config_path: str | Path, env=os.environ, home: Path | None = None) -> CloudPlan:
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    if raw.get("backend", backend) != backend:
        raise ValueError(f"config {config_path} is for backend {raw['backend']!r}, not {backend!r}")
    caps = raw.get("caps", {})
    job = JobSpec(run_id="plan", attempt_id="plan", train_config=raw.get("train", {}),
                  **{k: caps[k] for k in ("timeout_s", "grace_s", "cpu", "memory_mb", "max_containers", "max_retries") if k in caps})
    if backend != "daytona":  # Daytona configs belong to its own owner; everything else must resolve to a runnable bundle
        validate(resolve(job))
    steps = job.train_config.get("total_steps", 0)
    core_hours_cap = job.cpu * job.max_containers * (1 + job.max_retries) * job.timeout_s / 3600
    est = {"label": "ESTIMATE - not provider-reported usage; no currency cost is inferred",
           "cpu_core_hours_upper_bound": round(core_hours_cap, 4),
           "upper_bound_provenance": f"cpu({job.cpu}) x max_containers({job.max_containers}) x (1+max_retries({job.max_retries})) x timeout_s({job.timeout_s})/3600; the caps cannot be exceeded",
           "expected_runtime_s": None, "expected_provenance": "none: no steps_per_sec given"}
    sps = (raw.get("estimate") or {}).get("steps_per_sec")
    if sps:
        est["expected_runtime_s"] = round(min(job.timeout_s, steps / sps), 1)
        est["expected_provenance"] = f"total_steps({steps}) / steps_per_sec({sps}); steps_per_sec is a user-supplied assumption (config estimate.steps_per_sec)"
    has_creds = ok = credentials_present(backend, env, home)
    s = SECRETS[backend]
    notes = []
    if not ok:
        notes.append(f"{backend} credentials not found; falling back to local. Setup: {s['setup']}")
    if backend == "daytona":
        notes.append("DEFERRED: Daytona backend is not implemented; falling back to local.")
        ok = False
    return CloudPlan(
        requested_backend=backend, effective_backend=backend if ok else "local", config_path=str(config_path),
        resolved={"backend": backend, **{k: v for k, v in raw.items() if k not in ("caps", "estimate")}, "job": job.model_dump(exclude={"run_id", "attempt_id"})},
        hard_caps={**caps, "max_steps": steps}, estimate=est,
        credentials={"secret_names": s["env"], "present": has_creds, "setup": s["setup"]},
        notes=notes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cloud-plan")
    p.add_argument("--backend", default="modal", choices=sorted(SECRETS))
    p.add_argument("--config", required=True)
    a = p.parse_args(argv)
    print(cloud_plan(a.backend, a.config).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
