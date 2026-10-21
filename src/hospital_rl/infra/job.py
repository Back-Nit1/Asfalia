"""The unit of work every backend runs: one bounded training attempt.

Layout under ``root`` (a local dir or a mounted Volume):
  <run>/<attempt>/        learner-owned outputs (checkpoints, result.json, manifest.json)
  <run>/control/<attempt> backend-owned control file: "drain" | "stop"   (kept apart from learner writes)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

from hospital_rl.infra.artifacts import MANIFEST, DirArtifactStore
from hospital_rl.infra.backend import JobSpec, JobState
from hospital_rl.infra.bundle import repo_root, resolve, validate
from hospital_rl.policies.checkpoint import atomic_write
from hospital_rl.training.train import TrainConfig, control_from_file, train

STATE_BY_REASON = {"done": JobState.DONE, "drain": JobState.DRAINED, "stop": JobState.STOPPED}
RESUME_FILES = {"train": ("latest.pt", "latest.pt.meta.json"),
                "school": ("latest.pt", "latest.pt.meta.json", "school_state.json", "best_score.json", "best.pt",
                           "best.pt.meta.json", "metrics.jsonl")}
PROGRESS = "progress.json"  # learner-side durable-checkpoint pointer; the backend reads it for freshness


def control_path(root: str | Path, run_id: str, attempt_id: str) -> Path:
    return Path(root) / run_id / "control" / attempt_id


def apply_limits(spec: JobSpec) -> None:
    """Best-effort self-imposed caps. RLIMIT_CPU is honored on Linux and macOS; RLIMIT_AS only on Linux
    (macOS ignores it, so memory is NOT enforced locally there)."""
    import resource
    cpu_s = int(spec.timeout_s * spec.cpu) + spec.grace_s
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
    if sys.platform.startswith("linux"):
        mem = spec.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))


def _resume_from(store: DirArtifactStore, spec: JobSpec, out: Path) -> None:
    """Seed a new attempt from a previous one. A finished source must verify; a killed one (no manifest) is
    usable because checkpoints are published atomically."""
    prev = store.attempt_dir(spec.run_id, spec.resume_from_attempt)
    if (prev / MANIFEST).exists() and (problems := store.verify_manifest(spec.run_id, spec.resume_from_attempt)):
        raise RuntimeError(f"resume source {spec.resume_from_attempt} failed manifest verification: {problems}")
    for name in RESUME_FILES[spec.kind]:
        if not (prev / name).exists():
            if name in ("latest.pt", "latest.pt.meta.json"):
                raise FileNotFoundError(f"resume source {spec.resume_from_attempt} has no {name}")
            continue
        shutil.copyfile(prev / name, out / f".{name}.copy.tmp")
        (out / f".{name}.copy.tmp").replace(out / name)


def _write_progress(out: Path, env_steps: int | None = None) -> None:
    meta = out / "latest.pt.meta.json"
    if env_steps is None and meta.exists():
        env_steps = json.loads(meta.read_text()).get("training_steps")
    atomic_write(out / PROGRESS, json.dumps({"checkpoint": "latest.pt", "env_steps": env_steps,
                                              "written_at": time.time()}).encode())


def _run_school(spec: JobSpec, out: Path, ctl: Callable[[], str | None], hook: Callable[[Path], None], log) -> tuple[str, int]:
    """Role x seed school under the same control contract. ``run_school`` resumes from school_state.json and does
    exactly one validation chunk when asked for ``done + eval_every`` steps, so drain/stop are honored at every
    chunk boundary (granularity = eval_every env steps) without touching the training code."""
    from hospital_rl.training.schools import run_school

    c = spec.train_config
    cfg, every = str(repo_root() / c["config"]), c.get("eval_every", 8192)
    state = out / "school_state.json"
    done = json.loads(state.read_text())["steps"] if state.exists() else 0
    reason = "done"
    while done < c["steps"]:
        if (sig := ctl()) in ("drain", "stop"):
            reason = sig
            break
        r = run_school(c["role"], cfg, min(done + every, c["steps"]), c["seed"], out, every, c.get("n_val", 20),
                       c.get("n_envs", 8), c.get("rollout_len", 64), ppo=c.get("ppo") or None, log=log)
        done = r["steps"]
        hook(out / "latest.pt")
    return reason, done


def run_job(spec: JobSpec, root: str | Path, *, control: Callable[[], str | None] | None = None,
            after_save: Callable[[Path], None] = lambda p: None, log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    spec = resolve(spec) if not spec.config_hash else spec
    validate(spec, check_code=bool(spec.code_hash))
    store = DirArtifactStore(root)
    out = store.attempt_dir(spec.run_id, spec.attempt_id)
    out.mkdir(parents=True, exist_ok=True)
    if (out / MANIFEST).exists() and (out / "result.json").exists():  # a retry after completion must not redo or re-publish
        return json.loads((out / "result.json").read_text())
    if spec.resume_from_attempt and not (out / "latest.pt").exists():
        _resume_from(store, spec, out)
    file_ctl = control or control_from_file(control_path(root, spec.run_id, spec.attempt_id))
    timed_out = False

    def ctl() -> str | None:
        """Soft deadline at timeout_s: ask for checkpoint-and-stop so the checkpoint lands inside the grace window
        before the hard kill (local client / Modal function timeout = timeout_s + grace_s)."""
        nonlocal timed_out
        if time.time() - t0 >= spec.timeout_s:
            timed_out = True
            return "stop"
        return file_ctl()

    def hook(p: Path) -> None:
        _write_progress(out)
        after_save(p)

    if spec.kind == "school":
        reason, steps = _run_school(spec, out, ctl, hook, log)
        result = {"env_steps": steps, "last_checkpoint": "latest.pt", "final_return": None}
    else:
        cfg = TrainConfig.model_validate(spec.train_config)
        # retries land in the same attempt dir and resume from its latest.pt, so steps are never double counted
        res = train(cfg, out, resume=(out / "latest.pt").exists(), control=ctl, log=log, after_save=hook)
        reason = res.stop_reason
        result = {"env_steps": res.env_steps, "last_checkpoint": res.last_checkpoint.name,
                  "final_return": res.history[-1]["mean_return"] if res.history else None}
    state = JobState.TIMED_OUT if timed_out and reason == "stop" else STATE_BY_REASON[reason]
    _write_progress(out, result["env_steps"])
    atomic_write(out / "result.json", json.dumps({"state": state.value, **result}).encode())
    store.publish_manifest(spec.run_id, spec.attempt_id)
    after_save(out)
    return {"state": state.value, **result}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.infra.job")
    p.add_argument("--spec", required=True)
    p.add_argument("--root", required=True)
    a = p.parse_args(argv)
    spec = JobSpec.model_validate_json(Path(a.spec).read_text())
    apply_limits(spec)
    run_job(spec, a.root, log=lambda s: print(s, file=sys.stderr, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
