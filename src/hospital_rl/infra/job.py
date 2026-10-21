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
from pathlib import Path
from typing import Callable

from hospital_rl.infra.artifacts import DirArtifactStore
from hospital_rl.infra.backend import JobSpec, JobState
from hospital_rl.policies.checkpoint import atomic_write
from hospital_rl.training.train import TrainConfig, control_from_file, train

STATE_BY_REASON = {"done": JobState.DONE, "drain": JobState.DRAINED, "stop": JobState.STOPPED}


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


def run_job(spec: JobSpec, root: str | Path, *, control: Callable[[], str | None] | None = None,
            after_save: Callable[[Path], None] = lambda p: None, log: Callable[[str], None] = print) -> dict:
    store = DirArtifactStore(root)
    out = store.attempt_dir(spec.run_id, spec.attempt_id)
    out.mkdir(parents=True, exist_ok=True)
    if spec.resume_from_attempt and not (out / "latest.pt").exists():
        prev = store.attempt_dir(spec.run_id, spec.resume_from_attempt)
        for name in ("latest.pt", "latest.pt.meta.json"):
            shutil.copyfile(prev / name, out / f".{name}.copy.tmp")
            (out / f".{name}.copy.tmp").replace(out / name)
    ctl = control or control_from_file(control_path(root, spec.run_id, spec.attempt_id))
    cfg = TrainConfig.model_validate(spec.train_config)
    # retries land in the same attempt dir and resume from its latest.pt, so steps are never double counted
    res = train(cfg, out, resume=(out / "latest.pt").exists(), control=ctl, log=log, after_save=after_save)
    result = {"state": STATE_BY_REASON[res.stop_reason].value, "env_steps": res.env_steps,
              "last_checkpoint": res.last_checkpoint.name, "final_return": res.history[-1]["mean_return"] if res.history else None}
    atomic_write(out / "result.json", json.dumps(result).encode())
    store.publish_manifest(spec.run_id, spec.attempt_id)
    after_save(out)
    return result


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
