"""Presets, validation, resource caps and the (clearly labelled) estimate. Validation never starts anything."""
from __future__ import annotations

from pydantic import BaseModel, Field

# Hard ceilings for anything started from the dashboard; a preset can only be tightened, never loosened.
MAX_STEPS, MAX_TIMEOUT_S, MAX_CPU, MAX_MEMORY_MB, MAX_CONTAINERS = 200_000, 3600, 4.0, 8192, 1
FACTORY = "hospital_rl.envs.factories:school_env"

PRESETS: dict[str, dict] = {
    "attacker-school-smoke": {"role": "attacker", "school": "configs/attacker_school.yaml", "total_steps": 2048},
    "defender-school-smoke": {"role": "defender", "school": "configs/defender_school.yaml", "total_steps": 2048},
}


class RunRequest(BaseModel):
    preset: str
    seed: int = 0
    total_steps: int | None = None
    timeout_s: int = 300
    grace_s: int = 30
    cpu: float = 1.0
    memory_mb: int = 2048
    backend: str | None = None          # must match the configured backend when given


class Caps(BaseModel):
    total_steps: int
    timeout_s: int
    grace_s: int
    cpu: float
    memory_mb: int
    max_containers: int = 1


class Estimate(BaseModel):
    steps_per_sec: float | None
    seconds: float | None
    basis: str                       # "measured" | "unavailable"
    measured_at: float | None = None
    note: str = ""


class Resolved(BaseModel):
    preset: str
    role: str
    seed: int
    train_config: dict
    caps: Caps
    estimate: Estimate
    warnings: list[str] = Field(default_factory=list)


def resolve(req: RunRequest, backend_name: str, measured: tuple[float, float] | None = None) -> Resolved:
    """Raises ValueError listing every problem. `measured` = (steps_per_sec, measured_at) from stored real metrics only."""
    errs: list[str] = []
    p = PRESETS.get(req.preset)
    if p is None:
        raise ValueError(f"unknown preset {req.preset!r}; known: {sorted(PRESETS)}")
    steps = req.total_steps or p["total_steps"]
    for name, val, hi in (("total_steps", steps, MAX_STEPS), ("timeout_s", req.timeout_s, MAX_TIMEOUT_S),
                          ("cpu", req.cpu, MAX_CPU), ("memory_mb", req.memory_mb, MAX_MEMORY_MB)):
        if not 0 < val <= hi:
            errs.append(f"{name}={val} outside (0, {hi}]")
    if req.grace_s < 0 or req.grace_s > req.timeout_s:
        errs.append("grace_s must be within [0, timeout_s]")
    if req.backend and req.backend != backend_name:
        errs.append(f"backend {req.backend!r} is not the configured backend {backend_name!r}")
    if errs:
        raise ValueError("; ".join(errs))
    tc = {"role": p["role"], "seed": req.seed, "total_steps": steps, "n_envs": 2, "rollout_len": 32, "checkpoint_every": 2,
          "env_factory": FACTORY, "env_kwargs": {"config": p["school"], "level": 0, "split": "train"}}
    est = (Estimate(steps_per_sec=measured[0], seconds=steps / measured[0], basis="measured", measured_at=measured[1],
                    note="from this installation's stored metrics; not a guarantee") if measured and measured[0] > 0
           else Estimate(steps_per_sec=None, seconds=None, basis="unavailable",
                         note="no completed measurement yet; no estimate is invented"))
    return Resolved(preset=req.preset, role=p["role"], seed=req.seed, train_config=tc, estimate=est,
                    caps=Caps(total_steps=steps, timeout_s=req.timeout_s, grace_s=req.grace_s, cpu=req.cpu, memory_mb=req.memory_mb))
