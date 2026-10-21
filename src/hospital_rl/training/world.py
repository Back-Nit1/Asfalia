"""Shared-world configuration: every field of the YAML is resolved (unknown keys are errors) and hashed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from ..envs.factories import hospital_penv
from ..envs.parallel_env import HospitalParallelEnv


class WorldConfigError(ValueError):
    pass


class WorldConfig(BaseModel, extra="forbid"):
    role: Literal["shared"] = "shared"
    size: int = 40
    horizon: int = 60
    difficulty: float = 0.7
    base_logging: int = 0

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


def load_world(path: str | Path) -> WorldConfig:
    p = Path(path)
    if not p.is_file():
        raise WorldConfigError(f"shared-world config not found: {p}")
    try:
        return WorldConfig.model_validate(yaml.safe_load(p.read_text()) or {})
    except (ValidationError, yaml.YAMLError) as e:
        raise WorldConfigError(f"invalid shared-world config {p}: {e}") from e


def world_penv(world: WorldConfig, split: str) -> HospitalParallelEnv:
    """Two-role env; reset(seed) maps the seed into the split's range and draws family/workload from the split."""
    return hospital_penv(world.size, split, world.difficulty, world.horizon, world.base_logging)
