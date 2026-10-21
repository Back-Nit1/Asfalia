"""Tiny REAL school checkpoints (actual hospital envs, no toy env) shared by training and evaluation tests."""
from pathlib import Path

import pytest

from hospital_rl.training.schools import run_school

WORLD = "configs/experiments/smoke_world.yaml"
ROOT = Path(__file__).resolve().parents[2]


def make_school_ckpts(root: Path) -> dict[str, str]:
    out = {}
    for r in ("attacker", "defender"):
        res = run_school(r, str(ROOT / f"configs/{r}_school.yaml"), 128, 1, root / r, eval_every=128, n_val=2,
                         n_envs=2, rollout_len=32, log=lambda s: None)
        out[r] = res["best"]
    return out


@pytest.fixture(scope="session")
def school_ckpts(tmp_path_factory) -> dict[str, str]:
    return make_school_ckpts(tmp_path_factory.mktemp("schools"))


@pytest.fixture(scope="session")
def world_path() -> str:
    return str(ROOT / WORLD)
