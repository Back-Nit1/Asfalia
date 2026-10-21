"""Experiment runner: schools -> frozen Stage D transfer -> shared-world adaptation arms -> comparison manifest.

    python -m hospital_rl.training.experiments run --config configs/experiments/smoke.yaml --out OUT
    python -m hospital_rl.training.experiments compare --rule equal_total RUN_A RUN_B

Exit codes: 0 ok, 2 bad input / incompatible checkpoint / lineage conflict / incomparable runs.
Arms: ``pretrained`` (school weights), ``scratch_same_adapt`` (random init, same adapt budget),
``scratch_equal_total`` (random init, adapt budget + the trainable role's school steps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from ..evaluation.transfer import CheckpointError, load_actor, report
from ..evaluation.strata import StratumError, check_training_config
from ..policies.checkpoint import IncompatibleCheckpointError, _code_revision
from .adapt import OTHER, _require, adapt
from .lineage import LineageError, claim_run
from .schools import run_school
from .world import WorldConfigError, load_world

ARMS = ("pretrained", "scratch_same_adapt", "scratch_equal_total")
RULES = {"equal_adapt": "adapt_env_steps", "equal_total": "total_env_steps"}


class ComparisonError(ValueError):
    """Runs are not comparable; they are rejected, never averaged."""


class School(BaseModel, extra="forbid"):
    attacker_config: str
    defender_config: str
    steps: int
    eval_every: int = 4096
    val_episodes: int = 20
    n_envs: int = 8
    rollout_len: int = 64


class AdaptCfg(BaseModel, extra="forbid"):
    steps: int
    eval_every: int = 2048
    val_episodes: int = 10
    n_envs: int = 4
    rollout_len: int = 64


class Transfer(BaseModel, extra="forbid"):
    split: Literal["train", "validation"] = "validation"   # never "test": no tuning/selection there
    episodes: int = 20


class ExperimentConfig(BaseModel, extra="forbid"):
    name: str
    world_config: str
    seed: int
    trainable_role: Literal["attacker", "defender"]
    league: Literal["on", "off"] = "off"
    school: School | None = None
    checkpoints: dict[str, str] | None = None          # explicit pretrained checkpoints instead of running schools
    adapt: AdaptCfg | None = None
    transfer: Transfer = Transfer()
    arms: list[Literal["pretrained", "scratch_same_adapt", "scratch_equal_total"]] = []

    def hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def load_experiment(path: str) -> ExperimentConfig:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"experiment config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
        check_training_config(raw, str(p))
        c = ExperimentConfig.model_validate(raw)
    except (ValidationError, yaml.YAMLError, StratumError) as e:
        raise ValueError(f"invalid experiment config {p}: {e}") from e
    if bool(c.school) == bool(c.checkpoints):
        raise ValueError("exactly one of `school` (train both roles) or `checkpoints` (explicit paths) is required")
    if c.arms and not c.adapt:
        raise ValueError("`arms` needs an `adapt` block")
    return c


def compare(manifests: list[dict], rule: str) -> dict:
    """Pairwise comparison of finished adapt runs; raises ComparisonError instead of averaging mismatches."""
    if rule not in RULES:
        raise ComparisonError(f"unknown budget rule {rule!r}; use one of {sorted(RULES)}")
    if len(manifests) < 2:
        raise ComparisonError("need at least two runs to compare")
    for m in manifests:
        if m.get("kind") != "adapt":
            raise ComparisonError(f"only one-side adapt runs are comparable, got kind={m.get('kind')!r}")
        b = m["budget"]
        if b["adapt_env_steps"] < b["requested_adapt_steps"]:
            raise ComparisonError(f"run {m['checkpoints']['best_path']} is unfinished "
                                  f"({b['adapt_env_steps']}/{b['requested_adapt_steps']} steps)")
        if not m["opponent_frozen"]["unchanged"]:
            raise ComparisonError(f"run {m['checkpoints']['best_path']}: frozen opponent checkpoint changed")
    ref = manifests[0]["comparability"]
    for m in manifests[1:]:
        bad = [f"{k}: {ref[k]!r} != {m['comparability'][k]!r}" for k in ref if ref[k] != m["comparability"][k]]
        if bad:
            raise ComparisonError("incomparable runs - " + "; ".join(bad))
    key = RULES[rule]
    budgets = {m["checkpoints"]["best_path"]: m["budget"][key] for m in manifests}
    if len(set(budgets.values())) != 1:
        raise ComparisonError(f"budget rule {rule} ({key}) violated: {budgets}")
    return {"rule": rule, "budget_key": key, "budget": next(iter(budgets.values())), "comparability": ref,
            "runs": [{"init": m["identity"]["init"], "best_checkpoint": m["checkpoints"]["best"],
                      "budget": m["budget"], "best_validation_score": m["best_validation_score"],
                      "last_validation": m["validation"]} for m in manifests],
            "note": "validation split, single seed, uncalibrated rewards: descriptive only, no superiority claim"}


def run_experiment(cfg: ExperimentConfig, out: str | Path, log=print) -> dict:
    out = Path(out)
    load_world(cfg.world_config)                                   # fail early on a bad world
    claim_run(out, {"kind": "experiment", "name": cfg.name, "config_hash": cfg.hash()})
    role = cfg.trainable_role
    ckpts, school_steps = {}, {}
    if cfg.school:
        s = cfg.school
        for r, c in (("attacker", s.attacker_config), ("defender", s.defender_config)):
            res = run_school(r, c, s.steps, cfg.seed, out / "school" / r, s.eval_every, s.val_episodes, s.n_envs,
                             s.rollout_len, log=log)
            ckpts[r], school_steps[r] = res["best"], res["steps"]     # cost = steps run, not the best ckpt's step
    else:
        for r in OTHER:
            if r not in cfg.checkpoints:
                raise ValueError(f"checkpoints.{r} missing")
            ckpts[r], school_steps[r] = cfg.checkpoints[r], _require(cfg.checkpoints[r], r).training_steps
    for r in OTHER:
        _require(ckpts[r], r)
    t = cfg.transfer
    before = report(ckpts["attacker"], ckpts["defender"], cfg.world_config, t.split, t.episodes)
    (out / "stage_d").mkdir(exist_ok=True)
    (out / "stage_d" / "transfer_before_adaptation.json").write_text(json.dumps(before, indent=1))
    result = {"name": cfg.name, "config_hash": cfg.hash(), "code_revision": _code_revision(), "checkpoints": ckpts,
              "school_env_steps": school_steps,
              "stage_d_before": str(out / "stage_d" / "transfer_before_adaptation.json"), "arms": {}, "comparisons": []}
    if cfg.arms:
        a = cfg.adapt
        arm_args = {"pretrained": ("pretrained", a.steps, school_steps[role]),
                    "scratch_same_adapt": ("scratch", a.steps, None),
                    "scratch_equal_total": ("scratch", a.steps + school_steps[role], None)}
        runs = {}
        for arm in cfg.arms:
            init, steps, pre = arm_args[arm]
            runs[arm] = adapt(role=role, attacker_checkpoint=ckpts["attacker"], defender_checkpoint=ckpts["defender"],
                              world_config=cfg.world_config, steps=steps, seed=cfg.seed, out=out / "arms" / arm,
                              league=cfg.league == "on", init=init, eval_every=a.eval_every,
                              val_episodes=a.val_episodes, n_envs=a.n_envs, rollout_len=a.rollout_len,
                              pretrain_steps=pre, log=log)
            result["arms"][arm] = str(out / "arms" / arm / "adapt_manifest.json")
        if "pretrained" in runs:                                    # same report after adaptation, for contrast
            adapted = {**ckpts, role: runs["pretrained"]["checkpoints"]["best_path"]}
            after = report(adapted["attacker"], adapted["defender"], cfg.world_config, t.split, t.episodes)
            (out / "stage_d" / "transfer_after_adaptation.json").write_text(json.dumps(after, indent=1))
            result["stage_d_after"] = str(out / "stage_d" / "transfer_after_adaptation.json")
        for other, rule in (("scratch_same_adapt", "equal_adapt"), ("scratch_equal_total", "equal_total")):
            if "pretrained" in runs and other in runs:
                try:
                    result["comparisons"].append({"pair": ["pretrained", other], **compare([runs["pretrained"], runs[other]], rule)})
                except ComparisonError as e:
                    result["comparisons"].append({"pair": ["pretrained", other], "rule": rule, "rejected": str(e)})
    (out / "comparison_manifest.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.training.experiments")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config end to end (resumable)")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare finished adapt run dirs under a budget rule")
    c.add_argument("--rule", required=True, choices=sorted(RULES))
    c.add_argument("runs", nargs="+", help="adapt run directories")
    a = p.parse_args(argv)
    try:
        if a.cmd == "run":
            res = run_experiment(load_experiment(a.config), a.out)
            rejected = [x for x in res["comparisons"] if "rejected" in x]
            print(json.dumps({"manifest": str(Path(a.out) / "comparison_manifest.json"), "rejected": len(rejected)}))
            return 2 if rejected else 0
        ms = [json.loads((Path(d) / "adapt_manifest.json").read_text()) for d in a.runs]
        print(json.dumps(compare(ms, a.rule), indent=1))
    except (IncompatibleCheckpointError, CheckpointError, WorldConfigError, LineageError, ComparisonError,
            ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
