"""Stage D: both frozen checkpoints (and scripted baselines) in the shared hospital world. `hospital-rl evaluate`.

Exit codes of ``main``: 0 report written, 2 unusable input (missing/incompatible checkpoint or invalid world config).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..envs.encoding import obs_schema_hash
from ..envs.single_role import ScriptedOpponent
from ..evaluation.splits import SPLITS, split_hash
from ..policies.checkpoint import IncompatibleCheckpointError, _code_revision, load_checkpoint, load_frozen_agent
from ..policies.scripted import GreedyDefender, Noop, RandomPolicy, ScriptedAttacker, StaticHardening
from ..runner import summarize
from ..schemas import catalog_hash
from ..training.league import cross_play
from ..training.world import WorldConfig, WorldConfigError, load_world, world_penv

KEYS = ("records_exfiltrated", "weighted_exposure_frac", "critical_success", "noncritical_success", "spend",
        "added_latency")
SCRIPTED = {"ScriptedAttacker": ScriptedAttacker, "GreedyDefender": GreedyDefender, "Noop": Noop,
            "StaticHardening": StaticHardening,
            "RandomPolicy": RandomPolicy}


class CheckpointError(ValueError):
    """Checkpoint file missing or unreadable (incompatibility is IncompatibleCheckpointError)."""


class _Scripted:
    def __init__(self, factory):
        self.opp, self.id, self.provenance = ScriptedOpponent(factory), factory.__name__, {"kind": "scripted"}

    def reset(self):
        self.opp.reset()

    def reseed(self, seed):
        pass

    def bind(self, penv, role):
        self.penv, self.role = penv, role

    def act(self, obs):
        return self.opp(self.penv, self.role)


class _Learned:
    def __init__(self, agent, provenance: dict):
        self.agent, self.id, self.provenance = agent, agent.id, provenance

    def reset(self):
        self.agent.reset()

    def reseed(self, seed):
        self.agent.gen.manual_seed(seed)  # each pair is reproducible regardless of evaluation order

    def bind(self, penv, role):
        pass

    def act(self, obs):
        return self.agent.act(obs)


def scripted_actor(name: str) -> _Scripted:
    return _Scripted(SCRIPTED[name])


def load_actor(path: str, role: str, seed: int = 0, deterministic: bool = False) -> _Learned:
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"{role} checkpoint not found: {p}")
    try:
        meta = load_checkpoint(p).meta
    except Exception as e:  # torch/pydantic raise many types for torn or foreign files
        raise CheckpointError(f"{role} checkpoint {p} unreadable: {type(e).__name__}: {e}") from e
    a = load_frozen_agent(p, role, catalog_hash(), obs_schema_hash(role), seed=seed, deterministic=deterministic)
    assert a.agent.normalizer.frozen and not a.agent.policy.training  # frozen = stats fixed + eval mode
    for prm in a.agent.policy.parameters():
        prm.requires_grad_(False)
    a.id = f"{role}:{meta.checkpoint_id}"
    return _Learned(a, {"kind": "checkpoint", "path": str(p), "checkpoint_id": meta.checkpoint_id,
                        "parent_checkpoint_id": meta.parent_checkpoint_id, "training_steps": meta.training_steps,
                        "config_hash": meta.config_hash, "code_revision": meta.code_revision, "seed": meta.seed})


def play(penv, att, dfn, seed: int, sink=None) -> dict:
    if sink is not None:                                  # data.trajectory.TrajectoryWriter: same episode, recorded
        return sink(penv, att, dfn, seed)
    obs, _ = penv.reset(seed=seed)
    att.bind(penv, "attacker"); dfn.bind(penv, "defender")
    att.reset(); dfn.reset()
    while penv.agents:
        obs, *_ = penv.step({"attacker": att.act(obs["attacker"]), "defender": dfn.act(obs["defender"])})
    return summarize(penv.world) | {"family": penv.spec.name.split("-")[0]}


def eval_seeds(split: str, n: int) -> list[int]:
    if n > len(SPLITS[split]["seeds"]):
        raise ValueError(f"{n} episodes exceed the {split} seed range ({len(SPLITS[split]['seeds'])})")
    return [SPLITS[split]["seeds"].start + i for i in range(n)]


def evaluate_pair(att, dfn, world: WorldConfig, split: str, n: int, sink=None) -> dict:
    penv = world_penv(world, split)
    seeds = eval_seeds(split, n)
    att.reseed(seeds[0]); dfn.reseed(seeds[0])
    runs = [play(penv, att, dfn, s, sink) for s in seeds]
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[r["family"]].append(r)
    agg = lambda rs: {k: _ms([r[k] for r in rs]) for k in KEYS} | {
        "availability_violation_rate": float(np.mean([r["availability_violation"] for r in rs])),
        "contained_rate": float(np.mean([r["contained"] for r in rs])), "n": len(rs)}
    return {"attacker": att.id, "defender": dfn.id, "overall": agg(runs), "by_family": {f: agg(v) for f, v in by_fam.items()},
            "seeds": [seeds[0], seeds[-1]]}


def _ms(x):
    x = np.asarray(x, float)
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"mean": float(x.mean()), "ci95": [float(x.mean() - h), float(x.mean() + h)]}


def _flat(m: dict) -> dict[str, float]:
    o = m["overall"]
    return {**{k: o[k]["mean"] for k in KEYS}, "availability_violation_rate": o["availability_violation_rate"],
            "contained_rate": o["contained_rate"]}


def cross_play_report(attackers: list, defenders: list, world: WorldConfig, split: str, n: int,
                      metrics=("weighted_exposure_frac", "availability_violation_rate")) -> dict:
    """Every attacker x every defender, real actors, one evaluation per pair; labeled matrices + full pair results."""
    by_id = {x.id: x for x in attackers + defenders}
    done: dict[tuple[str, str], dict] = {}

    def ev(a: str, d: str) -> dict:
        done[a, d] = evaluate_pair(by_id[a], by_id[d], world, split, n)
        return _flat(done[a, d])

    aid, did = [x.id for x in attackers], [x.id for x in defenders]
    out = {"attackers": aid, "defenders": did, "matrices": {}}
    for i, metric in enumerate(metrics):
        mat, _ = cross_play(ev if i == 0 else (lambda a, d: _flat(done[a, d])), aid, did, metric)
        out["matrices"][metric] = {"rows": "attacker", "cols": "defender", "values": mat.tolist()}
    out["pairs"] = [done[a, d] for a in aid for d in did]
    return out


def provenance(world: WorldConfig, world_path: str, split: str, n: int, actors: list) -> dict:
    return {"world_config": world.model_dump(), "world_config_path": str(world_path), "world_config_hash": world.hash(),
            "split": split, "split_hash": split_hash(), "catalog_hash": catalog_hash(),
            "observation_schema_hash": {r: obs_schema_hash(r) for r in ("attacker", "defender")},
            "episodes": n, "eval_seeds": [eval_seeds(split, n)[0], eval_seeds(split, n)[-1]],
            "code_revision": _code_revision(), "torch": torch.__version__,
            "actors": {a.id: a.provenance for a in actors}}


def report(att_ckpt: str, def_ckpt: str, world_path: str, split: str, n: int,
           extra_attackers: tuple[str, ...] = (), extra_defenders: tuple[str, ...] = ()) -> dict:
    world = load_world(world_path)
    att, dfn = load_actor(att_ckpt, "attacker"), load_actor(def_ckpt, "defender")
    attackers = [att, *(load_actor(p, "attacker") for p in extra_attackers), scripted_actor("ScriptedAttacker")]
    defenders = [dfn, *(load_actor(p, "defender") for p in extra_defenders), scripted_actor("GreedyDefender"),
                 scripted_actor("Noop"), scripted_actor("StaticHardening")]
    cp = cross_play_report(attackers, defenders, world, split, n)
    return {"stage": "D", "world": f"hospital{world.size}", "frozen": True,
            "note": "one-seed checkpoints; uncalibrated; not a superiority claim",
            "provenance": provenance(world, world_path, split, n, attackers + defenders),
            "split": split, "split_hash": split_hash(), "catalog_hash": catalog_hash(), "episodes": n,
            "cross_play": {k: v for k, v in cp.items() if k != "pairs"}, "matchups": cp["pairs"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl evaluate", description=(
        "Stage D: frozen attacker/defender checkpoints plus scripted baselines, every pairing, in the shared "
        "world. Writes a machine-readable report (cross-play matrices, per-pair metrics, provenance). "
        "Exit 0: report produced; exit 2: missing/unreadable/incompatible checkpoint or invalid world config."))
    p.add_argument("--config", default="configs/hospital40.yaml", help="shared-world YAML (all fields honored)")
    p.add_argument("--attacker-checkpoint", required=True)
    p.add_argument("--defender-checkpoint", required=True)
    p.add_argument("--extra-attacker-checkpoint", action="append", default=[], help="more learned attackers in the matrix")
    p.add_argument("--extra-defender-checkpoint", action="append", default=[], help="more learned defenders in the matrix")
    p.add_argument("--split", choices=list(SPLITS), default="validation")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", help="write the JSON report here")
    a = p.parse_args(argv)
    try:
        r = report(a.attacker_checkpoint, a.defender_checkpoint, a.config, a.split, a.episodes,
                   tuple(a.extra_attacker_checkpoint), tuple(a.extra_defender_checkpoint))
    except (IncompatibleCheckpointError, CheckpointError, WorldConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{r['world']} split={r['split']} n={r['episodes']} (frozen policies; {r['note']})")
    print(f"{'attacker':34}{'defender':34}{'exposure':>9}{'crit_ok':>9}{'spend':>8}  availability_violations")
    for m in r["matchups"]:
        o = m["overall"]
        print(f"{m['attacker']:34}{m['defender']:34}{o['weighted_exposure_frac']['mean']:>9.3f}"
              f"{o['critical_success']['mean']:>9.3f}{o['spend']['mean']:>8.1f}  {o['availability_violation_rate']:.0%}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
