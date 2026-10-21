"""Stage D: both frozen checkpoints in the shared hospital world. `hospital-rl evaluate`."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..envs.encoding import obs_schema_hash
from ..envs.factories import hospital_penv
from ..envs.single_role import ScriptedOpponent
from ..evaluation.splits import SPLITS, split_hash
from ..policies.checkpoint import IncompatibleCheckpointError, load_frozen_agent
from ..policies.scripted import GreedyDefender, Noop, ScriptedAttacker
from ..runner import summarize
from ..schemas import catalog_hash

KEYS = ("records_exfiltrated", "weighted_exposure_frac", "critical_success", "noncritical_success", "spend",
        "added_latency")


class _Scripted:
    def __init__(self, factory):
        self.opp, self.id = ScriptedOpponent(factory), factory.__name__

    def reset(self):
        self.opp.reset()

    def bind(self, penv, role):
        self.penv, self.role = penv, role

    def act(self, obs):
        return self.opp(self.penv, self.role)


class _Learned:
    def __init__(self, agent):
        self.agent, self.id = agent, agent.id

    def reset(self):
        self.agent.reset()

    def bind(self, penv, role):
        pass

    def act(self, obs):
        return self.agent.act(obs)


def load_actor(path: str, role: str, seed: int = 0, deterministic: bool = False) -> _Learned:
    a = load_frozen_agent(path, role, catalog_hash(), obs_schema_hash(role), seed=seed, deterministic=deterministic)
    a.id = f"{role}:{Path(path).name}"
    return _Learned(a)


def play(penv, att, dfn, seed: int) -> dict:
    obs, _ = penv.reset(seed=seed)
    att.bind(penv, "attacker"); dfn.bind(penv, "defender")
    att.reset(); dfn.reset()
    while penv.agents:
        obs, *_ = penv.step({"attacker": att.act(obs["attacker"]), "defender": dfn.act(obs["defender"])})
    return summarize(penv.world) | {"family": penv.spec.name.split("-")[0]}


def evaluate_pair(att, dfn, size: int, split: str, n: int, horizon: int = 60, difficulty: float = 0.7) -> dict:
    penv = hospital_penv(size, split, difficulty, horizon)
    seeds = [SPLITS[split]["seeds"].start + i for i in range(n)]
    runs = [play(penv, att, dfn, s) for s in seeds]
    by_fam = defaultdict(list)
    for r in runs:
        by_fam[r["family"]].append(r)
    agg = lambda rs: {k: _ms([r[k] for r in rs]) for k in KEYS} | {
        "availability_violation_rate": float(np.mean([r["availability_violation"] for r in rs])),
        "contained_rate": float(np.mean([r["contained"] for r in rs])), "n": len(rs)}
    return {"attacker": att.id, "defender": dfn.id, "overall": agg(runs), "by_family": {f: agg(v) for f, v in by_fam.items()}}


def _ms(x):
    x = np.asarray(x, float)
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"mean": float(x.mean()), "ci95": [float(x.mean() - h), float(x.mean() + h)]}


def report(att_ckpt: str, def_ckpt: str, size: int, split: str, n: int) -> dict:
    att, dfn = load_actor(att_ckpt, "attacker"), load_actor(def_ckpt, "defender")
    pairs = [(att, dfn), (_Scripted(ScriptedAttacker), dfn), (att, _Scripted(GreedyDefender)),
             (att, _Scripted(Noop)), (_Scripted(ScriptedAttacker), _Scripted(GreedyDefender)),
             (_Scripted(ScriptedAttacker), _Scripted(Noop))]
    return {"world": f"hospital{size}", "split": split, "split_hash": split_hash(), "catalog_hash": catalog_hash(),
            "episodes": n, "frozen": True, "note": "one-seed checkpoints; uncalibrated; not a superiority claim",
            "matchups": [evaluate_pair(a, d, size, split, n) for a, d in pairs]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl evaluate")
    p.add_argument("--config", default="configs/hospital40.yaml")
    p.add_argument("--attacker-checkpoint", required=True)
    p.add_argument("--defender-checkpoint", required=True)
    p.add_argument("--split", choices=list(SPLITS), default="validation")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out")
    a = p.parse_args(argv)
    import yaml
    size = yaml.safe_load(open(a.config)).get("size", 40)
    try:
        r = report(a.attacker_checkpoint, a.defender_checkpoint, size, a.split, a.episodes)
    except IncompatibleCheckpointError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"{r['world']} split={r['split']} n={r['episodes']} (frozen policies; {r['note']})")
    print(f"{'attacker':26}{'defender':26}{'exposure':>9}{'crit_ok':>9}{'spend':>8}  availability_violations")
    for m in r["matchups"]:
        o = m["overall"]
        print(f"{m['attacker']:26}{m['defender']:26}{o['weighted_exposure_frac']['mean']:>9.3f}"
              f"{o['critical_success']['mean']:>9.3f}{o['spend']['mean']:>8.1f}  {o['availability_violation_rate']:.0%}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
