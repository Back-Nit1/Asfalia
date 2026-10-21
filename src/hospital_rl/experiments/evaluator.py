"""Real frozen-policy evaluator used by the local experiment controller."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from ..evaluation.splits import SPLITS, split_hash
from ..evaluation.transfer import KEYS, load_actor, play
from ..policies.checkpoint import atomic_write
from ..runner import summarize
from ..training.world import load_world, world_penv


def _summary(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": len(values)}


def _record_episode(penv, attacker, defender, seed: int) -> dict:
    """Run one frozen episode and retain the evaluator-side replay contract."""
    attacker.reseed(seed)
    defender.reseed(seed)
    obs, _ = penv.reset(seed=seed)
    attacker.bind(penv, "attacker")
    defender.bind(penv, "defender")
    attacker.reset()
    defender.reset()
    total = {"attacker": 0.0, "defender": 0.0}
    steps = []
    while penv.agents:
        encoded = {"attacker": attacker.act(obs["attacker"]), "defender": defender.act(obs["defender"])}
        actions = {role: penv.codec.decode(role, int(action)) for role, action in encoded.items()}
        obs, *_ = penv.step(encoded)
        truth = penv.world.truth[-1]
        for role in total:
            total[role] += truth["reward"][role]
        steps.append({"tick": truth["tick"], "attacker": actions["attacker"].__dict__,
                      "defender": actions["defender"].__dict__, "events": truth["events"],
                      "reward": truth["reward"], "components": truth["components"]})
    spec = penv.spec
    return {"scenario": spec.name, "spec": spec.model_dump(mode="json"), "scenario_hash": spec.hash(), "seed": seed,
            "tie_policy": spec.tie_policy, "return": total, "metrics": summarize(penv.world), "steps": steps,
            "topology": {"nodes": [{"id": n.id, "asset_type": n.asset_type, "zone": n.zone} for n in spec.nodes],
                         "edges": [{"src": e.src, "dst": e.dst, "kind": e.kind} for e in spec.edges]}}


class FrozenEvaluator:
    """Evaluate the submitted learned pair only; no checkpoint selection or policy updates occur."""

    name = "frozen-local"

    def __init__(self, world_config: str = "configs/hospital40.yaml", split: str = "validation"):
        if split not in ("validation", "test"):
            raise ValueError("controller evaluation split must be validation or test")
        self.world_config, self.world, self.split = world_config, load_world(world_config), split

    def run(self, spec: dict, checkpoint_paths: dict[str, str], out_dir: str) -> dict:
        if spec.get("suite") == "debug_fork":
            raise ValueError("real debug-fork evaluation is not implemented; submit a frozen two-role evaluation")
        n, first = int(spec["episodes"]), int(spec["seed"])
        allowed = SPLITS[self.split]["seeds"]
        seeds = list(range(first, first + n))
        if any(seed not in allowed for seed in seeds):
            raise ValueError(f"evaluation seeds {first}..{first + n - 1} are outside the {self.split} stratum")
        attacker = load_actor(checkpoint_paths["attacker"], "attacker", deterministic=True)
        defender = load_actor(checkpoint_paths["defender"], "defender", deterministic=True)
        penv = world_penv(self.world, self.split)
        attacker.reseed(first)
        defender.reseed(first)
        runs = [play(penv, attacker, defender, seed) for seed in seeds]
        values = {metric: [float(row[metric]) for row in runs] for metric in KEYS}
        values.update(availability_violation_rate=[float(row["availability_violation"]) for row in runs],
                      contained_rate=[float(row["contained"]) for row in runs])
        summaries = {metric: _summary(rows) for metric, rows in values.items()}
        pairs = [{"attacker": spec["attacker_checkpoint_id"], "defender": spec["defender_checkpoint_id"],
                  "metric": metric, **summary} for metric, summary in summaries.items()]

        export_root = Path(out_dir).parent
        export_root.mkdir(parents=True, exist_ok=True)
        identity = hashlib.sha256(f"{Path(out_dir).name}:{first}:{spec['attacker_sha256']}:{spec['defender_sha256']}".encode()).hexdigest()[:12]
        episode_id = f"eval_{identity}"
        recorded = _record_episode(world_penv(self.world, self.split), attacker, defender, first)
        atomic_write(export_root / f"{episode_id}.json", json.dumps(recorded).encode())
        result = {"pairs": pairs, "episode_ids": [episode_id], "frozen": True, "split": self.split,
                  "split_hash": split_hash(), "world_config": self.world_config, "world_config_hash": self.world.hash(),
                  "seeds": [seeds[0], seeds[-1]], "note": "frozen learned pair; descriptive, uncalibrated"}
        report_dir = Path(out_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(report_dir / "evaluation.json", json.dumps(result, indent=1).encode())
        return result
