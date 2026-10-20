"""Versioned JSONL trajectory export. Actor-visible rows and evaluator-only truth live in separate files."""
from __future__ import annotations

import hashlib
import json
import pathlib

TRAJECTORY_VERSION = 1


def export(ep: dict, out_dir: str, episode_id: str, policy_ids: dict[str, str], config_hash: str = "") -> dict:
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    base = {"trajectory_version": TRAJECTORY_VERSION, "episode_id": episode_id, "seed": ep["seed"],
            "scenario_hash": ep["scenario_hash"], "config_hash": config_hash, "catalog_hash": ep["catalog_hash"]}
    with open(d / f"{episode_id}.jsonl", "w") as f:
        for r in ep["rows"]:
            f.write(json.dumps({**base, **r, "checkpoint_id": policy_ids[r["role"]]}) + "\n")
    # privileged artifact: true state / labels, never actor input
    (d / f"{episode_id}.privileged.json").write_text(json.dumps({
        "PRIVILEGED": True, "episode_id": episode_id, "scenario_hash": ep["scenario_hash"], "truth": ep["privileged"]}))
    (d / f"{episode_id}.episode.json").write_text(json.dumps({k: v for k, v in ep.items() if k not in ("rows", "privileged")}))
    return {"rows": len(ep["rows"]), "sha256": hashlib.sha256((d / f"{episode_id}.jsonl").read_bytes()).hexdigest()}


def validate(path: str) -> int:
    """Check required fields; verify no evaluator-only key appears in actor rows."""
    need = {"episode_id", "role", "step", "observation", "legal", "action", "reward_components", "terminated",
            "truncated", "checkpoint_id", "scenario_hash", "seed", "trajectory_version"}
    banned = {"implant", "true_positive", "staged_true", "privileged", "truth"}
    n = 0
    for line in open(path):
        r = json.loads(line)
        assert need <= r.keys(), need - r.keys()
        assert not (banned & set(_keys(r["observation"]))), "evaluator field in actor observation"
        n += 1
    return n


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
