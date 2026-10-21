"""Versioned trajectory export/validation.

Per episode three files:
  <id>.jsonl             actor rows: the exact encoded observation + action mask each policy saw when it chose
  <id>.privileged.json   evaluator truth, keyed by the same (role, step) pairs; never actor input
  <id>.manifest.json     sha256 of both files plus every shared hash/version

Rows are written from the real two-role env (``TrajectoryWriter``), so ``observation``/``action_mask`` are exactly what
the policy received. ``validate`` is a whitelist check (exact key sets, shapes, dtypes, masks, hashes, ordering, cross-file
keys); it raises ``TrajectoryError`` rather than asserting.

    python -m hospital_rl.data.trajectory export --attacker-checkpoint A --defender-checkpoint D --out DIR
    python -m hospital_rl.data.trajectory validate DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ..envs.encoding import E_DIM, F_DIM, G_DIM, HIST, MAX_NODES, n_actions, obs_schema_hash
from ..evaluation.splits import split_hash
from ..evaluation.strata import benchmark_hash
from ..schemas import catalog_hash

TRAJECTORY_VERSION = 2
NORMALIZER_VERSION = "running-meanstd-v1"
ROLES = ("attacker", "defender")
ROW_KEYS = {"trajectory_version", "episode_id", "rollout_id", "role", "step", "seed", "policy_id", "checkpoint_id",
            "normalizer_version", "normalizer_hash", "observation_schema_hash", "catalog_hash", "scenario_hash",
            "config_hash", "split", "split_hash", "benchmark_hash", "observation", "action_mask", "action", "reward",
            "reward_components", "terminated", "truncated"}
CONSTANT = ("trajectory_version", "episode_id", "rollout_id", "seed", "scenario_hash", "config_hash", "split",
            "split_hash", "benchmark_hash", "catalog_hash")
POLICY_KEYS = ("policy_id", "checkpoint_id", "normalizer_version", "normalizer_hash")
FORBIDDEN = {"implant", "true_positive", "staged_true", "privileged", "truth", "taken", "staged", "cred_valid"}


class TrajectoryError(ValueError):
    pass


def obs_schema(role: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {"node_features": ("float32", (MAX_NODES, F_DIM[role])), "node_mask": ("int8", (MAX_NODES,)),
            "adjacency": ("int8", (MAX_NODES, MAX_NODES)), "history": ("float32", (HIST, E_DIM)),
            "globals": ("float32", (G_DIM,))}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _identity(actor) -> dict:
    prov = getattr(actor, "provenance", {})
    norm = getattr(getattr(getattr(actor, "agent", None), "agent", None), "normalizer", None)
    out = {"policy_id": actor.id, "checkpoint_id": prov.get("checkpoint_id", "scripted")}
    if norm is None:
        return out | {"normalizer_version": "none", "normalizer_hash": "none"}
    blob = json.dumps(norm.stats_json(), sort_keys=True).encode()
    return out | {"normalizer_version": NORMALIZER_VERSION, "normalizer_hash": hashlib.sha256(blob).hexdigest()[:16]}


class TrajectoryWriter:
    """Callable sink for ``evaluation.transfer.play``: records one full episode from the real env and writes it."""

    def __init__(self, out_dir: str | Path, config_hash: str, split: str):
        self.dir, self.config_hash, self.split, self.rollout = Path(out_dir), config_hash, split, 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, penv, att, dfn, seed: int) -> dict:
        from ..runner import summarize
        obs, _ = penv.reset(seed=seed)
        att.bind(penv, "attacker"); dfn.bind(penv, "defender")
        att.reset(); dfn.reset()
        actors, rows, eid = {"attacker": att, "defender": dfn}, [], f"r{self.rollout:05d}-s{seed}"
        ident = {r: _identity(a) for r, a in actors.items()}
        base = {"trajectory_version": TRAJECTORY_VERSION, "episode_id": eid, "rollout_id": self.rollout, "seed": seed,
                "scenario_hash": penv.spec.hash(), "config_hash": self.config_hash, "split": self.split,
                "split_hash": split_hash(), "benchmark_hash": benchmark_hash(), "catalog_hash": catalog_hash()}
        step = 0
        while penv.agents:
            seen = {r: {k: np.array(v) for k, v in obs[r].items()} for r in ROLES}      # what each policy is shown
            acts = {r: int(actors[r].act(obs[r])) for r in ROLES}
            obs, rew, term, trunc, info = penv.step(acts)
            for r in ROLES:
                o = seen[r]
                rows.append({**base, "role": r, "step": step, **ident[r], "observation_schema_hash": obs_schema_hash(r),
                             "observation": {k: o[k].tolist() for k in obs_schema(r)},
                             "action_mask": o["action_mask"].tolist(), "action": acts[r], "reward": float(rew[r]),
                             "reward_components": info[r]["reward_components"],
                             "terminated": bool(term[r]), "truncated": bool(trunc[r])})
            step += 1
        self._write(eid, base, rows, penv.world.truth, ident)
        self.rollout += 1
        return summarize(penv.world) | {"family": penv.spec.name.split("-")[0]}

    def _write(self, eid, base, rows, truth, ident):
        d = self.dir
        (d / f"{eid}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        (d / f"{eid}.privileged.json").write_text(json.dumps({
            "PRIVILEGED": True, "trajectory_version": TRAJECTORY_VERSION, "episode_id": eid, "seed": base["seed"],
            "scenario_hash": base["scenario_hash"], "keys": [[r["role"], r["step"]] for r in rows], "truth": truth},
            sort_keys=True))
        (d / f"{eid}.manifest.json").write_text(json.dumps({
            "trajectory_version": TRAJECTORY_VERSION, "episode_id": eid, "rows": len(rows), "policies": ident,
            "files": {"actor": {"path": f"{eid}.jsonl", "sha256": _sha(d / f"{eid}.jsonl")},
                      "privileged": {"path": f"{eid}.privileged.json", "sha256": _sha(d / f"{eid}.privileged.json")}},
            **{k: base[k] for k in ("scenario_hash", "config_hash", "split", "split_hash", "benchmark_hash", "catalog_hash")}},
            sort_keys=True, indent=1))


def load_rows(path: str | Path) -> list[dict]:
    """Round-trip: JSON rows back to typed numpy arrays (observation, action_mask)."""
    out = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        sch = obs_schema(r["role"])
        r["observation"] = {k: np.array(v, dtype=sch[k][0]) for k, v in r["observation"].items()}
        r["action_mask"] = np.array(r["action_mask"], dtype=np.int8)
        out.append(r)
    return out


def _need(cond: bool, msg: str):
    if not cond:
        raise TrajectoryError(msg)


def _array(v, dtype: str, shape: tuple, what: str) -> np.ndarray:
    a = np.array(v)
    _need(a.shape == shape, f"{what}: shape {a.shape} != {shape}")
    _need(a.dtype.kind in ("i", "u") if dtype == "int8" else a.dtype.kind in ("f", "i"), f"{what}: bad JSON dtype {a.dtype}")
    _need(bool(np.isfinite(a).all()), f"{what}: non-finite values")
    if dtype == "int8":
        _need(bool(np.isin(a, (0, 1)).all()), f"{what}: not binary")
    else:
        with np.errstate(over="ignore"):
            _need(bool((a.astype(np.float32).astype(np.float64) == a).all()), f"{what}: not float32-representable")
    return a


def _scan_keys(o, where: str):
    if isinstance(o, dict):
        for k, v in o.items():
            _need(str(k) not in FORBIDDEN, f"{where}: evaluator-only key {k!r}")
            _scan_keys(v, where)
    elif isinstance(o, list):
        for v in o:
            _scan_keys(v, where)


def validate(manifest: str | Path) -> dict:
    """Validate one episode (pass its ``.manifest.json``). Returns a summary; raises TrajectoryError."""
    mp = Path(manifest)
    m = json.loads(mp.read_text())
    _need(m.get("trajectory_version") == TRAJECTORY_VERSION, f"trajectory_version {m.get('trajectory_version')!r}")
    paths = {k: mp.parent / m["files"][k]["path"] for k in ("actor", "privileged")}
    for k, p in paths.items():
        _need(p.is_file(), f"{k} file missing: {p.name}")
        _need(_sha(p) == m["files"][k]["sha256"], f"{k} file hash mismatch (tampered or truncated): {p.name}")
    rows = [json.loads(x) for x in paths["actor"].read_text().splitlines()]
    _need(len(rows) == m["rows"] and rows and len(rows) % 2 == 0, "row count mismatch")
    first, per_role = rows[0], {}
    for i, r in enumerate(rows):
        _need(set(r) == ROW_KEYS, f"row {i}: keys differ: extra {sorted(set(r) - ROW_KEYS)} missing {sorted(ROW_KEYS - set(r))}")
        _scan_keys(r, f"row {i}")
        role = r["role"]
        _need(role == ROLES[i % 2] and r["step"] == i // 2, f"row {i}: ordering (role {role}, step {r['step']})")
        _need(all(r[k] == first[k] for k in CONSTANT), f"row {i}: shared header field changed mid-episode")
        for k in ("scenario_hash", "config_hash", "split", "split_hash", "benchmark_hash", "catalog_hash"):
            _need(r[k] == m[k], f"row {i}: {k} differs from manifest")
        _need(r["catalog_hash"] == catalog_hash(), f"row {i}: catalog_hash is not the current catalog")
        _need(r["observation_schema_hash"] == obs_schema_hash(role), f"row {i}: observation schema hash mismatch")
        ident = {k: r[k] for k in POLICY_KEYS}
        _need(per_role.setdefault(role, ident) == ident, f"row {i}: {role} policy/normalizer identity changed")
        _need(ident == m["policies"][role], f"row {i}: policy identity differs from manifest")
        sch = obs_schema(role)
        _need(set(r["observation"]) == set(sch), f"row {i}: observation keys {sorted(r['observation'])}")
        obs = {k: _array(r["observation"][k], *sch[k], f"row {i} {k}") for k in sch}
        n = n_actions(role)
        mask = _array(r["action_mask"], "int8", (n,), f"row {i} action_mask")
        _need(mask[0] == 1, f"row {i}: wait must be legal")
        _need(isinstance(r["action"], int) and 0 <= r["action"] < n and mask[r["action"]] == 1,
              f"row {i}: action {r['action']!r} not legal under its mask")
        nm = obs["node_mask"].astype(bool)
        _need(not obs["adjacency"][~nm].any() and not obs["adjacency"][:, ~nm].any(), f"row {i}: adjacency touches padding")
        _need(not obs["node_features"][~nm].any(), f"row {i}: padding slots carry features")
        _need(isinstance(r["reward"], float) and bool(np.isfinite(r["reward"])), f"row {i}: reward")
        _need(not r["terminated"] and r["truncated"] == (i >= len(rows) - 2), f"row {i}: termination flags")
    pv = json.loads(paths["privileged"].read_text())
    _need(pv.get("PRIVILEGED") is True, "privileged file not marked PRIVILEGED")
    for k in ("episode_id", "seed", "scenario_hash", "trajectory_version"):
        _need(pv[k] == first[k], f"privileged {k} differs from actor rows")
    _need(pv["keys"] == [[r["role"], r["step"]] for r in rows], "privileged keys do not match actor rows")
    _need(len(pv["truth"]) == len(rows) // 2, "privileged truth length != steps")
    _need("observation" not in pv and "action_mask" not in pv, "actor input found in the privileged artifact")
    return {"episode_id": first["episode_id"], "rows": len(rows), "steps": len(rows) // 2}


def validate_dir(d: str | Path) -> list[dict]:
    d = Path(d)
    ms = sorted(d.glob("*.manifest.json"))
    _need(bool(ms), f"no manifests in {d}")
    known = {n for m in ms for n in (m.name.replace(".manifest.json", ".jsonl"),
                                      m.name.replace(".manifest.json", ".privileged.json"))}
    orphans = sorted(p.name for p in d.iterdir() if (p.suffix == ".jsonl" or p.name.endswith(".privileged.json"))
                     and p.name not in known)
    _need(not orphans, f"files without a manifest: {orphans}")
    return [validate(m) for m in ms]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital_rl.data.trajectory")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="evaluate a frozen attacker/defender pair and write trajectories")
    e.add_argument("--attacker-checkpoint", required=True)
    e.add_argument("--defender-checkpoint", required=True)
    e.add_argument("--config", default="configs/hospital40.yaml")
    e.add_argument("--split", choices=["train", "validation"], default="validation")   # test is report-only: no data export
    e.add_argument("--episodes", type=int, default=4)
    e.add_argument("--out", required=True)
    v = sub.add_parser("validate", help="validate every episode in a directory")
    v.add_argument("dir")
    a = p.parse_args(argv)
    try:
        if a.cmd == "validate":
            print(json.dumps(validate_dir(a.dir), indent=1))
            return 0
        from ..evaluation.transfer import evaluate_pair, load_actor
        from ..training.world import load_world
        world = load_world(a.config)
        w = TrajectoryWriter(a.out, world.hash(), a.split)
        r = evaluate_pair(load_actor(a.attacker_checkpoint, "attacker"), load_actor(a.defender_checkpoint, "defender"),
                          world, a.split, a.episodes, sink=w)
        print(json.dumps({"episodes": a.episodes, "validated": validate_dir(a.out), "overall": r["overall"]}, indent=1))
    except Exception as ex:   # noqa: BLE001  bad checkpoints, bad config, invalid trajectories all map to exit 2
        print(f"error: {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
