"""Stage E: shared-world adaptation of exactly one role against a frozen opponent (optionally a bounded league),
plus alternating adaptation at explicit round boundaries. `hospital-rl adapt`.

Exit codes of ``main``: 0 done, 2 unusable input (missing/incompatible checkpoint, bad world config, lineage conflict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

from ..envs.encoding import obs_schema_hash
from ..envs.single_role import SingleRoleEnv
from ..evaluation.splits import split_hash
from ..evaluation.transfer import (SCRIPTED, CheckpointError, _Learned, _Scripted, eval_seeds, load_actor, play,
                                   scripted_actor)
from ..policies.checkpoint import IncompatibleCheckpointError, _code_revision, load_for_role
from ..schemas import catalog_hash
from .league import Entry, League
from .lineage import LineageError, claim_run
from .schools import score_rows
from .train import TrainConfig, train
from .world import WorldConfig, WorldConfigError, load_world, world_penv

FACTORY = "hospital_rl.training.adapt:adapt_env"
OTHER = {"attacker": "defender", "defender": "attacker"}
# scripted opponents by the OPPONENT's role: sampled for updates / held out for evaluation only
TRAIN_SCRIPTED = {"attacker": "ScriptedAttacker", "defender": "GreedyDefender"}
EVAL_SCRIPTED = {"attacker": ["RandomPolicy"], "defender": ["Noop", "StaticHardening"]}


# ---------------------------------------------------------------- env with a league-sampled frozen opponent
class LeagueOpponent:
    """Samples one frozen entry from the league TRAIN pool at each episode reset (never the eval pool)."""

    def __init__(self, league_path: str, role: str, seed: int = 0):
        self.league, self.role, self.rng = League.load(league_path), role, random.Random(seed)
        self.cache: dict[str, _Learned | _Scripted] = {}
        self.cur, self.counts = None, {}

    def _actor(self, e: Entry):
        if e.id not in self.cache:
            self.cache[e.id] = (scripted_actor(e.id.split(":", 1)[1]) if e.kind == "scripted"
                                else load_actor(e.path, self.role))
        return self.cache[e.id]

    def reset(self):
        e = self.league.sample(self.rng)
        self.cur, self.counts[e.id] = self._actor(e), self.counts.get(e.id, 0) + 1
        self.cur.reseed(self.rng.randrange(2 ** 31))
        self.cur.reset()

    def __call__(self, penv, role):
        self.cur.bind(penv, role)
        return self.cur.act(penv.last_obs[role])


class AdaptEnv(SingleRoleEnv):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.opp.rng.seed(seed)       # opponent draws are a function of the rollout seed -> resume-deterministic
        return super().reset(seed=seed, options=options)


def adapt_env(role: str, world: dict, league: str, split: str = "train", seed: int = 0) -> AdaptEnv:
    return AdaptEnv(world_penv(WorldConfig(**world), split), role, LeagueOpponent(league, OTHER[role], seed))


# ---------------------------------------------------------------- helpers
def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _require(path: str, role: str):
    """Existing, readable and compatible checkpoint for `role`, else a clear error. Never searches runs/."""
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"{role} checkpoint not found: {p}")
    try:
        return load_for_role(p, role, catalog_hash(), obs_schema_hash(role)).meta
    except IncompatibleCheckpointError:
        raise
    except Exception as e:
        raise CheckpointError(f"{role} checkpoint {p} unreadable: {type(e).__name__}: {e}") from e


def frozen_digest(actor: _Learned) -> str:
    """Hash of everything that would change if the opponent were (wrongly) trained: weights + normalizer."""
    h = hashlib.sha256()
    for k, v in sorted(actor.agent.agent.policy.state_dict().items()):
        h.update(k.encode() + v.numpy().tobytes())
    h.update(json.dumps(actor.agent.agent.normalizer.stats_json(), sort_keys=True).encode())
    return h.hexdigest()[:16]


def build_league(path: Path, opp_role: str, frozen_opponent: str, enabled: bool, history: tuple[str, ...] = (),
                 eval_ckpts: tuple[str, ...] = (), max_checkpoints: int = 8) -> League:
    """Train pool: [scripted + historical + ] the frozen opponent as newest checkpoint. Eval pool: held-out only."""
    lg = League(max_checkpoints)
    if enabled:
        lg.add_train(Entry(f"scripted:{TRAIN_SCRIPTED[opp_role]}", "scripted"))
        for p in history:
            lg.add_train(Entry(_require(p, opp_role).checkpoint_id, "checkpoint", str(p)))
        for n in EVAL_SCRIPTED[opp_role]:
            lg.add_eval(Entry(f"scripted:{n}", "scripted"))
        for p in eval_ckpts:
            lg.add_eval(Entry(_require(p, opp_role).checkpoint_id, "checkpoint", str(p)))
    lg.add_train(Entry(_require(frozen_opponent, opp_role).checkpoint_id, "checkpoint", str(frozen_opponent)))
    lg.save(path)
    return lg


def validate_shared(ckpt: Path, role: str, opponent: str, world: WorldConfig, n: int) -> dict:
    """Validation split only, frozen deterministic actors, unshaped metrics. Selection never looks elsewhere."""
    me, opp = load_actor(str(ckpt), role, deterministic=True), load_actor(opponent, OTHER[role], deterministic=True)
    att, dfn = (me, opp) if role == "attacker" else (opp, me)
    penv = world_penv(world, "validation")
    rows = [play(penv, att, dfn, s) for s in eval_seeds("validation", n)]
    return score_rows(role, rows)


def _scripted_entry_actor(e: Entry, role: str):
    return scripted_actor(e.id.split(":", 1)[1]) if e.kind == "scripted" else load_actor(e.path, role, deterministic=True)


def eval_pool_report(ckpt: Path, role: str, league: League, world: WorldConfig, n: int) -> list[dict]:
    """Held-out opponents: reported, never sampled for updates and never used for selection."""
    out = []
    for e in league.eval_pool:
        opp = _scripted_entry_actor(e, OTHER[role])
        me = load_actor(str(ckpt), role, deterministic=True)
        att, dfn = (me, opp) if role == "attacker" else (opp, me)
        penv = world_penv(world, "validation")
        out.append({"opponent": e.id, **score_rows(role, [play(penv, att, dfn, s) for s in eval_seeds("validation", n)])})
    return out


def _train_to(role: str, target: int, out: Path, *, init: str | None, league: Path, world: WorldConfig,
              opponent: str, seed: int, eval_every: int, n_val: int, n_envs: int, rollout_len: int, ppo: dict,
              log) -> dict:
    """Chunked PPO of `role` in `out` until its own step counter reaches `target`; validation-only best.pt."""
    out.mkdir(parents=True, exist_ok=True)
    best_f, metrics_f = out / "best_score.json", out / "metrics.jsonl"
    best = json.loads(best_f.read_text())["score"] if best_f.exists() else -1e9
    while (cur := _steps(out)) < target or not (out / "best.pt").exists():
        cfg = TrainConfig(role=role, seed=seed, total_steps=min(cur + eval_every, target), n_envs=n_envs,
                          rollout_len=rollout_len, checkpoint_every=10 ** 6, env_factory=FACTORY, ppo=ppo,
                          init_checkpoint=init, env_kwargs={"role": role, "world": world.model_dump(),
                                                            "league": str(league), "split": "train", "seed": seed})
        res = train(cfg, out, resume=(out / "latest.pt").exists(), log=lambda s: None)
        val = validate_shared(out / "latest.pt", role, opponent, world, n_val)
        if val["score"] > best:
            best = val["score"]
            shutil.copy(out / "latest.pt", out / "best.pt")
            shutil.copy(out / "latest.pt.meta.json", out / "best.pt.meta.json")
            best_f.write_text(json.dumps({"score": best, "steps": res.env_steps}))
        with open(metrics_f, "a") as f:
            f.write(json.dumps({"env_steps": res.env_steps, "validation": val,
                                "train_mean_return": res.history[-1]["mean_return"] if res.history else None}) + "\n")
        log(f"{role} steps={res.env_steps} val_score={val['score']:.3f} exposure={val['exposure']:.3f} "
            f"feasible={val['feasible_rate']:.2f}")
    last = json.loads(metrics_f.read_text().splitlines()[-1])
    return {"steps": _steps(out), "best_score": best, "last_validation": last["validation"]}


def _steps(out: Path) -> int:
    from ..policies.checkpoint import load_checkpoint
    return load_checkpoint(out / "latest.pt").meta.training_steps if (out / "latest.pt").exists() else 0


def _ppo(lr: float, ent_coef: float) -> dict:
    return {"lr": lr, "ent_coef": ent_coef}


# ---------------------------------------------------------------- one-side adaptation
def adapt(*, role: str, attacker_checkpoint: str, defender_checkpoint: str, world_config: str, steps: int, seed: int,
          out: str | Path, league: bool, init: str = "pretrained", league_checkpoints: tuple[str, ...] = (),
          eval_checkpoints: tuple[str, ...] = (), eval_every: int = 2048, val_episodes: int = 10, n_envs: int = 4,
          rollout_len: int = 64, lr: float = 1e-3, ent_coef: float = 0.01, log=print) -> dict:
    """Adapt `role` for `steps` env steps while the other role is frozen. Weights start from the role's checkpoint
    (init=pretrained) or from scratch (init=scratch, same budget, same frozen opponent)."""
    if role not in OTHER:
        raise ValueError(f"trainable role must be attacker or defender, got {role!r}")
    if init not in ("pretrained", "scratch"):
        raise ValueError(f"init must be pretrained or scratch, got {init!r}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    ckpts = {"attacker": attacker_checkpoint, "defender": defender_checkpoint}
    metas = {r: _require(c, r) for r, c in ckpts.items()}      # both are compat-checked even for scratch
    world, out = load_world(world_config), Path(out)
    own, opp = metas[role], metas[OTHER[role]]
    identity = {"kind": "adapt", "role": role, "init": init, "source_checkpoint_id": own.checkpoint_id,
                "opponent_checkpoint_id": opp.checkpoint_id, "world_hash": world.hash(), "seed": seed,
                "league": league, "league_checkpoints": sorted(_require(p, OTHER[role]).checkpoint_id for p in league_checkpoints),
                "eval_checkpoints": sorted(_require(p, OTHER[role]).checkpoint_id for p in eval_checkpoints),
                "n_envs": n_envs, "rollout_len": rollout_len}
    resumed = claim_run(out, identity)
    lg_path = out / "league.json"
    lg = League.load(lg_path) if resumed and lg_path.exists() else build_league(
        lg_path, OTHER[role], ckpts[OTHER[role]], league, league_checkpoints, eval_checkpoints)
    opp_digest = _sha(ckpts[OTHER[role]])
    r = _train_to(role, steps, out, init=ckpts[role] if init == "pretrained" else None, league=lg_path, world=world,
                  opponent=ckpts[OTHER[role]], seed=seed, eval_every=eval_every, n_val=val_episodes, n_envs=n_envs,
                  rollout_len=rollout_len, ppo=_ppo(lr, ent_coef), log=log)
    best = load_for_role(out / "best.pt", role, catalog_hash(), obs_schema_hash(role)).meta
    lg.provenance = {"selection_split": "validation", "selection_metric": "score_rows.score vs frozen opponent",
                     "selected_checkpoint_id": best.checkpoint_id, "selected_score": r["best_score"],
                     "eval_pool_used_for_selection": False, "eval_pool_sampled_for_updates": False}
    lg.save(lg_path)
    pool = eval_pool_report(out / "best.pt", role, lg, world, val_episodes) if league else []
    (out / "eval_pool.json").write_text(json.dumps(pool, indent=1))
    pre = own.training_steps if init == "pretrained" else 0
    manifest = {
        "kind": "adapt", "identity": identity, "resumed": resumed, "code_revision": _code_revision(),
        "world_config": world.model_dump(), "world_config_path": str(world_config),
        "hashes": {"world": world.hash(), "split": split_hash(), "catalog": catalog_hash()},
        "checkpoints": {"source": own.checkpoint_id, "opponent": opp.checkpoint_id, "best": best.checkpoint_id,
                        "best_path": str(out / "best.pt"), "latest_path": str(out / "latest.pt")},
        "budget": {"requested_adapt_steps": steps, "adapt_env_steps": r["steps"], "pretrain_env_steps": pre,
                   "total_env_steps": pre + r["steps"]},
        "validation": r["last_validation"], "best_validation_score": r["best_score"],
        "opponent_frozen": {"checkpoint_sha_before": opp_digest, "checkpoint_sha_after": _sha(ckpts[OTHER[role]]),
                            "unchanged": opp_digest == _sha(ckpts[OTHER[role]])},
        "league": {"enabled": league, "manifest": str(lg_path), "eval_pool_report": str(out / "eval_pool.json")},
        "comparability": {"trainable_role": role, "world_hash": world.hash(), "split_hash": split_hash(),
                          "catalog_hash": catalog_hash(), "opponent_checkpoint_id": opp.checkpoint_id,
                          "league": league, "val_episodes": val_episodes, "seed": seed},
        "note": "smoke/short budgets are not evidence of policy superiority"}
    (out / "adapt_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------- alternating adaptation
def alternate(*, first_role: str, attacker_checkpoint: str, defender_checkpoint: str, world_config: str,
              rounds: int, steps_per_phase: int, seed: int, out: str | Path, league: bool,
              init: str = "pretrained", eval_every: int = 2048, val_episodes: int = 10, n_envs: int = 4,
              rollout_len: int = 64, lr: float = 1e-3, ent_coef: float = 0.01, log=print) -> dict:
    """Round r: first_role trains `steps_per_phase` vs a frozen snapshot of the other, then the other role trains
    vs a frozen snapshot of the first. Each phase has its own run dir and PPO counters, so rollouts are never
    shared between phases (PPO also rejects stale/duplicate rollout ids). State is committed per finished phase."""
    ckpts = {"attacker": attacker_checkpoint, "defender": defender_checkpoint}
    metas = {r: _require(c, r) for r, c in ckpts.items()}
    world, out = load_world(world_config), Path(out)
    identity = {"kind": "alternate", "first_role": first_role, "init": init, "world_hash": world.hash(), "seed": seed,
                "source_checkpoint_ids": {r: m.checkpoint_id for r, m in metas.items()}, "league": league,
                "steps_per_phase": steps_per_phase, "n_envs": n_envs, "rollout_len": rollout_len}
    claim_run(out, identity)
    st_f = out / "alt_state.json"
    state = json.loads(st_f.read_text()) if st_f.exists() else {"phases": []}
    order = [first_role, OTHER[first_role]]
    latest = dict(ckpts)  # role -> frozen snapshot to use as that role's opponent
    for ph in state["phases"]:
        latest[ph["role"]] = ph["snapshot"]
    seen: set[str] = set()
    for rnd in range(rounds):
        for role in order:
            tag = f"r{rnd}_{role}"
            if any(p["tag"] == tag for p in state["phases"]):
                continue
            opp_role, run = OTHER[role], out / role
            snap_in = out / "snapshots" / f"in_{tag}.pt"             # frozen opponent for this phase (reused on resume)
            if not snap_in.exists():
                snap_in.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(latest[opp_role], snap_in)
                shutil.copy(str(latest[opp_role]) + ".meta.json", str(snap_in) + ".meta.json")
            lg_path = out / f"league_{role}.json"
            hist = tuple(p["opponent_snapshot"] for p in state["phases"] if p["role"] == role)
            lg = build_league(lg_path, opp_role, str(snap_in), league, history=hist)
            r = _train_to(role, (rnd + 1) * steps_per_phase, run,
                          init=ckpts[role] if init == "pretrained" else None, league=lg_path, world=world,
                          opponent=str(snap_in), seed=seed, eval_every=eval_every, n_val=val_episodes, n_envs=n_envs,
                          rollout_len=rollout_len, ppo=_ppo(lr, ent_coef), log=log)
            snap_out = out / "snapshots" / f"out_{tag}.pt"           # checkpoint this side at the round boundary
            shutil.copy(run / "latest.pt", snap_out)
            shutil.copy(run / "latest.pt.meta.json", str(snap_out) + ".meta.json")
            state["phases"].append({"tag": tag, "round": rnd, "role": role, "opponent_snapshot": str(snap_in),
                                    "snapshot": str(snap_out), "env_steps": r["steps"],
                                    "validation": r["last_validation"]})
            st_f.write_text(json.dumps(state, indent=1))
            latest[role] = str(snap_out)
    manifest = {"kind": "alternate", "identity": identity, "rounds": rounds, "code_revision": _code_revision(),
                "phases": state["phases"], "hashes": {"world": world.hash(), "split": split_hash(),
                                                      "catalog": catalog_hash()},
                "final": {r: str(latest[r]) for r in order},
                "budget": {"adapt_env_steps": {r: rounds * steps_per_phase for r in order}}}
    (out / "alt_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl adapt", description=(
        "Stage E: adapt exactly one role in the shared hospital world while the other role is frozen. Nothing is "
        "selected from runs/; every checkpoint is explicit. Exit 0: finished; exit 2: missing/incompatible checkpoint, "
        "invalid world config, or the --out dir belongs to a different run."))
    p.add_argument("--trainable-role", required=True, choices=list(OTHER), help="role that learns (first role if --rounds)")
    p.add_argument("--attacker-checkpoint", required=True)
    p.add_argument("--defender-checkpoint", required=True)
    p.add_argument("--world-config", required=True, help="shared-world YAML, e.g. configs/hospital40.yaml")
    p.add_argument("--steps", type=int, required=True, help="env steps for the trainable role (per phase with --rounds)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--league", required=True, choices=["on", "off"],
                   help="on: opponent sampled from scripted + historical + frozen pool; off: the frozen checkpoint only")
    p.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained",
                   help="scratch: same frozen opponent and budget, trainable role starts from random weights")
    p.add_argument("--league-checkpoint", action="append", default=[], help="historical opponent checkpoint (train pool)")
    p.add_argument("--eval-checkpoint", action="append", default=[], help="held-out opponent checkpoint (eval pool)")
    p.add_argument("--rounds", type=int, help="alternating adaptation: number of rounds (one phase per role per round)")
    p.add_argument("--eval-every", type=int, default=2048)
    p.add_argument("--val-episodes", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    a = p.parse_args(argv)
    common = dict(attacker_checkpoint=a.attacker_checkpoint, defender_checkpoint=a.defender_checkpoint,
                  world_config=a.world_config, seed=a.seed, out=a.out, league=a.league == "on", init=a.init,
                  eval_every=a.eval_every, val_episodes=a.val_episodes, n_envs=a.n_envs, lr=a.lr, ent_coef=a.ent_coef)
    try:
        if a.rounds:
            m = alternate(first_role=a.trainable_role, rounds=a.rounds, steps_per_phase=a.steps, **common)
        else:
            m = adapt(role=a.trainable_role, steps=a.steps, league_checkpoints=tuple(a.league_checkpoint),
                      eval_checkpoints=tuple(a.eval_checkpoint), **common)
    except (IncompatibleCheckpointError, CheckpointError, WorldConfigError, LineageError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"kind": m["kind"], "budget": m["budget"], "out": a.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
