"""School training driver: chunked PPO + held-out validation + validation-gated curriculum + best.pt.

Validation uses split=validation scenarios and unshaped metrics from the engine, never training return.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..envs.curricula import load_config
from ..envs.factories import school_env
from ..policies.checkpoint import load_frozen_agent
from ..runner import summarize
from ..schemas import catalog_hash
from ..envs.encoding import obs_schema_hash
from .curriculum import Curriculum
from .train import TrainConfig, train

FACTORY = "hospital_rl.envs.factories:school_env"


def validate(ckpt: Path, config: str, level: int, n: int, seed: int = 10_000) -> dict:
    """Frozen, deterministic evaluation on validation scenarios. Returns unshaped objective metrics."""
    cfg = load_config(config)
    role = cfg["role"]
    actor = load_frozen_agent(ckpt, role, catalog_hash(), obs_schema_hash(role), seed=seed, deterministic=True)
    rows = []
    for i in range(n):
        env = school_env(config, level, "validation", seed=i)
        obs, _ = env.reset(seed=seed + i)
        actor.reset()
        done = False
        while not done:
            obs, _, te, tr, _ = env.step(actor.act(obs))
            done = te or tr
        rows.append(summarize(env.penv.world))
    exp = float(np.mean([r["weighted_exposure_frac"] for r in rows]))
    feas = float(np.mean([not r["availability_violation"] for r in rows]))
    if role == "attacker":
        success = float(np.mean([r["records_exfiltrated"] > 0 for r in rows]))
        score = exp                                       # higher exposure is better for the attacker
    else:
        success = float(np.mean([(not r["availability_violation"]) and r["weighted_exposure_frac"] <= 0.25
                                 for r in rows]))
        score = -exp - (1.0 - feas)                       # infeasible policies are penalized, not averaged away
    return {"level": level, "n": n, "success": success, "exposure": exp, "feasible_rate": feas,
            "critical_success": float(np.mean([r["critical_success"] for r in rows])),
            "spend": float(np.mean([r["spend"] for r in rows])), "score": score}


def run_school(role: str, config: str, steps: int, seed: int, out: Path, eval_every: int = 4096,
               n_val: int = 20, n_envs: int = 8, rollout_len: int = 64, ppo: dict | None = None, log=print) -> dict:
    cfg = load_config(config)
    assert cfg["role"] == role, f"config role {cfg['role']} != --role {role}"
    out.mkdir(parents=True, exist_ok=True)
    adv = cfg.get("advance_on", {})
    cur = Curriculum([str(i) for i in range(len(cfg["levels"]))], threshold=adv.get("threshold", 0.8), window=2)
    state_f, best_f = out / "school_state.json", out / "best_score.json"
    done_steps, best = 0, (-1, -1e9)
    if state_f.exists():
        st = json.loads(state_f.read_text())
        cur.load_state_dict(st["curriculum"])
        done_steps = st["steps"]
    if best_f.exists():
        b = json.loads(best_f.read_text())
        best = (b["level"], b["score"])
    while done_steps < steps:
        target = min(done_steps + eval_every, steps)
        tcfg = TrainConfig(role=role, seed=seed, total_steps=target, n_envs=n_envs, rollout_len=rollout_len,
                           checkpoint_every=10 ** 6, env_factory=FACTORY, ppo=ppo or {},
                           env_kwargs={"config": config, "level": cur.index, "split": "train"})
        res = train(tcfg, out, resume=(out / "latest.pt").exists(), log=lambda s: None)
        done_steps = res.env_steps
        v = validate(out / "latest.pt", config, cur.index, n_val)
        advanced = cur.report(v["success"])
        key = (v["level"], v["score"])                   # prefer the hardest level reached, then validation score
        if key > best:
            best = key
            shutil.copy(out / "latest.pt", out / "best.pt")
            shutil.copy(out / "latest.pt.meta.json", out / "best.pt.meta.json")
            best_f.write_text(json.dumps({"score": v["score"], "steps": done_steps, "level": v["level"]}))
        row = {"env_steps": done_steps, "train_mean_return": res.history[-1]["mean_return"] if res.history else None,
               "validation": v, "advanced": advanced, "curriculum_level": cur.index}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        state_f.write_text(json.dumps({"curriculum": cur.state_dict(), "steps": done_steps}))
        log(f"steps={done_steps} level={v['level']} val_success={v['success']:.2f} exposure={v['exposure']:.3f} "
            f"crit={v['critical_success']:.3f} score={v['score']:.3f}{' ADVANCE' if advanced else ''}")
    return {"steps": done_steps, "best": str(out / "best.pt"), "best_score": best[1], "level": cur.index}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hospital-rl train")
    p.add_argument("--role", required=True, choices=["attacker", "defender"])
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", help="run dir (default runs/<role>)")
    p.add_argument("--eval-every", type=int, default=4096)
    p.add_argument("--val-episodes", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3, help="PPO lr; 3e-4 collapsed the attacker on seed 7 (see docs)")
    p.add_argument("--ent-coef", type=float, default=0.01)
    a = p.parse_args(argv)
    r = run_school(a.role, a.config, a.steps, a.seed, Path(a.out or f"runs/{a.role}"), a.eval_every,
                   a.val_episodes, a.n_envs, ppo={"lr": a.lr, "ent_coef": a.ent_coef})
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
