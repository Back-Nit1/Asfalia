"""Rollout scaling: one env vs vectorized envs in one worker vs (opt-in) several Modal workers.

Equal work in every arm: same total env steps, same seed, same freshly initialised policy/normalizer (policy_version 0),
same real shared-world defender environment. Reports per arm: steps/s over the rollout loop, startup seconds (env object + policy
construction), CPU utilisation (process CPU time / wall; ~1.0 = one core saturated), and the resource assumptions
(logical cores, torch threads). The first env reset (the real scenario build cost) lands in the rollout loop, not in startup. Rollout collection only: no PPO update, so it isolates environment + policy-forward cost.

  PYTHONPATH=src python benchmarks/rollout_scaling/run.py --steps 8192 --n-envs 1,4,8 --out benchmarks/rollout_scaling/results_hospital_macos_arm64.json
  ASFALIA_LIVE_MODAL=1 ... --modal-workers 4      # opt-in: real compute; K train jobs of steps/K each, wall incl. startup/transfer

Distributed rollout workers are NOT implemented; the Modal arm only measures K independent bounded jobs so a decision
can be made from numbers. Results are whatever this machine produced; nothing here is a cloud-cost estimate.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from hospital_rl.envs.factories import hospital_single
from hospital_rl.policies.ppo import GraphPolicy, Normalizer, SyncVecEnv, collect_rollout

ENV_KW = dict(role="defender", size=12, split="train", difficulty=0.5, horizon=40, opponent="scripted")


def arm(n_envs: int, steps: int, rollout_len: int, seed: int) -> dict:
    t_start = time.perf_counter()
    venv = SyncVecEnv([lambda: hospital_single(**ENV_KW)] * n_envs)
    sp = venv.observation_space
    torch.manual_seed(seed), np.random.seed(seed)
    n_types = int(venv.action_space.n - 1) // sp["node_mask"].shape[0]
    policy = GraphPolicy(int(sp["node_features"].shape[1]), int(sp["history"].shape[1]), int(sp["globals"].shape[0]), n_types, 64)
    norm = Normalizer(policy.config["node_dim"], policy.config["hist_dim"], policy.config["glob_dim"])
    startup = time.perf_counter() - t_start
    per_round = n_envs * rollout_len
    rounds, done = max(1, steps // per_round), 0
    cpu0, t0 = time.process_time(), time.perf_counter()
    for r in range(rounds):
        collect_rollout(venv, policy, norm, rollout_len, policy_version=0, rollout_id=r + 1, seed=seed, gamma=0.99)
        done += per_round
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    return {"arm": f"vec{n_envs}" if n_envs > 1 else "single", "n_envs": n_envs, "env_steps": done, "rollout_len": rollout_len,
            "startup_s": round(startup, 3), "rollout_wall_s": round(wall, 3), "steps_per_s": round(done / wall, 1),
            "cpu_utilisation": round(cpu / wall, 2)}


def modal_arm(workers: int, steps: int, seed: int) -> dict:
    if os.environ.get("ASFALIA_LIVE_MODAL") != "1":
        return {"arm": f"modal{workers}", "status": "unexecuted: set ASFALIA_LIVE_MODAL=1 (spends real compute)"}
    import tempfile
    import uuid

    from hospital_rl.infra.backend import JobSpec
    from hospital_rl.infra.modal_app import ModalBackend, ModalConfig, RealModalGateway
    each = steps // workers
    cfg = ModalConfig(app_name="hospital-rl-bench", volume_name="hospital-rl-bench-runs", registry_dir=tempfile.mkdtemp(),
                      caps=dict(cpu=2, memory_mb=2048, timeout_s=300, grace_s=30, max_containers=workers, max_steps=each))
    gw = RealModalGateway(cfg)
    be, run = ModalBackend(cfg, gw), f"bench-{uuid.uuid4().hex[:6]}"
    tc = dict(role="defender", total_steps=each, n_envs=8, rollout_len=64, checkpoint_every=10 ** 6, seed=seed,
              env_factory="hospital_rl.envs.factories:hospital_single", env_kwargs=ENV_KW)
    t0 = time.perf_counter()
    try:
        ids = [be.submit(JobSpec(run_id=run, attempt_id=f"w{i}", train_config=tc, timeout_s=300, cpu=2, memory_mb=2048)) for i in range(workers)]
        sts = [be.wait(j, 400, poll_s=2) for j in ids]
        wall = time.perf_counter() - t0
    finally:
        gw.close()
    return {"arm": f"modal{workers}", "workers": workers, "env_steps": sum(s.result.get("env_steps", 0) for s in sts),
            "states": [s.state.value for s in sts], "wall_incl_startup_transfer_s": round(wall, 1), "call_ids": [be._rec(j)["remote_id"] for j in ids],
            "note": "includes training updates, unlike the local arms; compare scaling shape, not absolute steps/s"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--n-envs", default="1,4,8")
    p.add_argument("--rollout-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--modal-workers", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    res = {"host": {"platform": platform.platform(), "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(), "torch": torch.__version__},
           "assumptions": "same total steps/seed/policy init per arm; collection only; SyncVecEnv is single-process",
           "arms": [arm(int(n), a.steps, a.rollout_len, a.seed) for n in a.n_envs.split(",")]}
    if a.modal_workers:
        res["arms"].append(modal_arm(a.modal_workers, a.steps, a.seed))
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
