"""Compare inference backends on identical inputs: p50/p95 latency, throughput, load time.

  python benchmarks/inference/run.py [--checkpoint runs/defender/best.pt] [--backends python,onnxruntime,cpp]

Without --checkpoint a tiny toy-env policy is trained so the numbers describe the toy model only.
Latency covers the whole call (validation + normalization + inference + output copy), single process,
threads pinned to 1, same machine/inputs/warmup for every backend. No speedup is assumed or claimed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "training"), str(ROOT / "tests" / "infra")]


def corpus(b: int, max_nodes: int = 8, seed: int = 0) -> dict[str, np.ndarray]:
    from toy_env import ToyGraphEnv

    env, rng = ToyGraphEnv(max_nodes=max_nodes), np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(b):
        rows.append(obs)
        obs, *_ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        if env.t == 0:
            obs, _ = env.reset()
    return {k: np.ascontiguousarray(np.stack([r[k] for r in rows])) for k in rows[0]}


def bench(be, batch, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        be.run(batch)
    t = np.empty(iters)
    for i in range(iters):
        s = time.perf_counter()
        be.run(batch)
        t[i] = time.perf_counter() - s
    b = batch["node_mask"].shape[0]
    return {"p50_us": round(float(np.percentile(t, 50)) * 1e6, 1), "p95_us": round(float(np.percentile(t, 95)) * 1e6, 1),
            "throughput_samples_per_s": round(b / float(t.mean()))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--backends", default="python,onnxruntime,cpp")
    p.add_argument("--batches", default="1,32,256")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    a = p.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    from hospital_rl.infra.inference import NativeBackend, OnnxRuntimeBackend, PythonReferenceBackend, export_onnx

    tmp = Path(tempfile.mkdtemp())
    ckpt = Path(a.checkpoint) if a.checkpoint else None
    if ckpt is None:
        from hospital_rl.training.train import TrainConfig, train

        train(TrainConfig(role="defender", total_steps=256, n_envs=4, rollout_len=32, hidden=64, checkpoint_every=100,
                          env_factory="toy_env:make_env"), tmp, log=lambda s: None)
        ckpt = tmp / "latest.pt"
    onnx_path = tmp / "policy.onnx"
    export_onnx(ckpt, onnx_path)
    makers = {"python": lambda: PythonReferenceBackend(ckpt), "onnxruntime": lambda: OnnxRuntimeBackend(onnx_path),
              "cpp": lambda: NativeBackend(onnx_path)}
    report = {"device": "cpu", "threads": 1, "machine": platform.machine(), "platform": platform.platform(),
              "model": str(a.checkpoint or "toy (trained 256 steps)"), "results": {}}
    for name in a.backends.split(","):
        try:
            s = time.perf_counter()
            be = makers[name]()
            load_ms = round((time.perf_counter() - s) * 1e3, 1)
        except ImportError as e:
            report["results"][name] = {"unavailable": f"{type(e).__name__}: {e}"}
            continue
        report["results"][name] = {"load_ms": load_ms, **{f"batch_{b}": bench(be, corpus(int(b)), a.warmup, a.iters)
                                                          for b in a.batches.split(",")}}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
