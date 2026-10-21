"""``hospital-rl benchmark-inference``: compare inference backends on identical raw inputs.

Inputs come from a golden corpus of real hospital observations (several topology families and sizes, padded to
MAX_NODES, plus an extreme-but-finite variant). Every backend is first parity-checked against the Python reference on
that corpus; a backend that disagrees, or whose metadata names a different model, makes the comparison invalid and the
run exits 2 instead of publishing numbers. Backends that are merely not installed are reported as unavailable.

Timed unit = backend.run (validation + normalization + inference + output copy) + argmax action selection.
No speedup is assumed or claimed; the default backend stays Python/ONNX unless this report says otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
BACKENDS = ("python", "python_compiled", "onnxruntime", "cpp")
CELLS = [(f, s) for s in (12, 24, 40, 60) for f in ("hub", "tiered", "flat", "segmented")]
TOL = 1e-4
REPORT_KEYS = {"schema_version", "model", "provenance", "corpus", "env", "backends", "rows"}
ROW_KEYS = {"backend", "batch", "p50_us", "p95_us", "throughput_samples_per_s"}


def to_batch(rows: list[dict]) -> dict[str, np.ndarray]:
    """Stack env observations into the backend contract (env masks are int8; the contract is bool)."""
    from .inference import DTYPES

    return {k: np.ascontiguousarray(np.stack([r[k] for r in rows]), dtype=DTYPES[k]) for k in DTYPES}


def golden_corpus(role: str, n: int, seed: int = 0, extreme: bool = False) -> dict[str, np.ndarray]:
    """n real observations for ``role``, round-robin over (family, size) cells, random legal play for both roles."""
    from ..envs.parallel_env import HospitalParallelEnv
    from ..scenarios import generate

    rng = np.random.default_rng(seed)
    envs = [HospitalParallelEnv(lambda s, f=f, z=z: generate(f, z, s, difficulty=0.7)) for f, z in CELLS]
    obs = [e.reset(seed=seed + i)[0] for i, e in enumerate(envs)]
    rows = []
    for i in range(n):
        c = i % len(CELLS)
        rows.append(obs[c][role])
        pick = {r: int(rng.choice(np.flatnonzero(obs[c][r]["action_mask"]))) for r in obs[c]}
        obs[c], _, _, _, _ = envs[c].step(pick)
        if not envs[c].agents:
            obs[c] = envs[c].reset(seed=seed + n + i)[0]
    batch = to_batch(rows)
    if extreme:
        batch["node_features"] = np.ascontiguousarray(batch["node_features"] * np.float32(1e6))
        batch["globals"] = np.ascontiguousarray(batch["globals"] - np.float32(5e5))
        batch["history"] = np.ascontiguousarray(batch["history"] * np.float32(-1e4))
    return batch


def corpus_sha(batch: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(batch):
        h.update(k.encode() + np.ascontiguousarray(batch[k]).tobytes())
    return h.hexdigest()[:16]


def parity_error(ref, other, batch) -> str | None:
    """None if ``other`` matches ``ref`` per the documented contract, else a one-line reason."""
    r, o = ref.run(batch), other.run(batch)
    legal = batch["action_mask"]
    if not np.allclose(o.logits[legal], r.logits[legal], atol=TOL, rtol=TOL):
        return f"logits differ by up to {np.abs(o.logits[legal] - r.logits[legal]).max():.2e}"
    if not np.allclose(o.value, r.value, atol=TOL, rtol=TOL):
        return "values differ"
    top2 = np.sort(np.where(legal, r.logits, -np.inf), -1)[:, -2:]
    tie = (top2[:, 1] - top2[:, 0]) < TOL  # documented near-tie exception
    if (o.logits.argmax(-1)[~tie] != r.logits.argmax(-1)[~tie]).any():
        return "argmax differs outside the near-tie band"
    return None


def time_backend(be, batch, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        be.run(batch).logits.argmax(-1)
    t = np.empty(reps)
    for i in range(reps):
        s = time.perf_counter()
        be.run(batch).logits.argmax(-1)
        t[i] = time.perf_counter() - s
    b = batch["node_mask"].shape[0]
    return {"p50_us": round(float(np.percentile(t, 50)) * 1e6, 1), "p95_us": round(float(np.percentile(t, 95)) * 1e6, 1),
            "throughput_samples_per_s": round(b / float(t.mean()))}


def env_steps_per_s(be, role: str, steps: int, seed: int = 0) -> float:
    """Full single-role env loop (sim step + opponent + obs encode + backend at B=1), deterministic argmax."""
    from ..envs.factories import hospital_single

    env = hospital_single(role, size=40, split="validation")
    obs, _ = env.reset(seed=seed)
    s, n = time.perf_counter(), 0
    while n < steps:
        a = int(be.run(to_batch([obs])).logits.argmax(-1)[0])
        obs, _, te, tr, _ = env.step(a)
        n += 1
        if te or tr:
            obs, _ = env.reset(seed=seed + n)
    return round(n / (time.perf_counter() - s), 1)


def provenance(threads: int) -> dict:
    import torch

    p = {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
         "cpu_count": os.cpu_count(), "threads": threads, "device": "cpu", "torch": torch.__version__,
         "numpy": np.__version__}
    try:
        import onnxruntime
        p["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    return p


def validate_report(r: dict) -> None:
    """Raise AssertionError if ``r`` is not a schema-valid report."""
    assert REPORT_KEYS <= set(r) and r["schema_version"] == SCHEMA_VERSION, "bad report envelope"
    for row in r["rows"]:
        assert ROW_KEYS <= set(row), f"bad row {row}"
    for name, res in r["backends"].items():
        assert name in BACKENDS and ("unavailable" in res or {"load_ms", "first_run_ms", "batches"} <= set(res))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hospital-rl benchmark-inference", description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backends", default=",".join(BACKENDS), help=f"comma list of {BACKENDS}")
    p.add_argument("--batches", default="1,32,256")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repetitions", type=int, default=300)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--env-steps", type=int, default=300, help="full-env steps per backend (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the JSON report here (also printed to stdout)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    names = a.backends.split(",")
    bad = [n for n in names if n not in BACKENDS]
    if bad:
        print(f"benchmark-inference: unknown backends {bad}", file=sys.stderr)
        return 2
    if not Path(a.checkpoint).is_file():
        print(f"benchmark-inference: checkpoint not found: {a.checkpoint}", file=sys.stderr)
        return 2
    import torch

    from .inference import (NativeBackend, OnnxRuntimeBackend, PythonCompiledBackend, PythonReferenceBackend,
                            export_onnx)

    torch.set_num_threads(a.threads)
    try:
        ref = PythonReferenceBackend(a.checkpoint)
    except Exception as e:
        print(f"benchmark-inference: unusable checkpoint: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    role, meta = ref.meta["role"], ref.meta
    onnx_path = Path(tempfile.mkdtemp()) / "policy.onnx"
    makers = {"python": lambda: PythonReferenceBackend(a.checkpoint),
              "python_compiled": lambda: PythonCompiledBackend(a.checkpoint)}
    if {"onnxruntime", "cpp"} & set(names):
        export_onnx(a.checkpoint, onnx_path, expect_role=role)
        makers["onnxruntime"] = lambda: OnnxRuntimeBackend(onnx_path, meta["catalog_hash"], meta["observation_schema_hash"],
                                                           a.threads, role)
        makers["cpp"] = lambda: NativeBackend(onnx_path, meta["catalog_hash"], meta["observation_schema_hash"],
                                              a.threads, role)
    batches = [int(b) for b in a.batches.split(",")]
    parity_in = {"typical": golden_corpus(role, 64, a.seed), "extreme": golden_corpus(role, 64, a.seed, extreme=True)}
    report = {"schema_version": SCHEMA_VERSION, "model": {k: meta[k] for k in ("checkpoint_id", "role", "catalog_hash",
              "observation_schema_hash", "max_nodes", "policy_version", "training_steps")} | {"path": str(a.checkpoint)},
              "provenance": provenance(a.threads),
              "corpus": {"sha256": {k: corpus_sha(v) for k, v in parity_in.items()}, "seed": a.seed,
                         "cells": [f"{f}-{s}" for f, s in CELLS], "parity_rows": 64},
              "env": {"role": role, "size": 40, "steps": a.env_steps}, "backends": {}, "rows": []}
    for name in names:
        try:
            s = time.perf_counter()
            be = makers[name]()
            load_ms = round((time.perf_counter() - s) * 1e3, 1)
            s = time.perf_counter()
            be.run(parity_in["typical"])
            first_ms = round((time.perf_counter() - s) * 1e3, 1)
        except (ImportError, OSError, RuntimeError) as e:  # missing optional dependency / toolchain / unsupported platform
            report["backends"][name] = {"unavailable": f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"}
            continue
        if be.meta["checkpoint_id"] != meta["checkpoint_id"] or be.meta["role"] != role:
            print(f"benchmark-inference: {name} loaded a different model; refusing to compare", file=sys.stderr)
            return 2
        for kind, inp in parity_in.items():
            err = parity_error(ref, be, inp)
            if err:
                print(f"benchmark-inference: {name} not comparable ({kind} corpus): {err}", file=sys.stderr)
                return 2
        res = {"load_ms": load_ms, "first_run_ms": first_ms, "memory_mb": None, "batches": {}}
        for b in batches:
            res["batches"][str(b)] = t = time_backend(be, golden_corpus(role, b, a.seed), a.warmup, a.repetitions)
            report["rows"].append({"backend": name, "batch": b, **t})
        if a.env_steps:
            res["env_steps_per_s"] = env_steps_per_s(be, role, a.env_steps, a.seed)
        report["backends"][name] = res
    report["notes"] = ["memory_mb is null: peak RSS in one process cannot be attributed to a single backend",
                       "load_ms is in-process and uses a warm file cache; first_run_ms includes lazy init/compile"]
    validate_report(report)
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
