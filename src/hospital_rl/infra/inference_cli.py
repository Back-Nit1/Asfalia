"""``hospital-rl export-policy``: export a checkpoint to a verified ONNX model.

Exit 0: exported (one JSON result object on stdout). Exit 2: unusable input or refusal (message on stderr, JSON
``{"ok": false, "error": ...}`` on stdout). Nothing is written to ``--out`` unless the export fully succeeded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hospital-rl export-policy", description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="source .pt checkpoint")
    p.add_argument("--out", required=True, help="destination .onnx (metadata goes to <out>.meta.json)")
    p.add_argument("--format", choices=["onnx"], default="onnx")
    p.add_argument("--expect-role", choices=["attacker", "defender"])
    p.add_argument("--expect-catalog-hash")
    p.add_argument("--expect-observation-hash")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out (default: refuse)")
    return p


def _fail(msg: str) -> int:
    print(f"export-policy: {msg}", file=sys.stderr)
    print(json.dumps({"ok": False, "error": msg}))
    return 2


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    ckpt, out = Path(a.checkpoint), Path(a.out)
    meta_path = Path(str(out) + ".meta.json")
    if not ckpt.is_file():
        return _fail(f"checkpoint not found: {ckpt}")
    if not a.overwrite and (out.exists() or meta_path.exists()):
        return _fail(f"{out} (or its .meta.json) exists; pass --overwrite to replace")
    try:
        from .inference import OnnxRuntimeBackend, export_onnx
        from ..policies.checkpoint import IncompatibleCheckpointError

        out.parent.mkdir(parents=True, exist_ok=True)
        meta = export_onnx(ckpt, out, a.expect_role, a.expect_catalog_hash, a.expect_observation_hash)
        # verify what is on disk, not what we think we wrote
        OnnxRuntimeBackend(out, a.expect_catalog_hash, a.expect_observation_hash, expected_role=a.expect_role)
    except IncompatibleCheckpointError as e:
        return _fail(str(e))
    except Exception as e:  # unreadable/corrupt checkpoint, unsupported (e.g. recurrent) policy
        return _fail(f"{type(e).__name__}: {e}")
    print(json.dumps({"ok": True, "out": str(out), "meta": str(meta_path), "format": a.format,
                      **{k: meta[k] for k in ("checkpoint_id", "role", "catalog_hash", "observation_schema_hash",
                                              "max_nodes", "policy_version", "training_steps")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
