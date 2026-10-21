"""Read-only view of what a job actually wrote. The controller never trusts a backend's word for metrics/checkpoints:
it ingests files. Layout matches `infra/job.py`: <root>/<run>/<attempt>/{ckpt_*.pt(.meta.json), result.json, manifest.json,
metrics.jsonl, episodes.jsonl}.

`metrics.jsonl` / `episodes.jsonl` are NOT written by the current trainer (see docs/requests/agent-D.md); until they exist a
real run honestly shows no live metrics."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Protocol

from ..infra.artifacts import DirArtifactStore


class Reader(Protocol):
    def metrics(self, run: str, attempt: str) -> list[dict]: ...
    def episode_events(self, run: str, attempt: str) -> list[dict]: ...
    def checkpoints(self, run: str, attempt: str) -> list[dict]: ...
    def result(self, run: str, attempt: str) -> dict | None: ...
    def problems(self, run: str, attempt: str) -> list[str]: ...
    def files(self, run: str, attempt: str) -> list[dict]: ...
    def attempts(self) -> list[tuple[str, str]]: ...
    def delete_attempt(self, run: str, attempt: str) -> None: ...
    def checkpoint_path(self, run: str, attempt: str, name: str) -> Path: ...


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            break       # torn final line from a live writer; pick it up next poll
    return out


class FsReader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._store = DirArtifactStore(root)

    def _d(self, run: str, attempt: str) -> Path:
        return self._store.attempt_dir(run, attempt)

    def metrics(self, run, attempt):
        return _jsonl(self._d(run, attempt) / "metrics.jsonl")

    def episode_events(self, run, attempt):
        return _jsonl(self._d(run, attempt) / "episodes.jsonl")

    def checkpoints(self, run, attempt):
        out = []
        for meta in sorted(self._d(run, attempt).glob("ckpt_*.pt.meta.json")):
            pt = meta.with_name(meta.name[: -len(".meta.json")])
            if pt.exists():
                out.append({"name": pt.name, "path": str(pt.relative_to(self.root)), "size": pt.stat().st_size,
                            "sha256": hashlib.sha256(pt.read_bytes()).hexdigest(), "meta": json.loads(meta.read_text())})
        return out

    def result(self, run, attempt):
        p = self._d(run, attempt) / "result.json"
        return json.loads(p.read_text()) if p.exists() else None

    def problems(self, run, attempt):
        return self._store.verify_manifest(run, attempt)

    def files(self, run, attempt):
        d = self._d(run, attempt)
        return [{"name": str(p.relative_to(d)), "path": str(p.relative_to(self.root)), "size": p.stat().st_size}
                for p in sorted(d.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")]

    def attempts(self):
        return [(r.name, a.name) for r in sorted(self.root.glob("*")) if r.is_dir() for a in sorted(r.glob("*"))
                if a.is_dir() and a.name != "control"]

    def delete_attempt(self, run, attempt):
        shutil.rmtree(self._d(run, attempt))

    def checkpoint_path(self, run, attempt, name):
        p = (self._d(run, attempt) / name).resolve()
        if self._d(run, attempt).resolve() not in p.parents:
            raise ValueError("path escapes attempt dir")
        return p
