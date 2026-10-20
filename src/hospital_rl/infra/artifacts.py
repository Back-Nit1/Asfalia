"""Directory-backed ArtifactStore. On a Modal Volume mount pass ``commit``/``reload`` so published
files are made visible to other containers (a mounted Volume is not an immediately consistent bus)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from hospital_rl.infra.backend import ArtifactRef, ArtifactStore
from hospital_rl.policies.checkpoint import atomic_write

MANIFEST = "manifest.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DirArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path, commit: Callable[[], None] = lambda: None,
                 reload: Callable[[], None] = lambda: None):
        self.root, self._commit, self._reload = Path(root), commit, reload

    def attempt_dir(self, run_id: str, attempt_id: str) -> Path:
        return self.root / run_id / attempt_id

    def put(self, run_id, attempt_id, name, data):
        path = self.attempt_dir(run_id, attempt_id) / name
        if path.exists():
            raise FileExistsError(f"artifact {path} already published (immutable)")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, data)
        self._commit()
        return ArtifactRef(path=str(path.relative_to(self.root)), sha256=hashlib.sha256(data).hexdigest(), size=len(data))

    def get(self, ref):
        self._reload()
        data = (self.root / ref.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError(f"hash mismatch for {ref.path}")
        return data

    def _files(self, d: Path) -> dict[str, Path]:
        return {str(p.relative_to(d)): p for p in sorted(d.rglob("*"))
                if p.is_file() and p.name != MANIFEST and not p.name.endswith(".tmp")}

    def publish_manifest(self, run_id, attempt_id):
        d = self.attempt_dir(run_id, attempt_id)
        if (d / MANIFEST).exists():
            raise FileExistsError(f"manifest for {run_id}/{attempt_id} already published")
        manifest = {"run_id": run_id, "attempt_id": attempt_id,
                    "files": {n: {"sha256": sha256_file(p), "size": p.stat().st_size} for n, p in self._files(d).items()}}
        atomic_write(d / MANIFEST, json.dumps(manifest, indent=2).encode())
        self._commit()
        return manifest

    def verify_manifest(self, run_id, attempt_id):
        self._reload()
        d = self.attempt_dir(run_id, attempt_id)
        if not (d / MANIFEST).exists():
            return [f"{MANIFEST} missing"]
        want = json.loads((d / MANIFEST).read_text())["files"]
        have = self._files(d)
        problems = [f"missing {n}" for n in want if n not in have]
        problems += [f"changed {n}" for n, p in have.items() if n in want and sha256_file(p) != want[n]["sha256"]]
        problems += [f"unlisted {n}" for n in have if n not in want]
        return problems
