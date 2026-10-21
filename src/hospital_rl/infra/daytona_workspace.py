"""Bounded coding-workspace smoke: a sandbox at a pinned commit installs controlled dependencies, runs ONE real test
file and returns commit + artifact evidence. Code goes in as ``git archive`` of the pinned commit (no git credentials,
no clone, no host files); only a fixed argv is executed; the sandbox is always deleted at the end.
"""
from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from pathlib import Path

from hospital_rl.infra.bundle import DEPS, OWNER_LABEL, TORCH_CPU
from hospital_rl.infra.daytona_backend import HOME, LABEL_ATTEMPT, LABEL_RUN, SandboxClient

SHA = re.compile(r"^[0-9a-f]{40}$")
TEST_FILE = re.compile(r"^tests/[A-Za-z0-9_/]+\.py$")
EXTRA_DEPS = ["pytest"]  # controlled install list = runtime deps + pytest; nothing else


def pinned_archive(repo: Path, commit: str) -> tuple[str, bytes]:
    """Resolve ``commit`` to a full sha and return (sha, tar.gz of exactly that tree). argv-only, no shell."""
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"], capture_output=True, text=True, check=True).stdout.strip()
    if not SHA.match(sha):
        raise ValueError(f"commit did not resolve to a full sha: {commit!r}")
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz", sha], capture_output=True, check=True).stdout
    return sha, tar


def workspace_smoke(client: SandboxClient, repo: Path, commit: str, test_file: str, out_dir: Path, *,
                    run_id: str = "workspace-smoke", timeout_s: int = 600, cpu: int = 1, memory_gb: int = 2) -> dict:
    if not TEST_FILE.match(test_file) or ".." in test_file:
        raise ValueError(f"test_file {test_file!r} must be tests/<path>.py")
    sha, tar = pinned_archive(repo, commit)
    sid = client.create({**OWNER_LABEL, LABEL_RUN: run_id, LABEL_ATTEMPT: sha[:12]}, cpu, memory_gb)
    try:
        client.upload(sid, f"{HOME}/w.tgz", tar)
        setup = (f"cd {HOME} && mkdir -p w && tar xzf w.tgz -C w && pip install -q torch --index-url {shlex.quote(TORCH_CPU)} "
                 f"&& pip install -q {shlex.join(DEPS + EXTRA_DEPS)}")
        if (rc := client.run(sid, ["sh", "-c", setup], 1800)) != 0:
            raise RuntimeError(f"workspace setup failed (exit {rc})")
        report = f"{HOME}/report.txt"
        test = shlex.join(["python", "-m", "pytest", "-q", "-x", test_file])
        rc = client.run(sid, ["sh", "-c", f"cd {HOME}/w && PYTHONPATH=src {test} > {report} 2>&1"], timeout_s)
        data = client.download(sid, report)
        if data is None:
            raise RuntimeError("no test report was produced")
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{sha[:12]}.report.txt"
        dst.write_bytes(data)
        verified = hashlib.sha256(dst.read_bytes()).hexdigest() == hashlib.sha256(client.download(sid, report) or b"").hexdigest()
        return {"commit": sha, "archive_sha256": hashlib.sha256(tar).hexdigest(), "test_file": test_file, "exit_code": rc,
                "report": str(dst), "report_sha256": hashlib.sha256(data).hexdigest(), "upload_verified": verified}
    finally:
        client.delete(sid)
