"""Daytona backend: the common lifecycle (``lifecycle.GatewayBackend``) over a Daytona gateway.

One lifecycle implementation: ``DaytonaBackend`` is ``GatewayBackend`` + ``DaytonaGateway``; ``cloud_school`` is a thin
caller. The gateway talks to a ``SandboxClient`` (a ~10-method surface). ``SdkClient`` is the only code that imports the
Daytona SDK (pinned: ``SDK_VERSION``; signatures checked against that release, not against a live account).
Tests use ``FakeSandboxClient``. Execution status of this module: fake-provider tested; LIVE UNVERIFIED unless
``docs/daytona.md`` records a capped smoke.

Model: one sandbox per attempt, labelled (owner/run/attempt). The job is the same ``hospital_rl.infra.job`` process
the local and Modal backends run. Artifacts live under ``/home/daytona/runs`` in the sandbox and are *mirrored* to a
local store (``mirror_dir``) with hash verification. A sandbox is deleted only after its files are mirrored and
verified (manifest when present, size otherwise); a failed or timed-out attempt keeps its partial checkpoint in the
mirror so a new attempt can resume. Retries re-run the command in the same sandbox (resumes from its own latest.pt).
Restart recovery is stateless: sandbox id + labels + session command list are the durable provider state.

Credentials: the SDK reads DAYTONA_API_KEY / DAYTONA_API_URL from the process environment. This module never reads
``.env`` files or secret values, and the job inside the sandbox is given no provider credentials.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import tarfile
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.bundle import DEPS, OWNER_LABEL, SAFE_ID, TORCH_CPU, repo_root
from hospital_rl.infra.job import control_path
from hospital_rl.infra.lifecycle import GatewayBackend, Registry, ResourceCaps

SDK_VERSION = "0.225.0"
HOME = "/home/daytona"
ROOT = f"{HOME}/runs"
SESSION = "job"
LABEL_RUN, LABEL_ATTEMPT = "asfalia.run", "asfalia.attempt"
SAFE_REL = re.compile(r"^[A-Za-z0-9._/-]+$")


class DaytonaConfig(BaseModel):
    snapshot: str | None = None  # pinned prebuilt snapshot name (preferred); None builds the pinned image below
    python_version: str = "3.12"
    mirror_dir: str = "runs/daytona/store"
    registry_dir: str = "runs/.registry/daytona"
    auto_stop_min: int = 15  # idle stop
    ttl_min: int = 120  # provider-side auto-delete backstop if the controller never comes back
    mirror_timeout_s: int = 600
    caps: ResourceCaps = ResourceCaps()


def load_daytona_config(path: str | Path) -> DaytonaConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    caps = {**raw.get("caps", {}), "max_steps": raw.get("train", {}).get("total_steps", ResourceCaps().max_steps)}
    return DaytonaConfig(**{k: raw[k] for k in DaytonaConfig.model_fields if k in raw and k != "caps"}, caps=caps)


class SandboxClient(Protocol):
    def create(self, labels: dict[str, str], cpu: int, memory_gb: int) -> str: ...
    def find(self, labels: dict[str, str]) -> list[tuple[str, dict[str, str]]]: ...
    def upload(self, sid: str, path: str, data: bytes) -> None: ...
    def download(self, sid: str, path: str) -> bytes | None: ...
    def walk(self, sid: str, path: str) -> list[tuple[str, int]]:
        """Relative file paths (with sizes) under ``path``; [] if absent."""
    def run(self, sid: str, argv: list[str], timeout_s: int) -> int:
        """Blocking command (setup); returns the exit code."""
    def start(self, sid: str, argv: list[str]) -> str: ...
    def commands(self, sid: str) -> list[tuple[str, int | None]]:
        """[(command id, exit code or None while running)] of the job session, oldest first; [] if no session."""
    def delete(self, sid: str) -> None: ...


def code_bundle() -> bytes:
    """Compressed, immutable source+configs bundle (the same tree whose hash the spec pins)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.add(repo_root() / "src/hospital_rl", "src/hospital_rl", filter=lambda i: None if "__pycache__" in i.name else i)
        t.add(repo_root() / "configs", "configs")
    return buf.getvalue()


class DaytonaGateway:
    def __init__(self, cfg: DaytonaConfig, client: SandboxClient, *, now=time.time):
        self.cfg, self.c, self.now = cfg, client, now
        self.mirror = Path(cfg.mirror_dir)
        self._bundle: bytes | None = None

    # remote_id = "<sandbox>|<started epoch>|<run>|<attempt>": everything needed to re-attach after a restart
    @staticmethod
    def _parts(rid: str) -> tuple[str, float, str, str]:
        sid, t0, run, att = rid.split("|")
        return sid, float(t0), run, att

    def _spec_path(self, run: str, att: str) -> str:
        return str(control_path(ROOT, run, att).with_name(att + ".spec.json"))

    def spawn(self, spec: JobSpec) -> str:
        for v in (spec.run_id, spec.attempt_id, spec.resume_from_attempt):
            if v is not None and not SAFE_ID.match(v):
                raise ValueError(f"unsafe id {v!r}")
        labels = {**OWNER_LABEL, LABEL_RUN: spec.run_id, LABEL_ATTEMPT: spec.attempt_id}
        sid = self.c.create(labels, max(1, round(spec.cpu)), max(1, -(-spec.memory_mb // 1024)))
        try:
            self._bundle = self._bundle or code_bundle()
            self.c.upload(sid, f"{HOME}/b.tgz", self._bundle)
            self.c.upload(sid, self._spec_path(spec.run_id, spec.attempt_id), spec.model_dump_json().encode())
            if spec.resume_from_attempt:  # carry the previous attempt's mirrored files into this sandbox
                prev = self.mirror / spec.run_id / spec.resume_from_attempt
                for f in (p for p in prev.rglob("*") if p.is_file()):
                    self.c.upload(sid, f"{ROOT}/{spec.run_id}/{spec.resume_from_attempt}/{f.relative_to(prev)}", f.read_bytes())
            if (rc := self.c.run(sid, ["sh", "-c", self._setup()], 1800)) != 0:  # constant text: no user input in it
                raise RuntimeError(f"sandbox setup failed (exit {rc})")
            self._launch(sid, spec)
        except Exception:
            self.c.delete(sid)  # never leave a half-built sandbox behind
            raise
        return f"{sid}|{self.now()}|{spec.run_id}|{spec.attempt_id}"

    @staticmethod
    def _setup() -> str:
        deps = shlex.join(DEPS)
        return (f"cd {HOME} && tar xzf b.tgz && pip install -q torch --index-url {shlex.quote(TORCH_CPU)} "
                f"&& pip install -q {deps}")

    def _launch(self, sid: str, spec: JobSpec) -> None:
        argv = ["env", f"PYTHONPATH={HOME}/src", "python", "-m", "hospital_rl.infra.job",
                "--spec", self._spec_path(spec.run_id, spec.attempt_id), "--root", ROOT]
        self.c.start(sid, argv)

    # --- status ----------------------------------------------------------------------------------------------
    def poll(self, rid: str):
        sid, t0, run, att = self._parts(rid)
        cmds = self.c.commands(sid)
        local = self.mirror / run / att
        if not cmds and (local / "result.json").exists():  # sandbox already cleaned after a verified mirror
            return "done", json.loads((local / "result.json").read_text())
        if not cmds:
            return "lost", f"sandbox {sid} or its job session is gone and no mirrored result exists"
        spec = JobSpec.model_validate_json(self.c.download(sid, self._spec_path(run, att)))
        _, code = cmds[-1]
        if code is None:
            if self.now() - t0 > spec.timeout_s + spec.grace_s + 1800:  # + setup time budget
                self._mirror(sid, run, att)
                self.cancel(rid)
                return "timed_out", "hard-killed after timeout + grace; last mirrored checkpoint kept"
            return "running", {}
        if code != 0 and len(cmds) <= spec.max_retries:  # bounded same-attempt retry; resumes from its own latest.pt
            self._launch(sid, spec)
            return "running", {}
        problems = self._mirror(sid, run, att)
        res = local / "result.json"
        if code == 0 and res.exists() and not problems:
            self.c.delete(sid)  # only after the mirror verified
            return "done", json.loads(res.read_text())
        if code == 0:
            return "failed", f"artifact mirror failed verification, sandbox kept: {problems or 'no result.json'}"
        if problems:
            return "failed", f"job exited {code}; mirror problems {problems}, sandbox kept"
        self.c.delete(sid)
        return "failed", f"job exited {code}; partial artifacts mirrored to {local}"

    def _mirror(self, sid: str, run: str, att: str) -> list[str]:
        """Pull the attempt dir and its control-free files to the local store; returns verification problems."""
        base, problems = f"{ROOT}/{run}/{att}", []
        for rel, size in self.c.walk(sid, base):
            if not SAFE_REL.match(rel) or ".." in rel.split("/"):
                problems.append(f"unsafe path {rel!r}")
                continue
            data = self.c.download(sid, f"{base}/{rel}")
            if data is None or len(data) != size:
                problems.append(f"incomplete {rel}")
                continue
            dst = self.mirror / run / att / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp")
            tmp.write_bytes(data)
            tmp.replace(dst)
        man = self.mirror / run / att / "manifest.json"
        if man.exists():
            for name, want in json.loads(man.read_text())["files"].items():
                f = self.mirror / run / att / name
                if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != want["sha256"]:
                    problems.append(f"manifest mismatch {name}")
        return problems

    # --- control ---------------------------------------------------------------------------------------------
    def cancel(self, rid: str) -> None:
        sid, _, run, att = self._parts(rid)
        self._mirror(sid, run, att)  # keep whatever checkpoint exists before the sandbox goes
        self.c.delete(sid)

    def write_control(self, run_id, attempt_id, word):
        for sid, _ in self.c.find({LABEL_RUN: run_id, LABEL_ATTEMPT: attempt_id, **OWNER_LABEL}):
            self.c.upload(sid, str(control_path(ROOT, run_id, attempt_id)), word.encode())

    def read_file(self, run_id, attempt_id, name):
        p = self.mirror / run_id / attempt_id / name
        if p.exists():
            return p.read_bytes()
        for sid, _ in self.c.find({LABEL_RUN: run_id, LABEL_ATTEMPT: attempt_id, **OWNER_LABEL}):
            return self.c.download(sid, f"{ROOT}/{run_id}/{attempt_id}/{name}")
        return None

    def list_attempts(self):
        """Mirrored attempts plus every owner-labelled sandbox the provider reports (orphans show up here)."""
        got = {(r.name, a.name) for r in self.mirror.glob("*") if r.is_dir() for a in r.glob("*") if a.is_dir()} if self.mirror.exists() else set()
        got |= {(lb[LABEL_RUN], lb[LABEL_ATTEMPT]) for _, lb in self.c.find(OWNER_LABEL) if LABEL_RUN in lb}
        return sorted(got)

    def delete_attempt(self, run_id, attempt_id):
        import shutil
        for sid, _ in self.c.find({LABEL_RUN: run_id, LABEL_ATTEMPT: attempt_id, **OWNER_LABEL}):
            self.c.delete(sid)
        shutil.rmtree(self.mirror / run_id / attempt_id, ignore_errors=True)


class DaytonaBackend(GatewayBackend):
    def __init__(self, cfg: DaytonaConfig, gateway: DaytonaGateway, registry: Registry | str | Path | None = None):
        reg = registry if isinstance(registry, Registry) else Registry(registry or cfg.registry_dir)
        super().__init__("daytona", gateway, reg, cfg.caps)
        self.cfg = cfg


class SdkClient:
    """The only Daytona-SDK code. Authentication: the SDK reads DAYTONA_API_KEY from the process environment."""

    def __init__(self, cfg: DaytonaConfig):
        from importlib.metadata import version

        import daytona

        if version("daytona") != SDK_VERSION:
            raise RuntimeError(f"daytona SDK {version('daytona')} installed, {SDK_VERSION} pinned; install daytona=={SDK_VERSION}")
        self.d, self.cfg, self.client = daytona, cfg, daytona.Daytona()

    def _sb(self, sid):
        return self.client.get(sid)

    def create(self, labels, cpu, memory_gb):
        d = self.d
        kw = dict(labels=labels, auto_stop_interval=self.cfg.auto_stop_min, auto_delete_interval=self.cfg.ttl_min)
        if self.cfg.snapshot:
            p = d.CreateSandboxFromSnapshotParams(snapshot=self.cfg.snapshot, **kw)
        else:
            p = d.CreateSandboxFromImageParams(image=d.Image.debian_slim(self.cfg.python_version),
                                               resources=d.Resources(cpu=cpu, memory=memory_gb, disk=10), **kw)
        return self.client.create(p, timeout=300).id

    def find(self, labels):
        return [(s.id, dict(s.labels or {})) for s in self.client.list(self.d.ListSandboxesQuery(labels=labels))]

    def upload(self, sid, path, data):
        self._sb(sid).fs.upload_file(data, path)

    def download(self, sid, path):
        try:
            return self._sb(sid).fs.download_file(path)
        except self.d.DaytonaNotFoundError:
            return None

    def walk(self, sid, path):
        fs, out = self._sb(sid).fs, []
        try:
            stack = [path]
            while stack:
                d = stack.pop()
                for f in fs.list_files(d):
                    full = f"{d}/{f.name}"
                    (stack.append(full) if f.is_dir else out.append((full[len(path) + 1:], int(f.size))))
        except self.d.DaytonaNotFoundError:
            return []
        return out

    def run(self, sid, argv, timeout_s):
        return self._sb(sid).process.exec(shlex.join(argv), cwd=HOME, timeout=timeout_s).exit_code

    def start(self, sid, argv):
        pr = self._sb(sid).process
        try:
            pr.create_session(SESSION)
        except self.d.DaytonaConflictError:
            pass
        r = pr.execute_session_command(SESSION, self.d.SessionExecuteRequest(command=shlex.join(argv), run_async=True))
        return r.cmd_id

    def commands(self, sid):
        try:
            s = self._sb(sid).process.get_session(SESSION)
        except (self.d.DaytonaNotFoundError, self.d.DaytonaError):
            return []
        return [(c.id, c.exit_code) for c in s.commands or []]

    def delete(self, sid):
        try:
            self._sb(sid).delete()
        except self.d.DaytonaNotFoundError:
            pass

