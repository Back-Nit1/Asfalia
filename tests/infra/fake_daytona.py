"""In-process stand-in for the Daytona SDK (SandboxClient): each sandbox is a temp dir; the job is the real run_job."""
import threading
from pathlib import Path

from hospital_rl.infra.backend import JobSpec
from hospital_rl.infra.daytona_backend import HOME
from hospital_rl.infra.job import run_job


class FakeSandboxClient:
    def __init__(self, base: Path, script=None):
        self.base, self.script = Path(base), list(script or [])  # script: per job start, "run" | "fail" | "hang"
        self.boxes: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.trim: set[str] = set()  # file names whose download comes back truncated (partial transfer)
        self.n = 0

    def _p(self, sid, path):  # sandbox path -> host path
        return self.base / sid / str(path).removeprefix(HOME + "/")

    def create(self, labels, cpu, memory_gb):
        self.n += 1
        sid = f"sb-{self.n}"
        self.boxes[sid] = {"labels": labels, "cmds": [], "cpu": cpu, "mem": memory_gb}
        (self.base / sid).mkdir(parents=True)
        return sid

    def adopt(self, labels):  # a sandbox created by something else/earlier (orphan or foreign)
        sid = self.create(labels, 1, 1)
        return sid

    def find(self, labels):
        return [(s, b["labels"]) for s, b in self.boxes.items() if s not in self.deleted
                and all(b["labels"].get(k) == v for k, v in labels.items())]

    def upload(self, sid, path, data):
        p = self._p(sid, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def download(self, sid, path):
        p = self._p(sid, path)
        if not p.exists() or sid in self.deleted:
            return None
        d = p.read_bytes()
        return d[: len(d) // 2] if p.name in self.trim else d

    def walk(self, sid, path):
        p = self._p(sid, path)
        return [(str(f.relative_to(p)), f.stat().st_size) for f in sorted(p.rglob("*")) if f.is_file()] if p.exists() else []

    def run(self, sid, argv, timeout_s):
        return 0

    def start(self, sid, argv):
        box, kind = self.boxes[sid], (self.script.pop(0) if self.script else "run")
        cmd = {"id": f"c{len(box['cmds'])}", "code": None}
        box["cmds"].append(cmd)
        if kind == "fail":
            cmd["code"] = 1
        elif kind == "run":
            spec = JobSpec.model_validate_json(self._p(sid, argv[argv.index("--spec") + 1]).read_bytes())

            def work():
                try:
                    run_job(spec, self._p(sid, argv[argv.index("--root") + 1]), log=lambda s: None)
                    cmd["code"] = 0
                except Exception:  # noqa: BLE001
                    cmd["code"] = 1

            cmd["thread"] = threading.Thread(target=work, daemon=True)
            cmd["thread"].start()
        return cmd["id"]

    def commands(self, sid):
        return [(c["id"], c["code"]) for c in self.boxes[sid]["cmds"]] if sid in self.boxes and sid not in self.deleted else []

    def delete(self, sid):
        for c in self.boxes[sid]["cmds"]:
            if t := c.get("thread"):
                t.join(60)
        self.deleted.append(sid)
