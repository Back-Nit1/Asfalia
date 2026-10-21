"""Deterministic in-process test doubles (named "fake" everywhere they surface). They write the same files a real job does,
so the controller's file ingestion is exercised for real. Never used unless explicitly selected (`--fake-backend`, tests)."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..infra.artifacts import DirArtifactStore
from ..infra.backend import JobBackend, JobSpec, JobState, JobStatus
from ..envs.encoding import obs_schema_hash
from ..policies.checkpoint import atomic_write
from ..schemas import catalog_hash

STEP = 64          # env steps per fake round


class FakeBackend(JobBackend):
    """One round of progress per `status()` call, so tests and the e2e advance deterministically with the controller's polls."""
    name = "fake"

    def __init__(self, root: str | Path, ckpt_every: int = 2):
        self.root, self.every, self.jobs = Path(root), ckpt_every, {}
        self.submits: list[str] = []

    def _dir(self, s: JobSpec) -> Path:
        return DirArtifactStore(self.root).attempt_dir(s.run_id, s.attempt_id)

    def submit(self, spec: JobSpec) -> str:
        jid = f"fake:{spec.run_id}:{spec.attempt_id}"
        if jid in self.jobs:
            return jid
        self.submits.append(jid)
        d = self._dir(spec)
        d.mkdir(parents=True, exist_ok=True)
        steps, version, parent = 0, 0, None
        if spec.resume_from_attempt:
            prev = DirArtifactStore(self.root).attempt_dir(spec.run_id, spec.resume_from_attempt)
            meta = json.loads((prev / "latest.pt.meta.json").read_text())
            steps, version, parent = meta["training_steps"], meta["policy_version"], meta["checkpoint_id"]
        self.jobs[jid] = {"spec": spec, "state": JobState.RUNNING, "steps": steps, "version": version, "parent": parent,
                          "rounds": 0, "ctl": None, "saved": steps, "detail": ""}
        return jid

    def _save(self, j) -> None:
        s, d = j["spec"], self._dir(j["spec"])
        j["version"] += 1
        cid = hashlib.sha256(f"{s.run_id}/{s.attempt_id}/{j['version']}".encode()).hexdigest()[:16]
        pt = d / f"ckpt_{j['version']:06d}.pt"
        atomic_write(pt, f"fake-weights:{cid}".encode())
        role = s.train_config["role"]
        meta = {"checkpoint_id": cid, "parent_checkpoint_id": j["parent"], "role": role, "catalog_hash": catalog_hash(),
                "observation_schema_hash": obs_schema_hash(role), "training_steps": j["steps"], "policy_version": j["version"],
                "config_hash": "fake", "seed": s.train_config["seed"]}
        atomic_write(pt.with_name(pt.name + ".meta.json"), json.dumps(meta).encode())
        atomic_write(d / "latest.pt", pt.read_bytes())
        atomic_write(d / "latest.pt.meta.json", json.dumps(meta).encode())
        j["parent"], j["saved"] = cid, j["steps"]

    def _finish(self, j, state: JobState, reason: str) -> None:
        s, d = j["spec"], self._dir(j["spec"])
        if j["saved"] != j["steps"] or not (d / "latest.pt").exists():
            self._save(j)
        atomic_write(d / "result.json", json.dumps({"state": state.value, "env_steps": j["steps"], "reason": reason}).encode())
        DirArtifactStore(self.root).publish_manifest(s.run_id, s.attempt_id)
        j["state"] = state

    def status(self, job_id: str) -> JobStatus:
        if job_id not in self.jobs:
            raise LookupError(job_id)
        j = self.jobs[job_id]
        if j["state"] is JobState.RUNNING:
            s, d = j["spec"], self._dir(j["spec"])
            if j["ctl"] == "stop":
                self._finish(j, JobState.STOPPED, "checkpoint-and-stop")
            else:
                j["steps"] += STEP
                j["rounds"] += 1
                row = {"env_steps": j["steps"], "policy_version": j["version"] + 1, "mean_return": round(-5 + j["steps"] / 100, 3),
                       "t": float(j["rounds"])}
                with open(d / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                with open(d / "episodes.jsonl", "a") as f:
                    f.write(json.dumps({"event_type": "episode_sample", "visibility": "operator", "episode_id": f"{s.run_id}-e{j['rounds']}",
                                        "payload": {"return": row["mean_return"]}}) + "\n")
                if j["rounds"] % self.every == 0:
                    self._save(j)
                if j["steps"] >= s.train_config["total_steps"]:
                    self._finish(j, JobState.DONE, "done")
                elif j["ctl"] == "drain":
                    self._finish(j, JobState.DRAINED, "drain")
        return JobStatus(job_id=job_id, state=j["state"], detail=j["detail"])

    def drain(self, job_id): self.jobs[job_id]["ctl"] = "drain"
    def checkpoint_and_stop(self, job_id): self.jobs[job_id]["ctl"] = "stop"

    def force_cancel(self, job_id):
        j = self.jobs[job_id]
        if not j["state"].terminal:
            j["state"], j["detail"] = JobState.CANCELLED, "force-cancelled"


class FakeEvaluator:
    """Deterministic scores derived from checkpoint step counts. Exports the fixture episode as the replayable episode and
    marks it as such (label FIXTURE, or DEBUG for forks) so no fabricated trace passes as a real recording."""
    name = "fake"

    def __init__(self, fixture: str | Path):
        self.fixture = Path(fixture)

    def run(self, spec: dict, checkpoint_paths: dict[str, str], out_dir: str) -> dict:
        if spec.get("fail"):
            raise RuntimeError("fake evaluator failure")
        seeds = int(hashlib.sha256((spec["attacker_sha256"] + spec["defender_sha256"]).encode()).hexdigest()[:6], 16)
        mean = round(0.3 + (seeds % 400) / 1000, 3)
        half = round(1.96 * 0.2 / max(spec["episodes"], 1) ** 0.5, 3)
        raw = json.loads(self.fixture.read_text())
        raw.update(fixture=True, debug=spec["suite"] == "debug_fork")
        eid = f"{'debug' if raw['debug'] else 'eval'}_{seeds:06x}"
        Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
        atomic_write(Path(out_dir).parent / f"{eid}.json", json.dumps(raw).encode())
        return {"pairs": [{"attacker": spec["attacker_checkpoint_id"], "defender": spec["defender_checkpoint_id"],
                           "metric": "defender_win_rate", "mean": mean, "ci_low": round(mean - half, 3), "ci_high": round(mean + half, 3),
                           "n": spec["episodes"]}], "episode_ids": [eid], "frozen": True, "note": "deterministic fake evaluator"}
