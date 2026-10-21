"""Deterministic experiment controller. Commands only write intent + state to SQLite and return; `tick()` (run by a worker
thread, or by tests directly) submits to the injected backend, polls it, and ingests the files the job really wrote.
State is never inferred beyond what the backend or the artifact files report: a lost job becomes `failed`, not `completed`."""
from __future__ import annotations

import json
import re
import threading
import time
from functools import wraps
from typing import Callable, Protocol

from pydantic import BaseModel

from ..events import EventEnvelope
from ..infra.backend import JobBackend, JobSpec, JobState
from .config import Resolved, RunRequest, resolve
from .db import ExpDb
from .reader import Reader
from .states import ACTIVE, RESUMABLE, TERMINAL, IllegalRunTransition, RunState, check

MAX_INGEST = 500            # metric rows per run per tick
MAX_LIVE_EVENTS = 20        # events published to the live feed per run per tick (sampling bound)
MAX_EPISODE_EVENTS = 50
MAX_EVAL_EPISODES = 20
SUITES = {"baseline", "transfer", "adaptation", "cross_play", "frozen"}
_SECRET = re.compile(r"(?i)((?:token|secret|api[_-]?key|password|authorization)\W{0,3})\S+")


def redact(s: object) -> str:
    return _SECRET.sub(r"\1[redacted]", str(s))[:500]


class ConfigError(ValueError):
    """422: the request is invalid (config, caps, checkpoint compatibility). Carries every problem."""
    def __init__(self, errors: list[str] | str):
        self.errors = [errors] if isinstance(errors, str) else errors
        super().__init__("; ".join(self.errors))


class Evaluator(Protocol):
    name: str
    def run(self, spec: dict, checkpoint_paths: dict[str, str], out_dir: str) -> dict: ...


class EvalRequest(BaseModel):
    suite: str = "frozen"
    attacker_checkpoint_id: str
    defender_checkpoint_id: str
    episodes: int = 4
    seed: int = 10_000


def audited(action: str):
    def deco(fn):
        @wraps(fn)
        def inner(self, *a, actor: str = "operator", **kw):
            target = str(a[0]) if a and isinstance(a[0], str) else None
            with self.mu:
                try:
                    out = fn(self, *a, actor=actor, **kw)
                except Exception as e:
                    self.audit(actor, action, target, "refused", redact(e))
                    raise
                self.audit(actor, action, target, "ok", "")
                return out
        return inner
    return deco


class Controller:
    def __init__(self, store, backend: JobBackend, reader: Reader, *, clock: Callable[[], float] = time.time,
                 stale_after: float = 30.0, max_active: int = 1, evaluator: Evaluator | None = None,
                 episodes_dir: str | None = None, on_new_episodes: Callable[[], None] = lambda: None):
        self.mu = threading.RLock()     # ponytail: one lock for commands + tick; slow remote backends would need per-run locks
        self.store, self.db, self.backend, self.reader, self.clock = store, ExpDb(store), backend, reader, clock
        self.stale_after, self.max_active, self.evaluator = stale_after, max_active, evaluator
        self.episodes_dir, self.on_new_episodes = episodes_dir, on_new_episodes

    # ---------- events / audit ----------
    def emit(self, kind: str, worker: str, event_type: str, run_id: str | None, payload: dict, visibility="operator",
             episode_id: str | None = None) -> None:
        self.store.append_event(EventEnvelope(source_kind=kind, worker_id=worker, sequence=self.store.next_sequence(kind, worker),
                                              event_type=event_type, run_id=run_id, visibility=visibility,
                                              episode_id=episode_id, payload=payload))

    def audit(self, actor, action, target, outcome, detail) -> None:
        self.db.x("INSERT INTO exp_audit (ts, actor, action, target, outcome, detail) VALUES (?,?,?,?,?,?)",
                  (self.clock(), actor, action, target, outcome, detail))

    def audit_log(self, limit: int = 100) -> list[dict]:
        return self.db.q("SELECT * FROM exp_audit ORDER BY id DESC LIMIT ?", (limit,))

    # ---------- reads ----------
    def get(self, run_id: str) -> dict:
        r = self.db.one("SELECT * FROM exp_runs WHERE id=?", (run_id,))
        if r is None:
            raise KeyError(run_id)
        return self._view(r)

    def list(self) -> list[dict]:
        return [self._view(r) for r in self.db.q("SELECT * FROM exp_runs ORDER BY n DESC")]

    def attempts(self, run_id: str) -> list[dict]:
        return self.db.q("SELECT * FROM exp_attempts WHERE run_id=? ORDER BY n", (run_id,))

    def _view(self, r: dict) -> dict:
        now = self.clock()
        hb = r["last_heartbeat"]
        active = r["state"] in {s.value for s in ACTIVE}
        r["heartbeat_age_s"] = None if hb is None else round(now - hb, 3)
        r["stale"] = bool(active and (hb is None and now - r["created_at"] > self.stale_after
                                      or hb is not None and now - hb > self.stale_after))
        r["resumable"] = r["state"] in {s.value for s in RESUMABLE} and bool(self.checkpoints(r["id"], latest_only=True))
        r["attempts"] = self.attempts(r["id"])
        r["latest_metric"] = (self.db.one("SELECT data FROM exp_metrics WHERE run_id=? ORDER BY rowid DESC LIMIT 1", (r["id"],)) or {}).get("data")
        return r

    def metrics(self, run_id: str, after: int = -1, limit: int = 500) -> list[dict]:
        self.get(run_id)
        rows = self.db.q("SELECT attempt_id, seq, env_steps, data FROM exp_metrics WHERE run_id=? ORDER BY rowid", (run_id,))
        return rows[after + 1: after + 1 + limit] if after >= -1 else rows[:limit]

    def checkpoints(self, run_id: str | None = None, latest_only: bool = False) -> list[dict]:
        rows = self.db.q("SELECT * FROM exp_checkpoints WHERE (? IS NULL OR run_id=?) ORDER BY created_at, rowid", (run_id, run_id))
        return rows[-1:] if latest_only else rows

    def artifacts(self, run_id: str) -> list[dict]:
        self.get(run_id)
        out = []
        for a in self.attempts(run_id):
            if a["cleaned_at"]:
                out.append({"attempt_id": a["attempt_id"], "cleaned": True, "files": [], "problems": [], "partial": False})
                continue
            problems = self.reader.problems(run_id, a["attempt_id"])
            out.append({"attempt_id": a["attempt_id"], "cleaned": False, "files": self.reader.files(run_id, a["attempt_id"]),
                        "problems": problems, "partial": bool(problems) or a["ended_at"] is None})
        return out

    # ---------- commands ----------
    def validate(self, req: RunRequest) -> Resolved:
        try:
            return resolve(req, self.backend.name, self._measured())
        except ValueError as e:
            raise ConfigError(str(e).split("; ")) from e

    def _measured(self) -> tuple[float, float] | None:
        rows = self.db.q("SELECT data FROM exp_metrics ORDER BY rowid DESC LIMIT 2")
        pts = [x["data"] for x in rows if x["data"].get("env_steps") is not None and x["data"].get("t") is not None]
        if len(pts) == 2 and pts[0]["t"] > pts[1]["t"]:
            return (pts[0]["env_steps"] - pts[1]["env_steps"]) / (pts[0]["t"] - pts[1]["t"]), pts[0]["t"]
        return None

    def _check_capacity(self) -> None:
        n = self.db.one("SELECT COUNT(*) c FROM exp_runs WHERE state IN ('queued','starting','running','draining','checkpointing')")["c"]
        if n >= self.max_active:
            raise IllegalRunTransition(f"{n} run(s) already active or queued (cap {self.max_active})")

    @audited("create_run")
    def create(self, req: RunRequest, *, actor: str) -> dict:
        res = self.validate(req)
        self._check_capacity()
        now = self.clock()
        n = (self.db.one("SELECT COALESCE(MAX(n),0)+1 AS n FROM exp_runs") or {"n": 1})["n"]
        rid = f"run-{n:04d}"
        self.db.x("INSERT INTO exp_runs (id, owner, preset, state, backend, resolved, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                  (rid, actor, req.preset, RunState.queued.value, self.backend.name, res.model_dump_json(), now, now))
        self._new_attempt(rid, None, None)
        self.emit("experiment", "controller", "run_queued", rid, {"preset": req.preset, "owner": actor})
        return self.get(rid)

    def _new_attempt(self, rid: str, resume_from: str | None, resume_ckpt: str | None) -> str:
        n = self.db.one("SELECT COUNT(*) c FROM exp_attempts WHERE run_id=?", (rid,))["c"] + 1
        aid = f"a{n}"
        self.db.x("INSERT INTO exp_attempts (run_id, attempt_id, n, resume_from, resume_checkpoint) VALUES (?,?,?,?,?)",
                  (rid, aid, n, resume_from, resume_ckpt))
        return aid

    def _cur(self, rid: str) -> dict:
        return self.db.one("SELECT * FROM exp_attempts WHERE run_id=? ORDER BY n DESC LIMIT 1", (rid,))

    def _set(self, rid: str, dst: RunState, **f) -> None:
        cur = RunState(self.db.one("SELECT state FROM exp_runs WHERE id=?", (rid,))["state"])
        check(cur, dst)
        now = self.clock()
        if dst in TERMINAL:
            f.setdefault("ended_at", now)
        self.db.update("exp_runs", {"id": rid}, state=dst.value, updated_at=now, **f)
        self.emit("experiment", "controller", f"run_{dst.value}", rid, {"from": cur.value, "error": f.get("error")})

    def _stop_cmd(self, rid: str, dst: RunState, call: str) -> dict:
        run = self.get(rid)
        cur = RunState(run["state"])
        if cur is dst:
            return run                         # repeated command: idempotent no-op
        if cur not in (RunState.starting, RunState.running) or not self._cur(rid)["job_id"]:
            raise IllegalRunTransition(f"cannot {call} a run in state {cur.value}")
        getattr(self.backend, call)(self._cur(rid)["job_id"])
        self._set(rid, dst)
        return self.get(rid)

    @audited("drain")
    def drain(self, rid: str, *, actor: str) -> dict:
        return self._stop_cmd(rid, RunState.draining, "drain")

    @audited("checkpoint_and_stop")
    def checkpoint_and_stop(self, rid: str, *, actor: str) -> dict:
        return self._stop_cmd(rid, RunState.checkpointing, "checkpoint_and_stop")

    @audited("force_cancel")
    def cancel(self, rid: str, *, actor: str) -> dict:
        run = self.get(rid)
        cur = RunState(run["state"])
        if cur is RunState.cancelled:
            return run
        job = self._cur(rid)["job_id"]
        if cur in ACTIVE and job:
            self.backend.force_cancel(job)     # may raise: then the run is NOT marked cancelled
        self._set(rid, RunState.cancelled, error="force-cancelled: work since last checkpoint is lost" if cur in ACTIVE else None)
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": self._cur(rid)["attempt_id"]}, ended_at=self.clock())
        return self.get(rid)

    @audited("resume")
    def resume(self, rid: str, checkpoint_id: str | None = None, *, actor: str) -> dict:
        run = self.get(rid)
        if RunState(run["state"]) not in RESUMABLE:
            raise IllegalRunTransition(f"cannot resume a run in state {run['state']}")
        latest = self.checkpoints(rid, latest_only=True)
        if not latest:
            raise ConfigError("no checkpoint to resume from")
        ck = latest[0]
        if checkpoint_id and checkpoint_id != ck["id"]:
            raise ConfigError("the backend resumes from the attempt's latest.pt only; "
                              f"latest checkpoint is {ck['id']} (to branch from another, start a new run)")
        self._check_capacity()
        self._new_attempt(rid, ck["attempt_id"], ck["id"])
        self._set(rid, RunState.queued, error=None, ended_at=None, backend_error=None)
        return self.get(rid)

    # ---------- worker ----------
    def tick(self) -> None:
        with self.mu:
            self._tick()

    def _tick(self) -> None:
        for r in self.db.q("SELECT id, state FROM exp_runs WHERE state IN ('queued','starting') ORDER BY n"):
            self._dispatch(r["id"], r["state"])
        for r in self.db.q("SELECT id FROM exp_runs WHERE state IN ('starting','running','draining','checkpointing') ORDER BY n"):
            self._poll(r["id"])
        self._run_evals()

    def _dispatch(self, rid: str, state: str) -> None:
        att, run = self._cur(rid), self.get(rid)
        if att["job_id"]:
            return
        if state == "queued":
            self._set(rid, RunState.starting, started_at=run["started_at"] or self.clock())
        caps = run["resolved"]["caps"]
        spec = JobSpec(run_id=rid, attempt_id=att["attempt_id"], train_config=run["resolved"]["train_config"],
                       resume_from_attempt=att["resume_from"], timeout_s=caps["timeout_s"], grace_s=caps["grace_s"], cpu=caps["cpu"],
                       memory_mb=caps["memory_mb"], max_containers=caps["max_containers"])
        try:
            job = self.backend.submit(spec)          # idempotent per (run, attempt): safe to repeat after a crash
        except Exception as e:
            self._fail(rid, att, f"submit failed: {redact(e)}")
            return
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": att["attempt_id"]}, job_id=job, started_at=self.clock())
        self.db.update("exp_runs", {"id": rid}, last_heartbeat=self.clock())

    def _fail(self, rid: str, att: dict, msg: str) -> None:
        self._set(rid, RunState.failed, error=msg)
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": att["attempt_id"]}, ended_at=self.clock(), error=msg)

    def _poll(self, rid: str) -> None:
        att, run = self._cur(rid), self.get(rid)
        if not att["job_id"]:
            return
        try:
            st = self.backend.status(att["job_id"])
        except LookupError:
            return self._reconcile_lost(rid, att)
        except Exception as e:
            self.db.update("exp_runs", {"id": rid}, backend_error=redact(e))      # heartbeat deliberately not refreshed
            return
        self.db.update("exp_runs", {"id": rid}, last_heartbeat=self.clock(), backend_error=None)
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": att["attempt_id"]}, job_state=st.state.value)
        self._ingest(rid, att["attempt_id"])
        cur = RunState(run["state"])
        if st.state is JobState.RUNNING and cur is RunState.starting:
            self._set(rid, RunState.running)
        elif st.state.terminal:
            self._finish(rid, att, st)

    def _finish(self, rid: str, att: dict, st) -> None:
        a = att["attempt_id"]
        has_ckpt = any(c["attempt_id"] == a for c in self.checkpoints(rid))
        result = self.reader.result(rid, a)
        if st.state is JobState.DONE:
            if result is None:
                return self._fail(rid, att, "backend reported done but result.json is missing; not marking completed")
            dst, err = RunState.completed, None
        elif st.state in (JobState.DRAINED, JobState.STOPPED):
            dst, err = RunState.stopped, None
        elif st.state is JobState.TIMED_OUT:
            dst, err = (RunState.stopped, "timed out; checkpoint kept") if has_ckpt else (RunState.failed, "timed out with no checkpoint")
        elif st.state is JobState.CANCELLED:
            dst, err = RunState.cancelled, st.detail or None
        else:
            dst, err = RunState.failed, redact(st.detail or "job failed")
        problems = self.reader.problems(rid, a) if result is not None else ["result.json missing"]
        self.db.update("exp_runs", {"id": rid}, artifact_problems=problems)
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": a}, ended_at=self.clock(), result=result, error=err)
        if RunState(self.get(rid)["state"]) is not dst:
            self._set(rid, dst, error=err)

    def _reconcile_lost(self, rid: str, att: dict) -> None:
        """The backend no longer knows this job (e.g. in-memory local backend after a restart). Use only durable files."""
        a = att["attempt_id"]
        self._ingest(rid, a)
        res = self.reader.result(rid, a)
        if res and not self.reader.problems(rid, a):
            dst = {"done": RunState.completed, "drained": RunState.stopped, "stopped": RunState.stopped}.get(res.get("state"), RunState.failed)
            self._set(rid, dst, error=None if dst is not RunState.failed else f"backend lost job; result state {res.get('state')}")
        else:
            self._fail(rid, att, "backend lost the job (restart?) and no verified result.json exists; outcome unknown, not completed")
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": a}, ended_at=self.clock(), result=res)

    # ---------- ingestion ----------
    def _ingest(self, rid: str, a: str) -> None:
        have = self.db.one("SELECT COUNT(*) c FROM exp_metrics WHERE run_id=? AND attempt_id=?", (rid, a))["c"]
        rows = self.reader.metrics(rid, a)[have: have + MAX_INGEST]
        for i, row in enumerate(rows):
            self.db.x("INSERT OR IGNORE INTO exp_metrics VALUES (?,?,?,?,?)", (rid, a, have + i, row.get("env_steps"), json.dumps(row)))
        for row in rows[-MAX_LIVE_EVENTS:]:        # live feed is sampled; the table keeps every row
            self.emit("experiment", f"metrics:{rid}", "training_metric", rid, {"attempt_id": a, **row})
        known = {c["id"] for c in self.checkpoints(rid)}
        for c in self.reader.checkpoints(rid, a):
            m = c["meta"]
            if m["checkpoint_id"] in known:
                continue
            self.db.x("INSERT OR IGNORE INTO exp_checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?)",
                      (m["checkpoint_id"], rid, a, c["name"], c["path"], c["sha256"], c["size"], m.get("training_steps"),
                       m.get("policy_version"), m.get("parent_checkpoint_id"), m.get("role"), json.dumps(m), self.clock()))
            self.emit("experiment", "controller", "checkpoint_written", rid, {"checkpoint_id": m["checkpoint_id"], "attempt_id": a,
                                                                             "env_steps": m.get("training_steps"), "sha256": c["sha256"]})
        seen = self.db.one("SELECT episodes_seen s FROM exp_attempts WHERE run_id=? AND attempt_id=?", (rid, a))["s"]
        new = self.reader.episode_events(rid, a)[seen: seen + MAX_EPISODE_EVENTS]
        self.db.update("exp_attempts", {"run_id": rid, "attempt_id": a}, episodes_seen=seen + len(new))
        for ev in new:
            # world-stream events from training default to evaluator-only visibility; the producer must opt in to more
            self.emit("world", f"train:{rid}", ev.get("event_type", "episode_event"), rid, ev.get("payload", ev),
                      visibility=ev.get("visibility", "evaluator"), episode_id=ev.get("episode_id"))

    # ---------- evaluation ----------
    def _ckpt(self, cid: str) -> dict:
        c = self.db.one("SELECT * FROM exp_checkpoints WHERE id=?", (cid,))
        if c is None:
            raise ConfigError(f"unknown checkpoint {cid}")
        return c

    def compatibility_errors(self, atk: dict, dfn: dict) -> list[str]:
        from ..envs.encoding import obs_schema_hash
        from ..schemas import catalog_hash

        errs = []
        if atk["role"] != "attacker":
            errs.append(f"{atk['id']} is a {atk['role']} checkpoint, expected attacker")
        if dfn["role"] != "defender":
            errs.append(f"{dfn['id']} is a {dfn['role']} checkpoint, expected defender")
        if atk["meta"].get("catalog_hash") != dfn["meta"].get("catalog_hash"):
            errs.append(f"catalog_hash: attacker has {atk['meta'].get('catalog_hash')!r}, "
                        f"defender has {dfn['meta'].get('catalog_hash')!r}")
        for role, checkpoint in (("attacker", atk), ("defender", dfn)):
            if checkpoint["role"] == role:
                expected = obs_schema_hash(role)
                if checkpoint["meta"].get("observation_schema_hash") != expected:
                    errs.append(f"{checkpoint['id']} observation_schema_hash is incompatible with current {role} schema")
                if checkpoint["meta"].get("catalog_hash") != catalog_hash():
                    errs.append(f"{checkpoint['id']} catalog_hash is incompatible with the current action catalog")
        for c in (atk, dfn):
            if not c["available"]:
                errs.append(f"{c['id']} artifacts were cleaned up")
        return errs

    @audited("submit_eval")
    def submit_eval(self, req: EvalRequest, *, actor: str) -> dict:
        errs = []
        if req.suite not in SUITES:
            errs.append(f"suite must be one of {sorted(SUITES)}")
        if not 1 <= req.episodes <= MAX_EVAL_EPISODES:
            errs.append(f"episodes must be 1..{MAX_EVAL_EPISODES}")
        if self.evaluator is None:
            errs.append("no evaluation backend configured")
        atk, dfn = self._ckpt(req.attacker_checkpoint_id), self._ckpt(req.defender_checkpoint_id)
        errs += self.compatibility_errors(atk, dfn)
        if errs:
            raise ConfigError(errs)
        spec = {**req.model_dump(), "frozen": True, "attacker_sha256": atk["sha256"], "defender_sha256": dfn["sha256"]}
        return self._add_eval(req.suite, False, spec, actor, atk["run_id"])

    @audited("debug_fork")
    def debug_fork(self, rid: str, checkpoint_id: str, tick: int, seed: int = 0, *, actor: str) -> dict:
        c = self._ckpt(checkpoint_id)
        if c["run_id"] != rid:
            raise ConfigError("checkpoint does not belong to this run")
        if self.evaluator is None:
            raise ConfigError("no evaluation backend configured")
        spec = {"suite": "debug_fork", "checkpoint_id": checkpoint_id, "sha256": c["sha256"], "tick": tick, "seed": seed, "episodes": 1,
                "frozen": True, "attacker_checkpoint_id": checkpoint_id, "defender_checkpoint_id": checkpoint_id,
                "attacker_sha256": c["sha256"], "defender_sha256": c["sha256"]}
        return self._add_eval("debug_fork", True, spec, actor, rid)

    def _add_eval(self, suite, debug, spec, actor, run_id) -> dict:
        eid = f"{'dbg' if debug else 'eval'}-{self.db.one('SELECT COUNT(*)+1 c FROM exp_evals')['c']:04d}"
        now = self.clock()
        self.db.x("INSERT INTO exp_evals VALUES (?,?,?,?,?,?,NULL,NULL,?,?,?)",
                  (eid, run_id, suite, int(debug), "queued", json.dumps(spec), actor, now, now))
        return self.get_eval(eid)

    def get_eval(self, eid: str) -> dict:
        e = self.db.one("SELECT * FROM exp_evals WHERE id=?", (eid,))
        if e is None:
            raise KeyError(eid)
        return {**e, "debug": bool(e["debug"]), "label": "DEBUG" if e["debug"] else None}

    def evals(self, *, debug: bool = False) -> list[dict]:
        """Benchmark queries never see debug forks; they are only returned when asked for explicitly."""
        return [{**e, "debug": bool(e["debug"]), "label": "DEBUG" if e["debug"] else None}
                for e in self.db.q("SELECT * FROM exp_evals WHERE debug=? ORDER BY created_at DESC, id DESC", (int(debug),))]

    def _run_evals(self) -> None:
        for e in self.db.q("SELECT * FROM exp_evals WHERE state='queued' ORDER BY created_at, id"):
            spec, eid = e["spec"], e["id"]
            self.db.update("exp_evals", {"id": eid}, state="running", updated_at=self.clock())
            try:
                paths, stale = {}, []
                for role in ("attacker", "defender"):
                    c = self._ckpt(spec[f"{role}_checkpoint_id"])
                    p = self.reader.checkpoint_path(c["run_id"], c["attempt_id"], c["name"])
                    if not p.exists() or c["sha256"] != spec[f"{role}_sha256"]:
                        stale.append(f"{role} checkpoint {c['id']} missing or changed since submission")
                    paths[role] = str(p)
                if stale:
                    raise ConfigError(stale)
                out = f"{self.episodes_dir}/{eid}" if self.episodes_dir else ""
                result = self.evaluator.run(spec, paths, out)
                self.db.update("exp_evals", {"id": eid}, state="done", result={**result, "evaluator": self.evaluator.name},
                               updated_at=self.clock())
                self.on_new_episodes()
            except Exception as ex:
                self.db.update("exp_evals", {"id": eid}, state="failed", error=redact(ex), updated_at=self.clock())
            self.emit("experiment", "controller", "evaluation_finished", e["run_id"], {"eval_id": eid})

    def overview(self) -> dict:
        runs = self.list()
        by = {}
        for r in runs:
            by[r["state"]] = by.get(r["state"], 0) + 1
        return {"runs": {"configured": True, "count": len(runs), "by_state": by, "stale": [r["id"] for r in runs if r["stale"]]},
                "infrastructure": {"configured": True, "backend": self.backend.name}}

    def latest_evaluation(self) -> dict | None:
        done = [e for e in self.evals() if e["state"] == "done"]
        return {"id": done[0]["id"], "suite": done[0]["suite"], "result": done[0]["result"]} if done else None

    # ---------- infrastructure ----------
    def infrastructure(self) -> dict:
        runs = self.list()
        jobs = [{"run_id": r["id"], "attempt_id": a["attempt_id"], "job_id": a["job_id"], "job_state": a["job_state"],
                 "run_state": r["state"], "heartbeat_age_s": r["heartbeat_age_s"], "stale": r["stale"],
                 "backend_error": r["backend_error"], "error": a["error"], "owner": r["owner"], "cleaned": bool(a["cleaned_at"])}
                for r in runs for a in r["attempts"]]
        pv = [{"checkpoint_id": c["id"], "run_id": c["run_id"], "policy_version": c["policy_version"], "role": c["role"]}
              for c in self.checkpoints()]
        m = self._measured()
        return {"configured": True, "backend": self.backend.name, "jobs": jobs, "failures": [j for j in jobs if j["error"] or j["backend_error"]],
                "policy_versions": pv, "orphans": self.orphans(),
                "estimate": {"basis": "measured" if m else "unavailable", "steps_per_sec": m[0] if m else None,
                             "measured_at": m[1] if m else None, "age_s": round(self.clock() - m[1], 1) if m else None},
                "evaluator": getattr(self.evaluator, "name", None)}

    def orphans(self) -> list[dict]:
        known = {(a["run_id"], a["attempt_id"]) for a in self.db.q("SELECT run_id, attempt_id FROM exp_attempts")}
        return [{"run_id": r, "attempt_id": a} for r, a in self.reader.attempts() if (r, a) not in known]

    @audited("cleanup")
    def cleanup(self, *, dry_run: bool = True, min_age_s: float = 0.0, actor: str) -> dict:
        """Deletes only attempts this controller owns, whose run is terminal and whose manifest verifies. Never deletes the
        newest checkpoint of a resumable run, anything an unfinished evaluation reads, or orphans (reported only)."""
        busy = {c for e in self.db.q("SELECT spec FROM exp_evals WHERE state IN ('queued','running')")
                for c in (e["spec"]["attacker_checkpoint_id"], e["spec"]["defender_checkpoint_id"])}
        plan = []
        for r in self.list():
            if r["state"] not in {s.value for s in TERMINAL} or self.clock() - r["updated_at"] < min_age_s:
                continue
            keep = self.checkpoints(r["id"], latest_only=True)[0]["attempt_id"] if r["resumable"] else None
            for a in r["attempts"]:
                aid, why = a["attempt_id"], None
                if a["cleaned_at"]:
                    continue
                if aid == keep:
                    why = "holds the newest checkpoint of a resumable run"
                elif any(c["attempt_id"] == aid and c["id"] in busy for c in self.checkpoints(r["id"])):
                    why = "an unfinished evaluation reads it"
                elif self.reader.problems(r["id"], aid):
                    why = "manifest does not verify; not deleting unverified data"
                plan.append({"run_id": r["id"], "attempt_id": aid, "action": "skip" if why else "delete", "reason": why})
        if not dry_run:
            for p in plan:
                if p["action"] == "delete":
                    self.reader.delete_attempt(p["run_id"], p["attempt_id"])
                    self.db.update("exp_attempts", {"run_id": p["run_id"], "attempt_id": p["attempt_id"]}, cleaned_at=self.clock())
                    self.db.x("UPDATE exp_checkpoints SET available=0 WHERE run_id=? AND attempt_id=?", (p["run_id"], p["attempt_id"]))
        return {"dry_run": dry_run, "plan": plan, "orphans_reported_not_deleted": self.orphans()}
