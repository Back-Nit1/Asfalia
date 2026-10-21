"""Operator-plane routes for the experiment controller. Everything here sits behind `require_operator`; the evaluator token
is not accepted. Mutations require an Idempotency-Key and are audited by the controller. Experiment events are source_kind
"experiment" and are unreachable from the perspective routes, which read world-source events only."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..experiments.config import MAX_CPU, MAX_MEMORY_MB, MAX_STEPS, MAX_TIMEOUT_S, PRESETS, RunRequest
from ..experiments.controller import MAX_EVAL_EPISODES, SUITES, ConfigError, EvalRequest
from .events import start_cursor, stream

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MIN_LIVE_INTERVAL = 0.2          # live feed is capped at 5 Hz no matter what the client asks


def mount(op: APIRouter, idempotent, not_configured: dict) -> None:
    def ctl(request: Request):
        return request.app.state.controller

    def actor(request: Request) -> str:
        return (request.headers.get("x-actor") or "operator")[:64]

    def run_or_404(request: Request, run_id: str) -> dict:
        try:
            return ctl(request).get(run_id)
        except KeyError:
            raise HTTPException(404, "run not found")

    @op.get("/experiments/config")
    def config(request: Request):
        c = ctl(request)
        return {"configured": True, "backend": c.backend.name, "evaluator": getattr(c.evaluator, "name", None),
                "presets": PRESETS, "ceilings": {"total_steps": MAX_STEPS, "timeout_s": MAX_TIMEOUT_S, "cpu": MAX_CPU,
                                                 "memory_mb": MAX_MEMORY_MB},
                "suites": sorted(SUITES), "max_eval_episodes": MAX_EVAL_EPISODES, "stale_after_s": c.stale_after}

    @op.post("/runs/validate")
    def validate(req: RunRequest, request: Request):
        try:
            return ctl(request).validate(req)
        except ConfigError as e:
            return JSONResponse({"detail": str(e), "errors": e.errors}, 422)

    @op.get("/runs")
    def runs(request: Request):
        return {"configured": True, "backend": ctl(request).backend.name, "runs": ctl(request).list()}

    @op.post("/runs")
    def create(req: RunRequest, request: Request):
        return idempotent(request, lambda: ctl(request).create(req, actor=actor(request)))

    @op.get("/runs/{run_id}")
    def run(run_id: str, request: Request):
        return run_or_404(request, run_id)

    @op.get("/runs/{run_id}/metrics")
    def metrics(run_id: str, request: Request, after: int = -1, limit: int = Query(500, le=2000)):
        run_or_404(request, run_id)
        return {"run_id": run_id, "metrics": ctl(request).metrics(run_id, after, limit)}

    @op.get("/runs/{run_id}/checkpoints")
    def run_checkpoints(run_id: str, request: Request):
        run_or_404(request, run_id)
        return {"configured": True, "checkpoints": ctl(request).checkpoints(run_id)}

    @op.get("/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str, request: Request):
        run_or_404(request, run_id)
        return {"run_id": run_id, "attempts": ctl(request).artifacts(run_id)}

    for verb, method in (("drain", "drain"), ("checkpoint-stop", "checkpoint_and_stop"), ("cancel", "cancel")):
        def mutate(run_id: str, request: Request, method=method):
            return idempotent(request, lambda: getattr(ctl(request), method)(run_id, actor=actor(request)))
        op.post(f"/runs/{{run_id}}/{verb}")(mutate)

    @op.post("/runs/{run_id}/resume")
    def resume(run_id: str, request: Request, checkpoint_id: str | None = None):
        return idempotent(request, lambda: ctl(request).resume(run_id, checkpoint_id, actor=actor(request)))

    @op.post("/runs/{run_id}/debug-fork")
    def fork(run_id: str, request: Request, checkpoint_id: str, tick: int = Query(0, ge=0, le=10_000), seed: int = 0):
        return idempotent(request, lambda: ctl(request).debug_fork(run_id, checkpoint_id, tick, seed, actor=actor(request)))

    @op.get("/checkpoints")
    def checkpoints(request: Request):
        return {"configured": True, "checkpoints": ctl(request).checkpoints()}

    @op.get("/metrics")
    def all_metrics(request: Request):
        c = ctl(request)
        rows = [{"run_id": r["id"], "state": r["state"], "latest": r["latest_metric"]} for r in c.list()]
        return {"training": {"configured": True, "runs": rows}, "step_rate": (c.infrastructure()["estimate"]["steps_per_sec"])}

    @op.post("/evaluations")
    def submit_eval(req: EvalRequest, request: Request):
        return idempotent(request, lambda: ctl(request).submit_eval(req, actor=actor(request)))

    @op.get("/evaluations")
    def evals(request: Request):
        return {"evaluations": ctl(request).evals()}

    @op.get("/evaluations/{eval_id}")
    def get_eval(eval_id: str, request: Request):
        try:
            return ctl(request).get_eval(eval_id)
        except KeyError:
            raise HTTPException(404, "evaluation not found")

    @op.get("/experiments/results")
    def results(request: Request, suite: str | None = None):
        """Stored, finished, non-debug results only. Nothing is computed or filled in here."""
        rows = [e for e in ctl(request).evals() if e["state"] == "done" and (suite is None or e["suite"] == suite)]
        return {"results": rows, "evaluator": getattr(ctl(request).evaluator, "name", None)}

    @op.get("/debug-forks")
    def debug_forks(request: Request):
        return {"label": "DEBUG", "forks": ctl(request).evals(debug=True)}

    @op.get("/infrastructure")
    def infrastructure(request: Request):
        return ctl(request).infrastructure()

    @op.post("/infrastructure/drain")
    def drain_all(request: Request):
        def go():
            c = ctl(request)
            ids = [r["id"] for r in c.list() if r["state"] in ("starting", "running")]
            return {"drained": [c.drain(i, actor=actor(request))["id"] for i in ids]}
        return idempotent(request, go)

    @op.post("/infrastructure/cleanup")
    def cleanup(request: Request, dry_run: bool = True, min_age_s: float = Query(0, ge=0)):
        return idempotent(request, lambda: ctl(request).cleanup(dry_run=dry_run, min_age_s=min_age_s, actor=actor(request)))

    @op.get("/audit")
    def audit(request: Request, limit: int = Query(100, le=500)):
        return {"audit": ctl(request).audit_log(limit)}

    @op.get("/experiments/stream")
    async def exp_stream(request: Request, after: int | None = None, follow: bool = True, run_id: str | None = None):
        """Sampled, read-only live feed of experiment events. Rate-capped; training never waits on a reader."""
        s = request.app.state
        gen = stream(request, s.store, start_cursor(request, after), follow=follow,
                     interval=max(s.stream_interval, MIN_LIVE_INTERVAL), source_kinds=("experiment",), visibilities=("operator",),
                     snapshot=lambda: {"cursor": s.store.latest_cursor(), "runs": s.controller.list()}, run_id=run_id)
        return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)
