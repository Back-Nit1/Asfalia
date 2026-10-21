"""REST + SSE routes. Operator routes cover the development plane; `/api/world/<perspective>/...` are the only routes a
policy-facing consumer may use, and they never read development events or tasks."""
from __future__ import annotations

import json
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..events import EVENT_SCHEMA_VERSION
from ..orchestration import integration as git_ops
from ..orchestration.tasks import IllegalTransition, Status
from . import dto
from .auth import require_evaluator, require_operator
from ..experiments.controller import ConfigError
from .events import start_cursor, stream
from .replay import ALLOWED

NOT_CONFIGURED = {"configured": False, "state": "not_configured"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def idempotent(request: Request, fn: Callable[[], object]) -> JSONResponse:
    """Run a mutation once per (Idempotency-Key, method+path); a repeat replays the stored response, refusals included."""
    key = request.headers.get("idempotency-key")
    if not key:
        raise HTTPException(400, "Idempotency-Key header required")
    scope = f"{request.method} {request.url.path}"
    store = request.app.state.store
    prev = store.idem_get(key, scope)
    if prev:
        return JSONResponse(json.loads(prev["response"]), prev["status"], headers={"Idempotent-Replay": "true"})
    try:
        out, status = fn(), 200
    except KeyError as e:
        out, status = {"detail": f"not found: {e.args[0]}"}, 404
    except ConfigError as e:
        out, status = {"detail": str(e), "errors": e.errors}, 422
    except (IllegalTransition, ValueError) as e:
        out, status = {"detail": str(e)}, 409
    store.idem_put(key, scope, status, out)
    return JSONResponse(out, status)


def task_json(t) -> dict:
    return t.model_dump(mode="json")


def build_router(controller: bool = False) -> APIRouter:
    r = APIRouter(prefix="/api")
    op = APIRouter(dependencies=[Depends(require_operator)])

    @r.get("/health")
    def health(request: Request):
        return {"ok": True, "schema_version": EVENT_SCHEMA_VERSION, "auth_required": request.app.state.auth.require_operator}

    # ---- development plane (operator) ----
    def dev_snapshot(request: Request) -> dict:
        s = request.app.state
        return {"cursor": s.store.latest_cursor(), "tasks": [task_json(t) for t in s.scheduler.all()],
                "runtime": s.scheduler.runtime_status()}

    @op.get("/snapshot")
    def snapshot(request: Request):
        return dev_snapshot(request)

    @op.get("/overview")
    def overview(request: Request):
        s = request.app.state
        tasks = s.scheduler.all()
        counts = {st.value: sum(t.status is st for t in tasks) for st in Status}
        return {"tasks": counts, "task_total": len(tasks), "runtime": s.scheduler.runtime_status(),
                **(request.app.state.controller.overview() if controller else {"runs": {**NOT_CONFIGURED, "count": 0}, "infrastructure": NOT_CONFIGURED}),
                "stale_running": [t.id for t in tasks if t.status is Status.running and t.lease_expiry and t.lease_expiry < s.scheduler.clock()],
                "latest_evaluation": None, "step_rate": None, "spend_estimate": None} | (
            {"latest_evaluation": request.app.state.controller.latest_evaluation()} if controller else {})

    @op.get("/tasks")
    def tasks(request: Request):
        sch = request.app.state.scheduler
        all_ = sch.all()
        return {"runtime": sch.runtime_status(), "now": sch.clock(),
                "tasks": [{**task_json(t), "path_conflicts": sch.path_conflicts(t, all_)} for t in all_]}

    @op.get("/tasks/{task_id}")
    def task(task_id: str, request: Request):
        try:
            return task_json(request.app.state.scheduler.get(task_id))
        except KeyError:
            raise HTTPException(404, "task not found")

    @op.get("/tasks/{task_id}/events")
    def task_events(task_id: str, request: Request, after: int = 0, limit: int = Query(200, le=1000)):
        rows = request.app.state.store.events_after(after, limit=limit, source_kinds=("development",), task_id=task_id)
        return {"events": [{"cursor": c, **json.loads(e.model_dump_json())} for c, e in rows]}

    @op.get("/tasks/{task_id}/diff")
    def task_diff(task_id: str, request: Request):
        try:
            t = request.app.state.scheduler.get(task_id)
        except KeyError:
            raise HTTPException(404, "task not found")
        if not t.worktree:
            return {"available": False, "reason": "no worktree yet", "diff": ""}
        return {"available": True, "diff": git_ops.diff(t)}

    for verb in ("cancel", "retry", "integrate", "claim", "submit", "heartbeat"):
        def mutate(task_id: str, request: Request, verb=verb):
            sch = request.app.state.scheduler
            fn = {"submit": sch.submit_for_review}.get(verb) or getattr(sch, verb)
            return idempotent(request, lambda: task_json(fn(task_id)))
        op.post(f"/tasks/{{task_id}}/{verb}")(mutate)

    @op.get("/artifacts")
    def artifacts(request: Request, task_id: str | None = None):
        return {"artifacts": request.app.state.store.list_artifacts(task_id)}

    @op.get("/events")
    def events(request: Request, after: int = 0, limit: int = Query(200, le=1000)):
        rows = request.app.state.store.events_after(after, limit=limit, visibilities=("operator",), source_kinds=("development",))
        return {"events": [{"cursor": c, **json.loads(e.model_dump_json())} for c, e in rows]}

    @op.get("/events/stream")
    async def event_stream(request: Request, after: int | None = None, follow: bool = True, task_id: str | None = None):
        s = request.app.state
        gen = stream(request, s.store, start_cursor(request, after), follow=follow, interval=s.stream_interval,
                     source_kinds=("development",), visibilities=("operator",), snapshot=lambda: dev_snapshot(request), task_id=task_id)
        return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)

    if controller:
        from .experiment_routes import mount
        mount(op, idempotent, NOT_CONFIGURED)
    else:
        # ---- honest not-configured surfaces ----
        @op.get("/runs")
        def runs(request: Request):
            rows = request.app.state.store.list_runs()
            return {**NOT_CONFIGURED, "runs": rows, "message": "no experiment controller is connected; no runs exist"} if not rows else {"configured": True, "runs": rows}

        @op.post("/runs")
        @op.post("/runs/{run_id}/cancel")
        @op.post("/runs/{run_id}/resume")
        def runs_mutate(request: Request, run_id: str | None = None):
            return JSONResponse({**NOT_CONFIGURED, "message": "experiment controller not configured"}, 501)

        @op.get("/runs/{run_id}")
        def run(run_id: str):
            raise HTTPException(404, "run not found (no experiment controller configured)")

        @op.get("/checkpoints")
        def checkpoints():
            return {**NOT_CONFIGURED, "checkpoints": [], "message": "no training producer connected"}

        @op.get("/metrics")
        def metrics(request: Request):
            tasks = request.app.state.scheduler.all()
            return {"tasks_by_status": {st.value: sum(t.status is st for t in tasks) for st in Status},
                    "training": NOT_CONFIGURED, "step_rate": None}

        @op.get("/infrastructure")
        def infrastructure():
            return {**NOT_CONFIGURED, "workers": [], "sandboxes": [], "message": "no Modal/Daytona backend configured"}

        @op.post("/infrastructure/drain")
        @op.post("/infrastructure/cleanup")
        def infra_mutate():
            return JSONResponse({**NOT_CONFIGURED, "message": "no infrastructure backend configured"}, 501)

    @op.get("/world/episodes")
    def episodes(request: Request):
        lib = request.app.state.library
        lib.rescan()
        return {"episodes": [{"id": e.id, "label": e.label, "scenario": e.scenario, "seed": e.seed,
                              "ticks": len(e.steps), "topology_exported": e.has_topology} for e in lib.episodes.values()],
                "live": {**NOT_CONFIGURED, "message": "no live episode producer connected"}}

    r.include_router(op)
    for persp in ("attacker", "defender", "evaluator"):
        r.include_router(_perspective_router(persp))
    return r


def _perspective_router(persp: str) -> APIRouter:
    privileged = persp == "evaluator"
    dep = Depends(require_evaluator if privileged else require_operator)
    pr = APIRouter(prefix=f"/world/{persp}", dependencies=[dep])
    snap_model, replay_model = dto.SNAPSHOT[persp], dto.REPLAY[persp]

    def episode(request: Request, episode_id: str):
        ep = request.app.state.library.episodes.get(episode_id)
        if ep is None:
            raise HTTPException(404, "episode not found")
        return ep, request.app.state.library.frames(ep, persp)

    def extras(ep) -> dict:
        out = {}
        if persp != "attacker":     # the attacker only learns edges by exploring; the export carries no such knowledge
            out["edges"] = ep.edges
        if privileged:
            out["final_metrics"] = ep.final_metrics
        return out

    @pr.get("/episodes/{episode_id}/snapshot", response_model=snap_model)
    def snapshot(request: Request, episode_id: str, tick: int | None = None):
        ep, frames = episode(request, episode_id)
        if not frames:
            raise HTTPException(404, "episode has no steps")
        t = len(frames) - 1 if tick is None else max(0, min(tick, len(frames) - 1))
        f = frames[t]
        return {"episode_id": ep.id, "label": ep.label, "mode": "REPLAY", "scenario": ep.scenario, "tick": f["tick"],
                "last_tick": frames[-1]["tick"], "nodes": f["nodes"], "feed": [e for fr in frames[:t + 1] for e in fr["feed"]],
                "reward": f["reward"], "cumulative": f["cumulative"], "topology_exported": ep.has_topology, **extras(ep)}

    @pr.get("/episodes/{episode_id}/replay", response_model=replay_model)
    def replay(request: Request, episode_id: str):
        # ponytail: whole episode in one response; add from/to windows when real episodes get large
        ep, frames = episode(request, episode_id)
        return {"episode_id": ep.id, "label": ep.label, "scenario": ep.scenario, "topology_exported": ep.has_topology,
                "frames": frames, **extras(ep)}

    @pr.get("/episodes/{episode_id}/events", response_model=dto.EventsPage)
    def episode_events(request: Request, episode_id: str, after: int = -1, limit: int = Query(200, le=1000)):
        _, frames = episode(request, episode_id)
        feed = [e for fr in frames for e in fr["feed"] if e["idx"] > after][:limit]
        return {"perspective": persp, "privileged": privileged, "cursor": feed[-1]["idx"] if feed else after, "events": feed}

    @pr.get("/stream")
    async def world_stream(request: Request, after: int | None = None, follow: bool = True):
        """World-source events only, filtered by visibility in the query. Development events are unreachable here."""
        s = request.app.state
        gen = stream(request, s.store, start_cursor(request, after), follow=follow, interval=s.stream_interval,
                     source_kinds=("world",), visibilities=tuple(sorted(ALLOWED[persp])),
                     snapshot=lambda: {"cursor": s.store.latest_cursor(), "perspective": persp})
        return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)

    return pr
