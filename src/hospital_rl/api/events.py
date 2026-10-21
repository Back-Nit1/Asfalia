"""SSE over the event store: cursor ids, Last-Event-ID resume, resync after retention, bounded batches."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Callable

from fastapi import Request

from .storage import Store

BATCH = 200
KEEPALIVE_S = 15.0


def frame(event: str, data: object, id_: int | None = None) -> str:
    head = f"id: {id_}\n" if id_ is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data)}\n\n"


def start_cursor(request: Request, after: int | None) -> int:
    """Last-Event-ID (browser reconnect) beats ?after."""
    h = request.headers.get("last-event-id", "")
    return int(h) if h.isdigit() else (after or 0)


async def stream(request: Request, store: Store, cursor: int, *, follow: bool, interval: float,
                 source_kinds: tuple[str, ...] | None, visibilities: tuple[str, ...] | None,
                 snapshot: Callable[[], dict], task_id: str | None = None) -> AsyncIterator[str]:
    yield "retry: 1000\n\n"
    if cursor < store.pruned_through():     # events the client missed are gone: send current state instead
        snap = snapshot()
        cursor = snap["cursor"]
        yield frame("resync", snap, cursor)
    idle = 0.0
    while not await request.is_disconnected():
        rows = await asyncio.to_thread(store.events_after, cursor, limit=BATCH, source_kinds=source_kinds,
                                       visibilities=visibilities, task_id=task_id)
        for cur, ev in rows:
            cursor = cur
            yield frame("envelope", json.loads(ev.model_dump_json()), cur)
        if len(rows) == BATCH:
            continue
        if not follow:
            return
        idle += interval
        if rows:
            idle = 0.0
        elif idle >= KEEPALIVE_S:
            idle = 0.0
            yield ": keepalive\n\n"
        await asyncio.sleep(interval)
