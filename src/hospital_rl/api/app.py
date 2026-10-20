from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..orchestration.scheduler import Scheduler
from .auth import AuthConfig
from .replay import Library
from .routes import build_router
from .storage import Store

FIXTURES = Path(__file__).parent / "fixtures"


def create_app(store: Store, scheduler: Scheduler, auth: AuthConfig | None = None, library: Library | None = None,
               stream_interval: float = 0.25, static_dir: str | Path | None = None, run_scheduler: bool = False,
               tick_seconds: float = 2.0, retention: int = 100_000) -> FastAPI:
    app = FastAPI(title="hospital-rl dashboard API")
    app.state.store, app.state.scheduler = store, scheduler
    app.state.auth = auth or AuthConfig()
    app.state.library = library or Library(FIXTURES)
    app.state.stream_interval = stream_interval
    app.include_router(build_router())

    if run_scheduler:
        stop = threading.Event()

        def loop():
            while not stop.wait(tick_seconds):
                scheduler.tick()
                store.prune_events(retention)

        @app.on_event("startup")
        def _start():
            threading.Thread(target=loop, daemon=True).start()

        @app.on_event("shutdown")
        def _stop():
            stop.set()

    if static_dir and Path(static_dir).is_dir():
        root = Path(static_dir)
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = root / path
            return FileResponse(f if path and f.is_file() else root / "index.html")
    return app
