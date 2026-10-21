from __future__ import annotations

import logging
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
               tick_seconds: float = 2.0, retention: int = 100_000, controller=None) -> FastAPI:
    app = FastAPI(title="hospital-rl dashboard API")
    app.state.store, app.state.scheduler = store, scheduler
    app.state.auth = auth or AuthConfig()
    app.state.library = library or Library(FIXTURES)
    app.state.stream_interval = stream_interval
    app.state.controller = controller
    app.include_router(build_router(controller is not None))

    if controller is not None:
        controller.on_new_episodes = app.state.library.rescan
    if run_scheduler or controller is not None:
        stop = threading.Event()

        def loop():
            while not stop.wait(tick_seconds):
                try:
                    if run_scheduler:
                        scheduler.tick()
                    if controller is not None:
                        controller.tick()
                    store.prune_events(retention)
                except Exception:       # one bad tick must not stop the control loop; state stays as last written
                    logging.getLogger("hospital_rl.api").exception("control loop tick failed")

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
