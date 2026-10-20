"""Development-plane event emission and output redaction."""
from __future__ import annotations

import os
import re
import threading
from typing import Any

from ..api.storage import Store
from ..events import EventEnvelope

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)
_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{16,}", r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}", r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+")]


def secret_values() -> list[str]:
    return [v for k, v in os.environ.items() if _SECRET_ENV.search(k) and len(v) >= 8]


def redact(text: str) -> str:
    for v in secret_values():
        text = text.replace(v, "[REDACTED]")
    for p in _PATTERNS:
        text = p.sub("[REDACTED]", text)
    return text


def tail(text: str, n: int = 4000) -> str:
    return redact(text[-n:])


def clean_env() -> dict[str, str]:
    """Environment for child processes: provider keys and tokens never leak into acceptance commands."""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}


class Emitter:
    """Appends `development` events with persisted per-worker monotonic sequences."""

    def __init__(self, store: Store, worker_id: str = "orchestrator"):
        self.store, self.worker_id, self._lock = store, worker_id, threading.Lock()

    def emit(self, event_type: str, task_id: str | None = None, **payload: Any) -> EventEnvelope:
        with self._lock:
            ev = EventEnvelope(source_kind="development", event_type=event_type, task_id=task_id,
                               worker_id=self.worker_id, visibility="operator", payload=payload,
                               sequence=self.store.next_sequence("development", self.worker_id))
            self.store.append_event(ev)
        return ev
