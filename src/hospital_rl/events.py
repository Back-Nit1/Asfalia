"""Versioned event contract shared by dev agents, experiment controller, worlds, and infra."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EVENT_SCHEMA_VERSION = 1

SourceKind = Literal["development", "experiment", "world", "infrastructure"]
Visibility = Literal["attacker", "defender", "evaluator", "operator"]


class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_kind: SourceKind
    event_type: str
    run_id: str | None = None
    task_id: str | None = None
    episode_id: str | None = None
    worker_id: str
    sequence: int  # monotonic per (source_kind, worker_id)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tick: int | None = None
    visibility: Visibility = "operator"
    schema_version: int = EVENT_SCHEMA_VERSION
    payload: dict[str, Any] = Field(default_factory=dict)
