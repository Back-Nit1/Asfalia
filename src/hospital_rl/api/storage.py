"""SQLite (WAL) store for tasks, runs, events, artifacts index and idempotency keys.

Single writer process: one connection guarded by a lock. Event reads use an ingestion cursor
(`events.cursor`, arrival order). Order is only guaranteed per (source_kind, worker_id) via `sequence`;
there is no global ordering across workers.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..events import EventEnvelope

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  cursor INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  source_kind TEXT NOT NULL, worker_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL, visibility TEXT NOT NULL,
  run_id TEXT, task_id TEXT, episode_id TEXT,
  data TEXT NOT NULL,
  UNIQUE (source_kind, worker_id, sequence));
CREATE INDEX IF NOT EXISTS events_task ON events (task_id, cursor);
CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, task_id TEXT, run_id TEXT, path TEXT NOT NULL,
  sha256 TEXT, created_at TEXT NOT NULL, meta TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
  key TEXT NOT NULL, scope TEXT NOT NULL,
  status INTEGER NOT NULL, response TEXT NOT NULL, PRIMARY KEY (key, scope));
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # ---- tasks / runs: opaque JSON documents keyed by id, status mirrored for queries ----
    def put_task(self, task_id: str, status: str, data: dict) -> None:
        self._q("INSERT INTO tasks VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, data=excluded.data",
                (task_id, status, json.dumps(data)))

    def get_task(self, task_id: str) -> dict | None:
        r = self._q("SELECT data FROM tasks WHERE id=?", (task_id,))
        return json.loads(r[0]["data"]) if r else None

    def list_tasks(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._q("SELECT data FROM tasks ORDER BY rowid")]

    def put_run(self, run_id: str, status: str, data: dict) -> None:
        self._q("INSERT INTO runs VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, data=excluded.data",
                (run_id, status, json.dumps(data)))

    def list_runs(self) -> list[dict]:
        return [json.loads(r["data"]) for r in self._q("SELECT data FROM runs ORDER BY rowid")]

    # ---- events ----
    def append_event(self, ev: EventEnvelope) -> int | None:
        """Insert; returns the cursor, or None when `event_id` was already stored (deduped).
        A different event_id reusing a stored (source, worker, sequence) is rejected with ValueError."""
        with self._lock:
            try:
                cur = self._db.execute(
                    "INSERT INTO events (event_id, source_kind, worker_id, sequence, event_type, visibility,"
                    " run_id, task_id, episode_id, data) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ev.event_id, ev.source_kind, ev.worker_id, ev.sequence, ev.event_type, ev.visibility,
                     ev.run_id, ev.task_id, ev.episode_id, ev.model_dump_json()))
            except sqlite3.IntegrityError as e:
                if self._db.execute("SELECT 1 FROM events WHERE event_id=?", (ev.event_id,)).fetchone():
                    return None
                raise ValueError(f"sequence {ev.sequence} already used by {ev.source_kind}/{ev.worker_id}") from e
            return cur.lastrowid

    def next_sequence(self, source_kind: str, worker_id: str) -> int:
        r = self._q("SELECT COALESCE(MAX(sequence), -1) + 1 AS n FROM events WHERE source_kind=? AND worker_id=?",
                    (source_kind, worker_id))
        return r[0]["n"]

    def events_after(self, cursor: int = 0, *, limit: int = 500, source_kinds: tuple[str, ...] | None = None,
                    visibilities: tuple[str, ...] | None = None, task_id: str | None = None,
                    episode_id: str | None = None) -> list[tuple[int, EventEnvelope]]:
        where, args = ["cursor > ?"], [cursor]
        for col, vals in (("source_kind", source_kinds), ("visibility", visibilities)):
            if vals is not None:
                where.append(f"{col} IN ({','.join('?' * len(vals))})")
                args += list(vals)
        for col, val in (("task_id", task_id), ("episode_id", episode_id)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        rows = self._q(f"SELECT cursor, data FROM events WHERE {' AND '.join(where)} ORDER BY cursor LIMIT ?",
                       (*args, limit))
        return [(r["cursor"], EventEnvelope.model_validate_json(r["data"])) for r in rows]

    def stream_events(self, source_kind: str, worker_id: str) -> list[EventEnvelope]:
        """One source's events in sequence order (arrival order may differ)."""
        rows = self._q("SELECT data FROM events WHERE source_kind=? AND worker_id=? ORDER BY sequence",
                       (source_kind, worker_id))
        return [EventEnvelope.model_validate_json(r["data"]) for r in rows]

    def latest_cursor(self) -> int:
        return self._q("SELECT COALESCE(MAX(cursor), 0) AS c FROM events")[0]["c"]

    def oldest_cursor(self) -> int:
        """Smallest retained cursor; 0 when empty."""
        return self._q("SELECT COALESCE(MIN(cursor), 0) AS c FROM events")[0]["c"]

    def prune_events(self, keep: int) -> int:
        """Bounded retention: keep the newest `keep` events. Readers whose cursor fell behind must resync."""
        with self._lock:
            floor = self.latest_cursor() - keep
            if floor <= 0:
                return 0
            n = self._db.execute("DELETE FROM events WHERE cursor <= ?", (floor,)).rowcount
            self._db.execute("INSERT INTO meta VALUES ('pruned_through', ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                             (str(floor),))
            return n

    def pruned_through(self) -> int:
        r = self._q("SELECT v FROM meta WHERE k='pruned_through'")
        return int(r[0]["v"]) if r else 0

    # ---- artifacts ----
    def add_artifact(self, art: dict[str, Any]) -> None:
        self._q("INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,?,?,?)",
                (art["id"], art["kind"], art.get("task_id"), art.get("run_id"), art["path"], art.get("sha256"),
                 art["created_at"], json.dumps(art.get("meta", {}))))

    def list_artifacts(self, task_id: str | None = None) -> list[dict]:
        rows = self._q("SELECT * FROM artifacts WHERE (? IS NULL OR task_id=?) ORDER BY created_at", (task_id, task_id))
        return [{**dict(r), "meta": json.loads(r["meta"])} for r in rows]

    # ---- idempotency ----
    def idem_get(self, key: str, scope: str) -> sqlite3.Row | None:
        r = self._q("SELECT status, response FROM idempotency WHERE key=? AND scope=?", (key, scope))
        return r[0] if r else None

    def idem_put(self, key: str, scope: str, status: int, response: Any) -> None:
        self._q("INSERT OR IGNORE INTO idempotency VALUES (?,?,?,?)", (key, scope, status, json.dumps(response)))
