"""SQLite tables for the experiment plane. Shares the dashboard's single writer connection (WAL) but none of the
development tables. Provider credentials are never stored: only job ids, paths, hashes and redacted error text."""
from __future__ import annotations

import json
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS exp_runs (
  n INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, owner TEXT NOT NULL, preset TEXT NOT NULL,
  state TEXT NOT NULL, backend TEXT NOT NULL, resolved TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
  started_at REAL, ended_at REAL, error TEXT, last_heartbeat REAL, backend_error TEXT, artifact_problems TEXT NOT NULL DEFAULT '[]');
CREATE INDEX IF NOT EXISTS exp_runs_state ON exp_runs (state);
CREATE TABLE IF NOT EXISTS exp_attempts (
  run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, n INTEGER NOT NULL, job_id TEXT, job_state TEXT, resume_from TEXT,
  resume_checkpoint TEXT, started_at REAL, ended_at REAL, error TEXT, result TEXT, cleaned_at REAL,
  PRIMARY KEY (run_id, attempt_id));
CREATE TABLE IF NOT EXISTS exp_metrics (
  run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, seq INTEGER NOT NULL, env_steps INTEGER, data TEXT NOT NULL,
  PRIMARY KEY (run_id, attempt_id, seq));
CREATE TABLE IF NOT EXISTS exp_checkpoints (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL,
  sha256 TEXT NOT NULL, size INTEGER NOT NULL, env_steps INTEGER, policy_version INTEGER, parent_id TEXT, role TEXT,
  meta TEXT NOT NULL, available INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS exp_evals (
  id TEXT PRIMARY KEY, run_id TEXT, suite TEXT NOT NULL, debug INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL,
  spec TEXT NOT NULL, result TEXT, error TEXT, actor TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS exp_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT,
  outcome TEXT NOT NULL, detail TEXT NOT NULL);
"""

JSON_COLS = {"resolved", "result", "spec", "meta", "data", "artifact_problems"}


def _row(r) -> dict:
    return {k: (json.loads(r[k]) if k in JSON_COLS and r[k] is not None else r[k]) for k in r.keys()}


class ExpDb:
    def __init__(self, store):
        self.s = store
        with store._lock:
            store._db.executescript(SCHEMA)

    def q(self, sql: str, args: tuple = ()) -> list[dict]:
        return [_row(r) for r in self.s._q(sql, args)]

    def one(self, sql: str, args: tuple = ()) -> dict | None:
        r = self.q(sql, args)
        return r[0] if r else None

    def x(self, sql: str, args: tuple = ()) -> None:
        self.s._q(sql, args)

    def update(self, table: str, key: dict[str, Any], **fields: Any) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        where = " AND ".join(f"{k}=?" for k in key)
        vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
        self.x(f"UPDATE {table} SET {sets} WHERE {where}", (*vals, *key.values()))
