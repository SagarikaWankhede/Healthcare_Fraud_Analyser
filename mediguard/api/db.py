"""SQLite persistence for investigation records.

Investigation status is a *human* decision and is stored separately from model output. Every
status change is appended to ``investigation_events`` so the queue has an audit trail.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from mediguard.config import INVESTIGATION_STATUSES

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    provider_id TEXT PRIMARY KEY,
    status      TEXT NOT NULL CHECK (status IN ('not_reviewed','under_review','escalated','closed')),
    note        TEXT,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS investigation_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    status      TEXT NOT NULL,
    note        TEXT,
    model_version TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_provider ON investigation_events(provider_id);
"""


class InvestigationRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def statuses(self) -> dict[str, str]:
        with self._conn() as c:
            return {r["provider_id"]: r["status"] for r in c.execute("SELECT provider_id, status FROM investigations")}

    def get(self, provider_id: str) -> dict:
        with self._conn() as c:
            row = c.execute("SELECT * FROM investigations WHERE provider_id=?", (provider_id,)).fetchone()
            events = c.execute("SELECT status, note, model_version, created_at FROM investigation_events "
                               "WHERE provider_id=? ORDER BY id DESC LIMIT 50", (provider_id,)).fetchall()
        record = dict(row) if row else {"provider_id": provider_id, "status": "not_reviewed", "note": None,
                                        "updated_at": None}
        record["history"] = [dict(e) for e in events]
        return record

    def set_status(self, provider_id: str, status: str, note: str | None, model_version: str | None) -> dict:
        if status not in INVESTIGATION_STATUSES:
            raise ValueError(f"invalid status {status!r}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("INSERT INTO investigations (provider_id, status, note, updated_at) VALUES (?,?,?,?) "
                      "ON CONFLICT(provider_id) DO UPDATE SET status=excluded.status, note=excluded.note, "
                      "updated_at=excluded.updated_at", (provider_id, status, note, now))
            c.execute("INSERT INTO investigation_events (provider_id, status, note, model_version, created_at) "
                      "VALUES (?,?,?,?,?)", (provider_id, status, note, model_version, now))
        return self.get(provider_id)
