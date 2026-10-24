"""SQLite storage for audit events and evaluation runs."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id                  TEXT PRIMARY KEY,
    ts                  TEXT NOT NULL,
    ts_epoch            REAL NOT NULL,
    request_id          TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    status              TEXT NOT NULL,
    input_action        TEXT,
    output_action       TEXT,
    risk_score          REAL NOT NULL DEFAULT 0,
    risk_level          TEXT NOT NULL DEFAULT 'none',
    categories          TEXT NOT NULL DEFAULT '[]',
    detectors           TEXT NOT NULL DEFAULT '[]',
    reasons             TEXT NOT NULL DEFAULT '[]',
    findings            TEXT NOT NULL DEFAULT '[]',
    pii_entities        TEXT NOT NULL DEFAULT '{}',
    pii_count           INTEGER NOT NULL DEFAULT 0,
    redactions          INTEGER NOT NULL DEFAULT 0,
    model_id            TEXT,
    inference_attempted INTEGER NOT NULL DEFAULT 0,
    gateway_ms          REAL,
    inference_ms        REAL,
    total_ms            REAL,
    prompt_chars        INTEGER,
    prompt_fingerprint  TEXT,
    redacted_prompt     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

CREATE TABLE IF NOT EXISTS eval_runs (
    id              TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    label           TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    config          TEXT NOT NULL,
    metrics         TEXT NOT NULL
);

-- Real-model runs: GPU/CPU inference benchmarks and live security runs.
CREATE TABLE IF NOT EXISTS inference_runs (
    id      TEXT PRIMARY KEY,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    label   TEXT NOT NULL,
    model   TEXT NOT NULL,
    device  TEXT,
    metrics TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
