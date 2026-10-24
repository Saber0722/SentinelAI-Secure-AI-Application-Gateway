"""Structured security audit log.

By default no prompt or response text is stored: only metadata, safe evidence strings, a prompt
length and a keyed (HMAC) fingerprint that lets repeated prompts be correlated without storing
them. Setting ``SENTINEL_AUDIT_STORE_REDACTED_PROMPT=true`` additionally stores the prompt
*after* PII/secret redaction, truncated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinelai.storage.database import Database

REDACTED_PROMPT_MAX = 300


@dataclass(slots=True)
class AuditEvent:
    request_id: str
    event_type: str  # chat | scan | rejected | policy_change
    status: str
    input_action: str | None = None
    output_action: str | None = None
    risk_score: float = 0.0
    risk_level: str = "none"
    categories: list[str] = field(default_factory=list)
    detectors: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    pii_entities: dict[str, int] = field(default_factory=dict)
    redactions: int = 0
    model_id: str | None = None
    inference_attempted: bool = False
    gateway_ms: float | None = None
    inference_ms: float | None = None
    total_ms: float | None = None
    prompt_chars: int | None = None
    prompt_fingerprint: str | None = None
    redacted_prompt: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_epoch: float = field(default_factory=time.time)


def _load_or_create_key(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1], 3)


class AuditLogger:
    def __init__(self, db: Database, key_path: Path | None = None, store_redacted_prompt: bool = False) -> None:
        self.db = db
        self.store_redacted_prompt = store_redacted_prompt
        self._key = _load_or_create_key(key_path) if key_path else secrets.token_bytes(32)

    def fingerprint(self, text: str) -> str:
        return hmac.new(self._key, text.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    def record(self, event: AuditEvent) -> None:
        row = asdict(event)
        row["ts"] = datetime.fromtimestamp(event.ts_epoch, UTC).isoformat(timespec="milliseconds")
        row["pii_count"] = sum(event.pii_entities.values())
        row["inference_attempted"] = int(event.inference_attempted)
        if not self.store_redacted_prompt:
            row["redacted_prompt"] = None
        elif row["redacted_prompt"]:
            row["redacted_prompt"] = row["redacted_prompt"][:REDACTED_PROMPT_MAX]
        for key in ("categories", "detectors", "reasons", "findings", "pii_entities"):
            row[key] = json.dumps(row[key])
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self.db.cursor() as cur:
            cur.execute(f"INSERT INTO events ({cols}) VALUES ({marks})", row)

    # -- queries ---------------------------------------------------------------------------
    @staticmethod
    def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses, params = [], []
        for col in ("event_type", "status", "risk_level", "input_action", "output_action", "request_id"):
            if filters.get(col):
                clauses.append(f"{col} = ?")
                params.append(filters[col])
        if filters.get("category"):
            clauses.append("EXISTS (SELECT 1 FROM json_each(events.categories) WHERE value = ?)")
            params.append(filters["category"])
        if filters.get("since") is not None:
            clauses.append("ts_epoch >= ?")
            params.append(filters["since"])
        if filters.get("until") is not None:
            clauses.append("ts_epoch <= ?")
            params.append(filters["until"])
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def query(self, filters: dict[str, Any], limit: int = 50, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
        where, params = self._where(filters)
        with self.db.cursor() as cur:
            total = cur.execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0]
            rows = cur.execute(
                f"SELECT * FROM events{where} ORDER BY ts_epoch DESC LIMIT ? OFFSET ?", [*params, limit, offset]
            ).fetchall()
        return total, [self._decode(r) for r in rows]

    @staticmethod
    def _decode(row: Any) -> dict[str, Any]:
        d = dict(row)
        for key in ("categories", "detectors", "reasons", "findings", "pii_entities"):
            d[key] = json.loads(d[key])
        d["inference_attempted"] = bool(d["inference_attempted"])
        return d

    def metrics(self, since: float | None = None) -> dict[str, Any]:
        where, params = self._where({"since": since})
        traffic = (where + " AND " if where else " WHERE ") + "event_type IN ('chat', 'scan', 'rejected')"
        with self.db.cursor() as cur:

            def grouped(col: str) -> dict[str, int]:
                q = f"SELECT {col}, COUNT(*) FROM events{traffic} GROUP BY {col}"
                return {str(k): v for k, v in cur.execute(q, params).fetchall()}

            by_status = grouped("status")
            by_event_type = grouped("event_type")
            by_risk = grouped("risk_level")
            by_input_action = grouped("input_action")
            categories = dict(
                cur.execute(
                    f"SELECT j.value, COUNT(*) FROM events, json_each(events.categories) j{traffic} GROUP BY j.value",
                    params,
                ).fetchall()
            )
            pii = dict(
                cur.execute(
                    f"SELECT j.key, SUM(j.value) FROM events, json_each(events.pii_entities) j{traffic} GROUP BY j.key",
                    params,
                ).fetchall()
            )
            gateway = [
                r[0] for r in cur.execute(f"SELECT gateway_ms FROM events{traffic} AND gateway_ms IS NOT NULL", params)
            ]
            inference = [
                r[0] for r in cur.execute(f"SELECT inference_ms FROM events{traffic} AND inference_ms IS NOT NULL", params)
            ]
            timeline = [
                dict(zip(("bucket", "status", "count"), r, strict=True))
                for r in cur.execute(
                    f"SELECT strftime('%Y-%m-%dT%H:00', ts), status, COUNT(*) FROM events{traffic} GROUP BY 1, 2 ORDER BY 1",
                    params,
                ).fetchall()
            ]
            total = sum(by_status.values())
        return {
            "total_requests": total,
            "by_status": by_status,
            "by_event_type": by_event_type,
            "by_risk_level": by_risk,
            "by_input_action": by_input_action,
            "categories": categories,
            "pii_entities": pii,
            "high_risk_events": by_risk.get("high", 0) + by_risk.get("critical", 0),
            "latency_ms": {
                "gateway": {"p50": percentile(gateway, 50), "p95": percentile(gateway, 95), "n": len(gateway)},
                "inference": {"p50": percentile(inference, 50), "p95": percentile(inference, 95), "n": len(inference)},
            },
            "timeline": timeline,
        }

    # -- evaluation runs -------------------------------------------------------------------
    def record_eval_run(self, label: str, dataset_version: str, config: dict[str, Any], metrics: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_runs (id, ts, label, dataset_version, config, metrics) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    label,
                    dataset_version,
                    json.dumps(config),
                    json.dumps(metrics),
                ),
            )
        return run_id

    def eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.db.cursor() as cur:
            rows = cur.execute("SELECT * FROM eval_runs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(r), "config": json.loads(r["config"]), "metrics": json.loads(r["metrics"])} for r in rows]

    def record_inference_run(self, kind: str, label: str, model: str, device: str | None, metrics: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO inference_runs (id, ts, kind, label, model, device, metrics) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, datetime.now(UTC).isoformat(timespec="seconds"), kind, label, model, device, json.dumps(metrics)),
            )
        return run_id

    def inference_runs(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM inference_runs", []
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        with self.db.cursor() as cur:
            rows = cur.execute(query + " ORDER BY ts DESC LIMIT ?", [*params, limit]).fetchall()
        return [{**dict(r), "metrics": json.loads(r["metrics"])} for r in rows]


class NullAuditLogger(AuditLogger):
    """Discards events; used by the benchmark so evaluation traffic does not pollute the audit log."""

    def __init__(self) -> None:
        self.store_redacted_prompt = False
        self._key = secrets.token_bytes(32)

    def record(self, event: AuditEvent) -> None:
        return None


class SecretScrubFilter(logging.Filter):
    """Logging filter that removes protected secrets and credential patterns from log records."""

    def __init__(self, scrub: Any) -> None:
        super().__init__()
        self._scrub = scrub

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            message = str(record.msg)
        record.msg = self._scrub(message)
        record.args = None
        if record.exc_info and record.exc_info[1] is not None:
            # Keep the exception type, drop its message (may contain user content).
            record.exc_text = f"{type(record.exc_info[1]).__name__} (details scrubbed)"
            record.exc_info = None
        return True


def install_log_scrubber(scrub: Any) -> None:
    flt = SecretScrubFilter(scrub)
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "sentinelai"):
        for handler in logging.getLogger(name).handlers:
            if not any(isinstance(f, SecretScrubFilter) for f in handler.filters):
                handler.addFilter(flt)
