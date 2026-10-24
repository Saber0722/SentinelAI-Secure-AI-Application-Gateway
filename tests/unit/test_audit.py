import logging
import time

from sentinelai.audit import AuditEvent, AuditLogger, NullAuditLogger, SecretScrubFilter, percentile
from sentinelai.storage.database import Database


def event(**kw):
    base = dict(request_id="r1", event_type="chat", status="allowed", input_action="ALLOW", gateway_ms=1.0)
    return AuditEvent(**{**base, **kw})


def test_record_and_query_filters(memory_audit):
    memory_audit.record(event(status="blocked", risk_level="high", categories=["jailbreak"]))
    memory_audit.record(event(status="allowed"))
    memory_audit.record(event(status="redacted", pii_entities={"EMAIL_ADDRESS": 2}))
    total, _ = memory_audit.query({})
    assert total == 3
    assert memory_audit.query({"status": "blocked"})[0] == 1
    assert memory_audit.query({"category": "jailbreak"})[0] == 1
    assert memory_audit.query({"risk_level": "high"})[1][0]["categories"] == ["jailbreak"]
    assert memory_audit.query({"since": time.time() + 60})[0] == 0
    assert memory_audit.query({"until": time.time() + 60})[0] == 3
    _, page = memory_audit.query({}, limit=1, offset=1)
    assert len(page) == 1


def test_metrics_aggregate(memory_audit):
    for ms in (1.0, 2.0, 3.0, 4.0):
        memory_audit.record(
            event(gateway_ms=ms, inference_ms=ms * 10, risk_level="critical", categories=["pii"], pii_entities={"PERSON": 1})
        )
    memory_audit.record(event(event_type="policy_change", status="applied"))
    m = memory_audit.metrics()
    assert m["total_requests"] == 4  # policy changes are not traffic
    assert m["high_risk_events"] == 4
    assert m["categories"] == {"pii": 4} and m["pii_entities"] == {"PERSON": 4}
    assert m["latency_ms"]["gateway"]["p50"] == 2.5
    assert m["timeline"][0]["count"] == 4


def test_redacted_prompt_only_stored_when_enabled():
    db = Database(":memory:")
    AuditLogger(db).record(event(redacted_prompt="hello [EMAIL_REDACTED]"))
    AuditLogger(db, store_redacted_prompt=True).record(event(request_id="r2", redacted_prompt="x" * 1000))
    rows = {r["request_id"]: r for r in AuditLogger(db).query({})[1]}
    assert rows["r1"]["redacted_prompt"] is None
    assert len(rows["r2"]["redacted_prompt"]) == 300


def test_fingerprint_is_keyed_and_stable(tmp_path):
    key = tmp_path / "k"
    a = AuditLogger(Database(":memory:"), key)
    b = AuditLogger(Database(":memory:"), key)
    c = AuditLogger(Database(":memory:"))
    assert a.fingerprint("x") == b.fingerprint("x") != c.fingerprint("x")
    assert (key.stat().st_mode & 0o777) == 0o600


def test_eval_runs_roundtrip(memory_audit):
    memory_audit.record_eval_run("dev/protected", "v1", {"a": 1}, {"recall": 0.9})
    (run,) = memory_audit.eval_runs()
    assert run["metrics"]["recall"] == 0.9 and run["config"] == {"a": 1}


def test_null_audit_discards():
    NullAuditLogger().record(event())
    assert NullAuditLogger().fingerprint("x")


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_scrub_filter_removes_secrets_and_exception_text():
    flt = SecretScrubFilter(lambda s: s.replace("TOPSECRET", "[SECRET_REDACTED]"))
    try:
        raise ValueError("contains TOPSECRET")
    except ValueError:
        import sys

        exc_info = sys.exc_info()
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "value %s", ("TOPSECRET",), exc_info)
    assert flt.filter(record)
    assert record.getMessage() == "value [SECRET_REDACTED]"
    assert record.exc_info is None and "TOPSECRET" not in record.exc_text


def test_inference_runs_roundtrip(memory_audit):
    memory_audit.record_inference_run("inference", "gpu", "qwen2.5:1.5b", "gpu", {"tokens_per_s_p50": 176.5})
    memory_audit.record_inference_run("live_security", "baseline", "qwen2.5:1.5b", "gpu", {"returned_secret": 6})
    assert {r["kind"] for r in memory_audit.inference_runs()} == {"inference", "live_security"}
    (run,) = memory_audit.inference_runs(kind="inference")
    assert run["metrics"]["tokens_per_s_p50"] == 176.5 and run["device"] == "gpu"
