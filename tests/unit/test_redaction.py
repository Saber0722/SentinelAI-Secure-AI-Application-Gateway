from sentinelai.gateway.decisions import Category, Finding, Severity, Span
from sentinelai.gateway.redaction import placeholder, redact


def finding(*spans):
    return Finding("t", "1", Category.PII, 1.0, Severity.HIGH, "e", spans=tuple(spans))


def test_placeholders_are_consistent():
    assert placeholder("EMAIL_ADDRESS") == "[EMAIL_REDACTED]"
    assert placeholder("PERSON") == "[PERSON_REDACTED]"
    assert placeholder("JWT") == placeholder("PROTECTED_SECRET") == "[SECRET_REDACTED]"


def test_redacts_multiple_spans():
    text = "a@b.co and c@d.co"
    out, n = redact(text, [finding(Span(0, 6, "EMAIL_ADDRESS"), Span(11, 17, "EMAIL_ADDRESS"))])
    assert out == "[EMAIL_REDACTED] and [EMAIL_REDACTED]" and n == 2


def test_overlapping_spans_merge_and_longest_label_wins():
    text = "visit jane@example.com today"
    out, n = redact(text, [finding(Span(6, 22, "EMAIL_ADDRESS")), finding(Span(11, 22, "URL"))])
    assert out == "visit [EMAIL_REDACTED] today" and n == 1


def test_out_of_range_spans_are_ignored():
    assert redact("short", [finding(Span(3, 99, "X"))]) == ("short", 0)


def test_no_findings():
    assert redact("hello", []) == ("hello", 0)
