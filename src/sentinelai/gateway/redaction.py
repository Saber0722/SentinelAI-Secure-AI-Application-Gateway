"""Span-based redaction with consistent placeholders such as ``[EMAIL_REDACTED]``."""

from __future__ import annotations

from sentinelai.gateway.decisions import Finding, Span

_PLACEHOLDER_NAMES = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "IP_ADDRESS": "IP_ADDRESS",
    "US_SSN": "SSN",
    "IN_PAN": "PAN",
    "IN_AADHAAR": "AADHAAR",
    "IBAN_CODE": "IBAN",
}

_SECRET_LABELS = {
    "PROTECTED_SECRET",
    "PRIVATE_KEY",
    "AWS_ACCESS_KEY",
    "GITHUB_TOKEN",
    "SLACK_TOKEN",
    "API_KEY",
    "GOOGLE_API_KEY",
    "STRIPE_KEY",
    "VAULT_TOKEN",
    "JWT",
    "CONNECTION_STRING",
    "CREDENTIAL_ASSIGNMENT",
}


def placeholder(label: str) -> str:
    if label in _SECRET_LABELS:
        return "[SECRET_REDACTED]"
    return f"[{_PLACEHOLDER_NAMES.get(label, label)}_REDACTED]"


def _merge(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans; the longest span decides the label of a merged region."""
    merged: list[tuple[Span, int]] = []  # (span, length of the span that set the label)
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if merged and span.start < merged[-1][0].end:
            prev, prev_len = merged[-1]
            label = span.label if (span.end - span.start) > prev_len else prev.label
            new = Span(prev.start, max(prev.end, span.end), label)
            merged[-1] = (new, max(prev_len, span.end - span.start))
        else:
            merged.append((span, span.end - span.start))
    return [s for s, _ in merged]


def redact(text: str, findings: list[Finding]) -> tuple[str, int]:
    """Return ``text`` with every finding span replaced, and the number of replacements."""
    spans = [s for f in findings for s in f.spans if 0 <= s.start < s.end <= len(text)]
    if not spans:
        return text, 0
    out: list[str] = []
    cursor = 0
    merged = _merge(spans)
    for span in merged:
        out.append(text[cursor : span.start])
        out.append(placeholder(span.label))
        cursor = span.end
    out.append(text[cursor:])
    return "".join(out), len(merged)
