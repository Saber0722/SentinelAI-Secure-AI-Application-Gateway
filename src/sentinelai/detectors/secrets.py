"""Secret detection: configured (protected) secrets plus high-confidence credential patterns.

Protected secrets are matched exactly and in a few simple obfuscated forms (separators removed,
reversed, hex, base64). Evidence only names the secret's *id* -- never its value.
"""

from __future__ import annotations

import binascii
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sentinelai.gateway.decisions import Category, Finding, Severity, Span
from sentinelai.gateway.normalize import decode_base64_segments

DETECTOR_NAME = "secrets"
DETECTOR_VERSION = "1.0.0"
PLACEHOLDER = "[SECRET_REDACTED]"


@dataclass(frozen=True, slots=True)
class ProtectedSecret:
    secret_id: str
    value: str


@dataclass(frozen=True, slots=True)
class SecretsConfig:
    secrets: tuple[ProtectedSecret, ...]
    canary: str | None = None

    @classmethod
    def load(cls, path: Path) -> SecretsConfig:
        if not path.exists():
            return cls(secrets=())
        data = json.loads(path.read_text(encoding="utf-8"))
        secrets = tuple(ProtectedSecret(s["id"], s["value"]) for s in data.get("secrets", []))
        for s in secrets:
            if len(s.value) < 8:
                raise ValueError(f"protected secret {s.secret_id!r} is too short to match reliably")
        return cls(secrets=secrets, canary=data.get("canary"))

    @property
    def all_values(self) -> list[tuple[str, str]]:
        pairs = [(s.secret_id, s.value) for s in self.secrets]
        if self.canary:
            pairs.append(("canary", self.canary))
        return pairs


# (label, pattern, value group index or 0 for the whole match)
_GENERIC: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "PRIVATE_KEY",
        re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----[\s\S]*?(?:-----END (?:[A-Z]+ )*PRIVATE KEY-----|\Z)"),
        0,
    ),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b"), 0),
    ("SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), 0),
    ("API_KEY", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"), 0),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), 0),
    ("STRIPE_KEY", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[0-9a-zA-Z]{16,}\b"), 0),
    ("VAULT_TOKEN", re.compile(r"\bhvs\.[A-Za-z0-9_-]{20,}\b"), 0),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), 0),
    ("CONNECTION_STRING", re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/@]+:([^\s@]+)@"), 1),
    (
        "CREDENTIAL_ASSIGNMENT",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"client[_-]?secret)\b\s*(?:[:=]|\bis\b)\s*['\"]?([^\s'\",;]{6,})"
        ),
        1,
    ),
)

_ID_MENTION = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)+")


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


class SecretsDetector:
    name = DETECTOR_NAME
    version = DETECTOR_VERSION

    def __init__(self, config: SecretsConfig) -> None:
        self.config = config

    # -- protected secrets ---------------------------------------------------------------
    def _protected(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        compact = _compact(text)
        decoded = " ".join(decode_base64_segments(text, require_words=False))
        for secret_id, value in self.config.all_values:
            spans = tuple(Span(m.start(), m.end(), "PROTECTED_SECRET") for m in re.finditer(re.escape(value), text))
            forms: list[str] = []
            if spans:
                forms.append("exact")
            cv = _compact(value)
            if not spans and len(cv) >= 8 and cv in compact:
                forms.append("separator-obfuscated")
            if value[::-1] in text:
                forms.append("reversed")
            if binascii.hexlify(value.encode()).decode() in compact:
                forms.append("hex-encoded")
            if decoded and (value in decoded or (len(cv) >= 8 and cv in _compact(decoded))):
                forms.append("base64-encoded")
            if forms:
                findings.append(
                    Finding(
                        detector=self.name,
                        detector_version=self.version,
                        category=Category.SYSTEM_PROMPT_LEAK if secret_id == "canary" else Category.SECRET,
                        score=1.0,
                        severity=Severity.CRITICAL,
                        evidence=f"protected value '{secret_id}' present ({', '.join(forms)})",
                        rule_ids=(f"PROTECTED:{secret_id}",),
                        spans=spans,
                        entity_type="PROTECTED_SECRET",
                    )
                )
        return findings

    # -- generic credential patterns -------------------------------------------------------
    def _generic(self, text: str) -> list[Finding]:
        by_label: dict[str, list[Span]] = {}
        for label, pattern, group in _GENERIC:
            for m in pattern.finditer(text):
                start, end = m.span(group)
                by_label.setdefault(label, []).append(Span(start, end, label))
        findings = []
        for label, spans in by_label.items():
            weak = label == "CREDENTIAL_ASSIGNMENT"
            findings.append(
                Finding(
                    detector=self.name,
                    detector_version=self.version,
                    category=Category.SECRET,
                    score=0.6 if weak else 0.95,
                    severity=Severity.MEDIUM if weak else Severity.HIGH,
                    evidence=f"{len(spans)} {label} pattern match(es)",
                    rule_ids=(f"GENERIC:{label}",),
                    spans=tuple(spans),
                    entity_type=label,
                )
            )
        return findings

    def scan(self, text: str) -> list[Finding]:
        return self._protected(text) + self._generic(text)

    def scan_requests(self, text: str) -> list[Finding]:
        """Input-side: mentions of internal secret identifiers (e.g. ``internal_api_key``)."""
        ids = {s.secret_id for s in self.config.secrets}
        mentioned = sorted(ids.intersection(_ID_MENTION.findall(text.casefold())))
        if not mentioned:
            return []
        return [
            Finding(
                detector=self.name,
                detector_version=self.version,
                category=Category.SECRET_EXFILTRATION,
                score=0.75,
                severity=Severity.HIGH,
                evidence=f"references internal secret identifier(s): {', '.join(mentioned)}",
                rule_ids=tuple(f"SECRET_ID:{m}" for m in mentioned),
            )
        ]

    def scrub(self, text: str) -> str:
        """Best-effort removal of secrets from arbitrary text (used for log messages)."""
        for _, value in self.config.all_values:
            text = text.replace(value, PLACEHOLDER)
        for _, pattern, group in _GENERIC:
            text = pattern.sub(lambda m, g=group: m.group(0).replace(m.group(g), PLACEHOLDER) if g else PLACEHOLDER, text)
        return text
