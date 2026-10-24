"""Shared types: detector findings and policy decisions.

Detectors only *report* findings; the policy engine alone turns findings into actions.
Evidence strings are written for audit logs and must never contain matched sensitive text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Category(StrEnum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    SECRET_EXFILTRATION = "secret_exfiltration"
    OBFUSCATION = "obfuscation"
    ML_SUSPICIOUS = "ml_suspicious"
    PII = "pii"
    SECRET = "secret"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    POLICY_VIOLATION = "policy_violation"


# Categories that indicate an attempt to subvert the model (as opposed to sensitive data).
ATTACK_CATEGORIES = frozenset(
    {
        Category.PROMPT_INJECTION,
        Category.JAILBREAK,
        Category.SYSTEM_PROMPT_EXTRACTION,
        Category.SECRET_EXFILTRATION,
        Category.OBFUSCATION,
    }
)


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(StrEnum):
    ALLOW = "ALLOW"
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"
    REDACT_AND_ALLOW = "REDACT_AND_ALLOW"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


# Deterministic precedence: the highest-ranked action recommended by any finding wins.
ACTION_PRECEDENCE: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.ALLOW_WITH_WARNING: 1,
    Action.REDACT_AND_ALLOW: 2,
    Action.REVIEW: 3,
    Action.BLOCK: 4,
}


def strongest(actions: list[Action]) -> Action:
    return max(actions, key=ACTION_PRECEDENCE.__getitem__, default=Action.ALLOW)


def combine_scores(scores: list[float]) -> float:
    """Noisy-OR: independent signals of strength s_i combine to 1 - prod(1 - s_i)."""
    remaining = 1.0
    for s in scores:
        remaining *= 1.0 - s
    return 1.0 - remaining


def severity_for(score: float) -> Severity:
    if score >= 0.9:
        return Severity.CRITICAL
    if score >= 0.7:
        return Severity.HIGH
    if score >= 0.35:
        return Severity.MEDIUM
    return Severity.LOW


@dataclass(frozen=True, slots=True)
class Span:
    """A character range in the scanned text. Spans are used for redaction and never logged."""

    start: int
    end: int
    label: str


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str
    detector_version: str
    category: Category
    score: float
    severity: Severity
    evidence: str
    rule_ids: tuple[str, ...] = ()
    spans: tuple[Span, ...] = ()
    entity_type: str | None = None

    def public_dict(self) -> dict[str, object]:
        """Representation safe for API responses and audit logs (no spans, no raw text)."""
        return {
            "detector": self.detector,
            "detector_version": self.detector_version,
            "category": self.category.value,
            "score": round(self.score, 4),
            "severity": self.severity.value,
            "evidence": self.evidence,
            "rule_ids": list(self.rule_ids),
            "entity_type": self.entity_type,
        }


@dataclass(slots=True)
class Decision:
    action: Action
    risk_score: float
    findings: list[Finding] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if not self.findings and self.risk_score == 0:
            return "none"
        return severity_for(self.risk_score).value

    @property
    def categories(self) -> list[str]:
        return sorted({f.category.value for f in self.findings})
