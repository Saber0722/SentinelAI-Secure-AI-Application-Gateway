"""Typed request/response models. Requests forbid unknown fields, so a client cannot smuggle
policy overrides (e.g. ``"security_enabled": false``) into a chat request."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentinelai.config import OutputSecretAction, PiiAction
from sentinelai.gateway.decisions import Decision

# Hard schema cap; the configurable (smaller) limit is enforced by the gateway with HTTP 413.
SCHEMA_MAX_PROMPT = 100_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(_Strict):
    prompt: str = Field(..., max_length=SCHEMA_MAX_PROMPT, description="User prompt")


class ScanRequest(ChatRequest):
    pass


class FindingOut(BaseModel):
    detector: str
    detector_version: str
    category: str
    score: float
    severity: str
    evidence: str
    rule_ids: list[str]
    entity_type: str | None = None


class DecisionOut(BaseModel):
    action: str
    risk_score: float
    risk_level: str
    categories: list[str]
    reasons: list[str]
    findings: list[FindingOut]

    @classmethod
    def from_decision(cls, d: Decision) -> DecisionOut:
        return cls(
            action=d.action.value,
            risk_score=d.risk_score,
            risk_level=d.risk_level,
            categories=d.categories,
            reasons=d.reasons,
            findings=[FindingOut(**f.public_dict()) for f in d.findings],  # type: ignore[arg-type]
        )


class Timings(BaseModel):
    gateway_ms: float
    inference_ms: float | None
    total_ms: float
    output_tokens: int | None = None
    device: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    status: str
    response: str
    input: DecisionOut
    output: DecisionOut | None
    model_id: str | None
    inference_attempted: bool
    redactions: int
    timings: Timings


class ScanResponse(BaseModel):
    request_id: str
    decision: DecisionOut
    would_send_to_model: str | None = Field(
        None, description="The (redacted) prompt the model would receive; null if the request would be blocked"
    )
    gateway_ms: float


class EventsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]


class PolicyPatch(_Strict):
    input_rules_enabled: bool | None = None
    injection_block_threshold: float | None = Field(None, ge=0, le=1)
    injection_warn_threshold: float | None = Field(None, ge=0, le=1)
    pii_input_enabled: bool | None = None
    pii_input_action: PiiAction | None = None
    pii_output_action: PiiAction | None = None
    pii_entities: list[str] | None = None
    pii_min_score: float | None = Field(None, ge=0, le=1)
    input_secret_action: PiiAction | None = None
    classifier_enabled: bool | None = None
    classifier_block_threshold: float | None = Field(None, ge=0, le=1)
    classifier_warn_threshold: float | None = Field(None, ge=0, le=1)
    output_scanning_enabled: bool | None = None
    output_secret_action: OutputSecretAction | None = None
    output_denylist: list[str] | None = None


class ModelSelection(_Strict):
    backend: Literal["mock", "ollama", "openai_compatible"]
    name: str | None = Field(None, max_length=200, pattern=r"^[\w.:/-]+$")
    device: Literal["auto", "gpu", "cpu"] | None = None
    base_url: str | None = Field(None, max_length=200)


class SettingsUpdate(_Strict):
    policy: PolicyPatch | None = None
    model: ModelSelection | None = None


class EvaluateRequest(_Strict):
    include_classifier: bool = True
