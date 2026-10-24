"""REST endpoints. Every model call goes through ``SecurityGateway``; admin endpoints need a token."""

from __future__ import annotations

import hmac
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import ValidationError

from sentinelai import __version__
from sentinelai.api.schemas import (
    ChatRequest,
    ChatResponse,
    DecisionOut,
    EvaluateRequest,
    EventsResponse,
    ScanRequest,
    ScanResponse,
    SettingsUpdate,
    Timings,
)
from sentinelai.api.state import AppState
from sentinelai.audit import AuditEvent
from sentinelai.config import DEFAULT_PII_ENTITIES, SUPPORTED_PII_ENTITIES, SecurityPolicy, check_model_url
from sentinelai.environment import detect_nvidia_gpus
from sentinelai.gateway.decisions import Action
from sentinelai.gateway.pipeline import InputValidationError
from sentinelai.models import GenerationConfig, build_model

router = APIRouter()


def get_state(request: Request) -> AppState:
    return request.app.state.sentinel


State = Annotated[AppState, Depends(get_state)]


def require_admin(state: State, x_admin_token: Annotated[str | None, Header()] = None) -> None:
    configured = state.settings.admin_token
    if configured is None:
        raise HTTPException(403, "admin API disabled; set SENTINEL_ADMIN_TOKEN to enable it")
    if x_admin_token is None or not hmac.compare_digest(x_admin_token, configured.get_secret_value()):
        raise HTTPException(401, "invalid or missing X-Admin-Token")


@router.get("/health")
def health(state: State) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "model_backend": state.gateway.model.backend,
        "uptime_s": round(time.time() - state.started_at, 1),
    }


@router.get("/api/v1/status")
def status(state: State) -> dict[str, Any]:
    ms = state.gateway.model.status()
    policy = state.gateway.policy
    return {
        "version": __version__,
        "model": {
            "backend": ms.backend,
            "model_id": ms.model_id,
            "available": ms.available,
            "detail": ms.detail,
            "device": ms.device,
            **ms.extra,
        },
        "hardware": {"nvidia_gpus": detect_nvidia_gpus()},
        "classifier": state.classifier_status(),
        "policy": policy.model_dump(mode="json"),
        "detectors": state.gateway.detector_versions(),
        "limits": {"max_input_chars": state.gateway.max_input_chars},
        "admin_enabled": state.settings.admin_token is not None,
        "supported_pii_entities": SUPPORTED_PII_ENTITIES,
        "default_pii_entities": DEFAULT_PII_ENTITIES,
    }


def _validated(state: AppState, request: Request, prompt: str) -> None:
    try:
        state.gateway.validate(prompt)
    except InputValidationError as exc:
        state.gateway.record_rejection(request.state.request_id, str(exc), prompt)
        raise HTTPException(exc.status_code, str(exc)) from None


@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request, state: State) -> ChatResponse:
    _validated(state, request, body.prompt)
    try:
        r = state.gateway.chat(body.prompt, request_id=request.state.request_id)
    except InputValidationError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    return ChatResponse(
        request_id=r.request_id,
        status=r.status.value,
        response=r.response,
        input=DecisionOut.from_decision(r.input_decision),
        output=DecisionOut.from_decision(r.output_decision) if r.output_decision else None,
        model_id=r.model_id,
        inference_attempted=r.inference_attempted,
        redactions=r.redactions,
        timings=Timings(
            gateway_ms=round(r.gateway_ms, 3),
            inference_ms=round(r.inference_ms, 3) if r.inference_ms is not None else None,
            total_ms=round(r.total_ms, 3),
            output_tokens=r.output_tokens,
            device=r.device,
        ),
    )


@router.post("/api/v1/scan", response_model=ScanResponse)
def scan(body: ScanRequest, request: Request, state: State) -> ScanResponse:
    _validated(state, request, body.prompt)
    try:
        request_id, analysis = state.gateway.scan(body.prompt, request_id=request.state.request_id)
    except InputValidationError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    blocked = analysis.decision.action in (Action.BLOCK, Action.REVIEW)
    return ScanResponse(
        request_id=request_id,
        decision=DecisionOut.from_decision(analysis.decision),
        would_send_to_model=None if blocked else analysis.sanitized_prompt,
        gateway_ms=round(analysis.duration_ms, 3),
    )


@router.get("/api/v1/events", response_model=EventsResponse)
def events(
    state: State,
    event_type: str | None = None,
    status: str | None = None,
    risk_level: str | None = None,
    input_action: str | None = None,
    category: str | None = None,
    request_id: str | None = None,
    since: Annotated[float | None, Query(description="Unix timestamp (inclusive)")] = None,
    until: Annotated[float | None, Query(description="Unix timestamp (inclusive)")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EventsResponse:
    filters = {
        "event_type": event_type,
        "status": status,
        "risk_level": risk_level,
        "input_action": input_action,
        "category": category,
        "request_id": request_id,
        "since": since,
        "until": until,
    }
    total, items = state.audit.query(filters, limit=limit, offset=offset)
    return EventsResponse(total=total, limit=limit, offset=offset, items=items)


@router.get("/api/v1/metrics")
def metrics(state: State, since: float | None = None) -> dict[str, Any]:
    data = state.audit.metrics(since=since)
    runs = state.audit.eval_runs(limit=12)
    # Headline the most trustworthy configuration from the newest benchmark batch.
    batch = {r["label"]: r for r in runs if runs and r["ts"] == runs[0]["ts"]}
    preferred = ("holdout/protected_ml", "holdout/protected", "dev/protected_ml", "dev/protected")
    data["latest_evaluation"] = next((batch[k] for k in preferred if k in batch), runs[0] if runs else None)
    return data


@router.get("/api/v1/evaluations")
def evaluations(state: State, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[dict[str, Any]]:
    return state.audit.eval_runs(limit=limit)


@router.get("/api/v1/inference-runs")
def inference_runs(
    state: State, kind: str | None = None, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[dict[str, Any]]:
    return state.audit.inference_runs(kind=kind, limit=limit)


@router.post("/api/v1/evaluate", dependencies=[Depends(require_admin)])
def evaluate(state: State, body: EvaluateRequest | None = None) -> dict[str, Any]:
    body = body or EvaluateRequest()
    return state.run_evaluation(include_classifier=body.include_classifier)


@router.get("/api/v1/models")
def models(state: State) -> dict[str, Any]:
    return {
        "backends": ["mock", "ollama", "openai_compatible"],
        "active": {"backend": state.gateway.model.backend, "model_id": state.gateway.model.model_id},
        "available": state.gateway.model.list_models(),
    }


@router.put("/api/v1/settings", dependencies=[Depends(require_admin)])
def update_settings(body: SettingsUpdate, request: Request, state: State) -> dict[str, Any]:
    changes: list[str] = []
    if body.policy is not None:
        patch = body.policy.model_dump(exclude_none=True)
        if patch.get("classifier_enabled") and state.gateway.classifier is None:
            raise HTTPException(409, "classifier is not trained; run scripts/train_classifier.py first")
        try:
            new_policy = SecurityPolicy.model_validate({**state.gateway.policy.model_dump(), **patch})
        except ValidationError as exc:
            raise HTTPException(422, [str(e["msg"]) for e in exc.errors()]) from None
        state.gateway.update_policy(new_policy)
        changes += [f"policy.{k}" for k in sorted(patch)]
    if body.model is not None:
        s = state.settings
        base_url = body.model.base_url or s.model_base_url
        try:
            check_model_url(base_url, s.allow_remote_model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        config = GenerationConfig(s.max_output_tokens, s.temperature, s.model_timeout_s, s.seed)
        model = build_model(
            body.model.backend, body.model.name or s.model_name, base_url, body.model.device or s.inference_device, config
        )
        state.gateway.set_model(model)
        changes.append(f"model={model.backend}:{model.model_id}")
    if changes:
        state.audit.record(
            AuditEvent(
                request_id=request.state.request_id,
                event_type="policy_change",
                status="applied",
                reasons=changes,
                detectors=state.gateway.detector_versions(),
                model_id=state.gateway.model.model_id,
            )
        )
    return {"applied": changes, "policy": state.gateway.policy.model_dump(mode="json")}
