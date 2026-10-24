"""FastAPI application factory with request IDs, body-size limits and leak-free error handling."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from sentinelai import __version__
from sentinelai.api.routes import router
from sentinelai.api.state import AppState
from sentinelai.audit import install_log_scrubber
from sentinelai.config import Settings, get_settings
from sentinelai.models import ModelAdapter

log = logging.getLogger("sentinelai")

MAX_BODY_BYTES = 512 * 1024
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def create_app(settings: Settings | None = None, model: ModelAdapter | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState.build(settings, model=model)
        app.state.sentinel = state
        install_log_scrubber(state.gateway.secrets.scrub)
        log.info("SentinelAI ready: model=%s:%s", state.gateway.model.backend, state.gateway.model.model_id)
        yield
        state.db.close()

    app = FastAPI(
        title="SentinelAI Security Gateway",
        version=__version__,
        description="Local-first security gateway for LLM applications.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "")
        request.state.request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > MAX_BODY_BYTES):
            return JSONResponse({"detail": "request body too large", "request_id": request.state.request_id}, 413)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes the offending input back; it may contain PII or secrets.
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        state: AppState | None = getattr(request.app.state, "sentinel", None)
        if state is not None:
            state.gateway.record_rejection(request.state.request_id, "; ".join(str(e["msg"]) for e in errors))
        code = 413 if any(e["type"] == "string_too_long" for e in errors) else 422
        return JSONResponse({"detail": errors, "request_id": request.state.request_id}, code)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
            exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the type only; messages can contain user content.
        log.error("unhandled %s on %s %s", type(exc).__name__, request.method, request.url.path)
        return JSONResponse({"detail": "internal error", "request_id": getattr(request.state, "request_id", None)}, 500)

    app.include_router(router)
    return app
