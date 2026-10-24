"""Thin HTTP client for the gateway API. The UI never enforces security itself."""

from __future__ import annotations

import os
from typing import Any

import httpx

API_URL = os.environ.get("SENTINEL_API_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request(method: str, path: str, *, admin_token: str | None = None, timeout: float = 30.0, **kwargs: Any) -> Any:
    headers = kwargs.pop("headers", {})
    if admin_token:
        headers["X-Admin-Token"] = admin_token
    try:
        with httpx.Client(base_url=API_URL, timeout=timeout, trust_env=False) as client:
            resp = client.request(method, path, headers=headers, **kwargs)
    except httpx.HTTPError:
        raise ApiError(f"Cannot reach the SentinelAI API at {API_URL}. Start it with `uv run sentinelai serve`.") from None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text[:200]
        raise ApiError(f"HTTP {resp.status_code}: {detail}", resp.status_code)
    return resp.json()


def get(path: str, **kwargs: Any) -> Any:
    return _request("GET", path, **kwargs)


def post(path: str, json: dict[str, Any] | None = None, **kwargs: Any) -> Any:
    return _request("POST", path, json=json, **kwargs)


def put(path: str, json: dict[str, Any], **kwargs: Any) -> Any:
    return _request("PUT", path, json=json, **kwargs)
