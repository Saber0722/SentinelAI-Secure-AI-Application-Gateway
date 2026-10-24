"""Audit-event explorer with filters and pagination."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from sentinelai.ui import client
from sentinelai.ui.style import STATUS_LABELS

EVENT_TYPES = ["", "chat", "scan", "rejected", "policy_change"]
RISK = ["", "none", "low", "medium", "high", "critical"]
ACTIONS = ["", "ALLOW", "ALLOW_WITH_WARNING", "REDACT_AND_ALLOW", "REVIEW", "BLOCK"]
CATEGORIES = [
    "",
    "prompt_injection",
    "jailbreak",
    "system_prompt_extraction",
    "secret_exfiltration",
    "obfuscation",
    "ml_suspicious",
    "pii",
    "secret",
    "system_prompt_leak",
    "policy_violation",
]
WINDOWS = {"Any time": None, "Last hour": 3600, "Last 24 hours": 86400, "Last 7 days": 7 * 86400}
PAGE = 25


def render() -> None:
    st.title("Audit events")
    st.caption("Structured events from SQLite. Raw prompts and responses are not stored by default.")
    f = st.columns(6)
    event_type = f[0].selectbox("Event type", EVENT_TYPES)
    status = f[1].selectbox("Status", ["", *STATUS_LABELS, "applied"])
    risk = f[2].selectbox("Risk level", RISK)
    action = f[3].selectbox("Input action", ACTIONS)
    category = f[4].selectbox("Category", CATEGORIES)
    window = f[5].selectbox("Time range", list(WINDOWS))

    params = {
        k: v
        for k, v in {
            "event_type": event_type,
            "status": status,
            "risk_level": risk,
            "input_action": action,
            "category": category,
        }.items()
        if v
    }
    if WINDOWS[window]:
        params["since"] = time.time() - WINDOWS[window]
    page = st.session_state.setdefault("events_page", 0)
    params.update(limit=PAGE, offset=page * PAGE)
    try:
        data = client.get("/api/v1/events", params=params)
    except client.ApiError as exc:
        st.error(str(exc))
        return

    total = data["total"]
    pages = max(1, -(-total // PAGE))
    st.caption(f"{total} matching events · page {min(page + 1, pages)} of {pages}")
    if not data["items"]:
        st.info("No events match these filters.")
        return
    df = pd.DataFrame(data["items"])
    cols = [
        "ts",
        "event_type",
        "status",
        "input_action",
        "output_action",
        "risk_level",
        "risk_score",
        "categories",
        "pii_count",
        "redactions",
        "inference_attempted",
        "gateway_ms",
        "inference_ms",
        "model_id",
        "request_id",
    ]
    selection = st.dataframe(df[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")

    nav = st.columns([1, 1, 6])
    if nav[0].button("◀ Prev", disabled=page == 0):
        st.session_state.events_page = page - 1
        st.rerun()
    if nav[1].button("Next ▶", disabled=page + 1 >= pages):
        st.session_state.events_page = page + 1
        st.rerun()

    rows = selection.selection.rows if selection else []
    if rows:
        event = data["items"][rows[0]]
        st.subheader("Event detail")
        st.markdown("**Reasons**")
        for r in event["reasons"]:
            st.markdown(f"- {r}")
        if event["findings"]:
            st.dataframe(pd.DataFrame(event["findings"]), hide_index=True, width="stretch")
        st.json(
            {k: event[k] for k in ("id", "request_id", "detectors", "pii_entities", "prompt_chars", "prompt_fingerprint")}
        )
