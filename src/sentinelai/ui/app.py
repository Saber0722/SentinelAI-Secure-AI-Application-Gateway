"""Streamlit entry point: ``uv run sentinelai ui`` (or ``streamlit run src/sentinelai/ui/app.py``)."""

from __future__ import annotations

import streamlit as st

from sentinelai.ui import client
from sentinelai.ui.views import chat, dashboard, events, settings

st.set_page_config(page_title="SentinelAI", page_icon=":material/shield:", layout="wide")

pages = [
    st.Page(chat.render, title="Chat", icon=":material/chat:", url_path="chat", default=True),
    st.Page(dashboard.render, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
    st.Page(events.render, title="Audit events", icon=":material/list_alt:", url_path="events"),
    st.Page(settings.render, title="Settings", icon=":material/settings:", url_path="settings"),
]

with st.sidebar:
    st.markdown("### :material/shield: SentinelAI")
    try:
        health = client.get("/health", timeout=3)
        st.caption(f"Gateway online · v{health['version']} · model `{health['model_backend']}`")
    except client.ApiError:
        st.caption(f"Gateway offline ({client.API_URL})")

st.navigation(pages).run()
