"""Shared chart styling (validated categorical + reserved status palette, light and dark steps)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
# Status colours are reserved for request outcomes and always shown with a text label.
STATUS_COLORS = {
    "allowed": "#0ca30c",
    "allowed_with_warning": "#fab219",
    "redacted": "#2a78d6",
    "review": "#ec835a",
    "blocked": "#d03b3b",
    "output_blocked": "#d03b3b",
    "inference_error": "#898781",
    "rejected": "#898781",
}
STATUS_LABELS = {
    "allowed": "Allowed",
    "allowed_with_warning": "Allowed with warning",
    "redacted": "Redacted",
    "review": "Held for review",
    "blocked": "Blocked (input)",
    "output_blocked": "Blocked (output)",
    "inference_error": "Inference failure",
    "rejected": "Rejected (invalid)",
}
STATUS_ICONS = {
    "allowed": ":material/check_circle:",
    "allowed_with_warning": ":material/warning:",
    "redacted": ":material/visibility_off:",
    "review": ":material/pending:",
    "blocked": ":material/block:",
    "output_blocked": ":material/block:",
    "inference_error": ":material/error:",
    "rejected": ":material/cancel:",
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def series_colors() -> list[str]:
    return CATEGORICAL[mode()]


PLOTLY_CONFIG = {"displayModeBar": False}


def finish(fig: go.Figure, height: int = 320, **layout: object) -> go.Figure:
    """Recessive axes, no chart junk, one y-axis, hover enabled. Titles are rendered by Streamlit
    above the chart so they never collide with the legend."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        bargap=0.35,
        bargroupgap=0.08,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        **layout,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig
