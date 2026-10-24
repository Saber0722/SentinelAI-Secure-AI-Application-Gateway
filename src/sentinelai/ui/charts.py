"""Chart builders shared by the dashboard. One y-axis per chart, thin rounded bars, value labels, hover.

Colour follows the entity: each security configuration and each inference device keeps the same colour in
every chart. Status colours are only used for request outcomes and risk levels, always next to a label.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import plotly.graph_objects as go
import streamlit as st

from sentinelai.ui.style import PLOTLY_CONFIG, finish, mode, series_colors

CONFIG_ORDER = ["baseline", "protected", "protected_ml"]
CONFIG_LABELS = {"baseline": "Baseline (no controls)", "protected": "Rules only", "protected_ml": "Rules + ML"}
DEVICE_ORDER = ["gpu", "cpu", "auto"]
DEVICE_LABELS = {"gpu": "GPU", "cpu": "CPU", "auto": "Auto"}

RISK_ORDER = ["none", "low", "medium", "high", "critical"]
RISK_COLORS = {"none": "#898781", "low": "#0ca30c", "medium": "#fab219", "high": "#ec835a", "critical": "#d03b3b"}

# Sequential blue ramp (light -> dark) for magnitudes such as confusion-matrix counts.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def config_color(config: str) -> str:
    return series_colors()[CONFIG_ORDER.index(config) % 3]


def device_color(device: str) -> str:
    return series_colors()[DEVICE_ORDER.index(device) % 3]


def show(fig: go.Figure, container=st) -> None:
    container.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def grouped_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float | None]],
    colors: dict[str, str],
    *,
    percent: bool = False,
    height: int = 340,
    value_format: str = "{:.0f}",
) -> go.Figure:
    fig = go.Figure()
    for name, values in series.items():
        shown = [(v or 0) * (100 if percent else 1) for v in values]
        text = ["n/a" if v is None else (f"{v * 100:.0f}%" if percent else value_format.format(v)) for v in values]
        fig.add_bar(
            x=list(categories), y=shown, name=name, text=text, textposition="outside", cliponaxis=False,
            marker=dict(color=colors[name], cornerradius=4),
            hovertemplate=f"{name}<br>%{{x}}: %{{text}}<extra></extra>",
        )  # fmt: skip
    finish(fig, height=height, barmode="group")
    if percent:
        fig.update_yaxes(range=[0, 118], ticksuffix="%")
    return fig


def hbars(
    counts: dict[str, float],
    color: str | dict[str, str],
    *,
    percent: bool = False,
    order: list[str] | None = None,
    value_format: str = "{:,.0f}",
) -> go.Figure:
    keys = order if order else [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1])]
    keys = [k for k in keys if k in counts]
    values = [counts[k] for k in keys]
    colors = [color[k] for k in keys] if isinstance(color, dict) else color
    text = [f"{v * 100:.0f}%" if percent else value_format.format(v) for v in values]
    fig = go.Figure(
        go.Bar(
            x=[v * 100 if percent else v for v in values], y=keys, orientation="h", text=text, textposition="outside",
            cliponaxis=False, marker=dict(color=colors, cornerradius=4), hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )  # fmt: skip
    finish(fig, height=max(180, 38 * len(keys) + 40), showlegend=False)
    if percent:
        fig.update_xaxes(range=[0, 115], ticksuffix="%")
    elif values:
        fig.update_xaxes(range=[0, max(values) * 1.22 or 1])  # headroom for the outside value labels
    return fig


def grouped_hbars(categories: list[str], series: dict[str, list[float | None]], colors: dict[str, str]) -> go.Figure:
    """Horizontal grouped bars (recall per attack category, per configuration)."""
    fig = go.Figure()
    for name, values in series.items():
        fig.add_bar(
            y=categories, x=[(v or 0) * 100 for v in values], name=name, orientation="h",
            text=["n/a" if v is None else f"{v * 100:.0f}%" for v in values], textposition="outside", cliponaxis=False,
            marker=dict(color=colors[name], cornerradius=4), hovertemplate=f"{name}<br>%{{y}}: %{{text}}<extra></extra>",
        )  # fmt: skip
    finish(fig, height=max(260, 30 * len(categories) * len(series) + 60), barmode="group")
    fig.update_xaxes(range=[0, 118], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return fig


def confusion_matrix(cm: dict[str, int], title: str) -> go.Figure:
    """2x2 matrix: rows = actual, columns = predicted. Count encoded as sequential blue, always labelled."""
    z = [[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]]
    labels = [
        [f"TP<br><b>{cm['tp']}</b>", f"FN<br><b>{cm['fn']}</b>"],
        [f"FP<br><b>{cm['fp']}</b>", f"TN<br><b>{cm['tn']}</b>"],
    ]
    peak = max(1, max(max(r) for r in z))
    text_colors = [["#ffffff" if v / peak > 0.45 else "#0b0b0b" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=["Flagged", "Not flagged"], y=["Malicious", "Benign"],
            colorscale=[[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)], zmin=0, zmax=peak,
            showscale=False, xgap=2, ygap=2, hovertemplate="actual %{y}, %{x}: %{z}<extra></extra>",
        )
    )  # fmt: skip
    # Per-cell text colour keeps contrast on both light and dark cells.
    for i, row in enumerate(labels):
        for j, label in enumerate(row):
            fig.add_annotation(x=j, y=i, text=label, showarrow=False, font=dict(color=text_colors[i][j], size=14))
    finish(fig, height=240, title=dict(text=title, x=0, font=dict(size=13)), showlegend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=8))
    fig.update_yaxes(autorange="reversed", gridwidth=0, showgrid=False)
    return fig


def _jitter(case_id: str) -> float:
    """Deterministic vertical offset in [-0.3, 0.3] derived from the case id."""
    return (int(hashlib.sha1(case_id.encode()).hexdigest()[:6], 16) / 0xFFFFFF - 0.5) * 0.6


def risk_strip(cases: list[dict], block: float, warn: float, color: str) -> go.Figure:
    """Risk score per case on one axis, benign vs malicious by position, thresholds as reference lines."""
    fig = go.Figure()
    for base, label in ((0, "malicious"), (1, "benign")):
        pts = [c for c in cases if c["label"] == label]
        fig.add_scatter(
            x=[c["risk_score"] for c in pts], y=[base + _jitter(c["id"]) for c in pts], mode="markers", name=label,
            marker=dict(size=9, color=color, opacity=0.75 if label == "malicious" else 0.45,
                        symbol="circle" if label == "malicious" else "circle-open",
                        line=dict(width=2, color=color)),
            customdata=[[c["id"], c["category"], c["status"]] for c in pts],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>risk %{x:.2f} → %{customdata[2]}<extra></extra>",
        )  # fmt: skip
    ink = "#c3c2b7" if mode() == "dark" else "#52514e"
    for x, name in ((warn, "warn"), (block, "block")):
        fig.add_vline(x=x, line=dict(width=1, dash="dot", color=ink),
                      annotation=dict(text=f"{name} {x:.2f}", font=dict(size=11, color=ink)), annotation_position="top")  # fmt: skip
    finish(fig, height=240, showlegend=False)
    fig.update_xaxes(range=[-0.03, 1.05], title=None, showgrid=True)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["malicious", "benign"], range=[-0.5, 1.5], showgrid=False)
    fig.update_layout(margin=dict(l=8, r=8, t=28, b=8))
    return fig


def gauge_bar(used: float, total: float, label: str) -> go.Figure:
    """A single horizontal bar showing used vs total (e.g. VRAM)."""
    color = series_colors()[0]
    track = "#2c2c2a" if mode() == "dark" else "#e1e0d9"
    fig = go.Figure()
    fig.add_bar(x=[total], y=[label], orientation="h", marker=dict(color=track, cornerradius=4), hoverinfo="skip")
    fig.add_bar(x=[used], y=[label], orientation="h", marker=dict(color=color, cornerradius=4),
                hovertemplate=f"{label}: %{{x:,.0f}} MiB<extra></extra>")  # fmt: skip
    finish(fig, height=90, barmode="overlay", showlegend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(range=[0, total], showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
