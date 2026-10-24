"""Security dashboard: every statistic the gateway produces, read live from the API (nothing synthesised).

Tabs: Overview · Live traffic · Security benchmark · ML classifier · Inference & hardware.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sentinelai.ui import client
from sentinelai.ui.charts import (
    CONFIG_LABELS,
    CONFIG_ORDER,
    DEVICE_LABELS,
    DEVICE_ORDER,
    RISK_COLORS,
    RISK_ORDER,
    config_color,
    confusion_matrix,
    device_color,
    gauge_bar,
    grouped_bars,
    grouped_hbars,
    hbars,
    risk_strip,
    show,
)
from sentinelai.ui.style import STATUS_COLORS, STATUS_LABELS, finish, series_colors

_RATIO_COLS = {k: st.column_config.NumberColumn(k, format="%.3f") for k in ("precision", "recall", "F1", "FPR", "AUC")}
RANGES = {"All time": None, "Last hour": 3600, "Last 24 hours": 86400, "Last 7 days": 7 * 86400}
# Project targets from the specification.
TARGETS = [
    ("Attack recall", "≥ 85%", lambda m: m["attack_detection"]["recall"], lambda v: v is not None and v >= 0.85),
    ("Attack precision", "≥ 80%", lambda m: m["attack_detection"]["precision"], lambda v: v is not None and v >= 0.80),
    ("Benign false-positive rate", "≤ 10%", lambda m: m["attack_detection"]["false_positive_rate"], lambda v: v is not None and v <= 0.10),
    ("Secret leakage", "0", lambda m: m["secret_leakage"]["leaked"], lambda v: v == 0),
]  # fmt: skip


# ---------------------------------------------------------------------------------------- data
@st.cache_data(ttl=5, show_spinner=False)
def _get(path: str, params: tuple[tuple[str, Any], ...] = ()) -> Any:
    return client.get(path, params=dict(params) or None)


def _pct(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.1f} ms"


def _latest_benchmark(runs: list[dict]) -> dict[str, dict[str, dict]]:
    """{split: {config: run}} for the newest benchmark batch."""
    if not runs:
        return {}
    newest = runs[0]["ts"]
    out: dict[str, dict[str, dict]] = {}
    for run in runs:
        if run["ts"] != newest:
            continue
        split, _, config = run["label"].partition("/")
        out.setdefault(split, {})[config] = run
    return out


def _latest_by_label(runs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for run in runs:  # newest first
        out.setdefault(run["label"], run)
    return out


def _live_config(label: str) -> str:
    """Live runs use the default policy, which includes the classifier whenever it is trained."""
    return "baseline" if label == "baseline" else "protected_ml"


def _live_name(label: str) -> str:
    return CONFIG_LABELS[_live_config(label)]


def _card(title: str, caption: str | None = None):
    box = st.container(border=True)
    box.markdown(f"**{title}**")
    if caption:
        box.caption(caption)
    return box


# ------------------------------------------------------------------------------------- header
def _status_strip(status: dict | None) -> None:
    cols = st.columns(4)
    if status is None:
        cols[0].error("Gateway offline", icon=":material/cloud_off:")
        return
    model, clf = status["model"], status["classifier"]
    gpus = status["hardware"]["nvidia_gpus"]
    cols[0].markdown(f":material/shield: **Gateway** v{status['version']}  \n"
                     f":green[● online] · {len(status['detectors'])} detectors")  # fmt: skip
    avail = ":green[● available]" if model["available"] else ":red[● unavailable]"
    cols[1].markdown(f":material/smart_toy: **Model** `{model['model_id']}`  \n{avail} · {model['backend']} · "
                     f"device {model.get('device') or 'not loaded'}")  # fmt: skip
    if gpus:
        g = gpus[0]
        cols[2].markdown(f":material/memory: **{g['name'].replace('NVIDIA GeForce ', '')}**  \n"
                         f"{g['memory_used_mib']:,}/{g['memory_total_mib']:,} MiB · {g.get('utilization_pct', '—')}% util · "
                         f"{g.get('temperature_c', '—')} °C")  # fmt: skip
    else:
        cols[2].markdown(":material/memory: **No GPU**  \nCPU inference only")
    state = "enabled" if clf["enabled"] else ("available, off" if clf["available"] else "not trained")
    cols[3].markdown(
        f":material/psychology: **ML classifier**  \n{state}" + (f" · `{clf['version']}`" if clf.get("version") else "")
    )


# ------------------------------------------------------------------------------------ overview
def _overview(status: dict | None, metrics: dict, bench: dict, inference: dict[str, dict], live: dict[str, dict]) -> None:
    st.subheader("Live traffic")
    s = metrics["by_status"]
    total = metrics["total_requests"]
    blocked = s.get("blocked", 0) + s.get("output_blocked", 0) + s.get("review", 0)
    k = st.columns(5)
    k[0].metric("Requests", f"{total:,}", border=True)
    k[1].metric("Blocked / held", f"{blocked:,}", f"{blocked / total:.0%} of traffic" if total else None,
                delta_color="off", delta_arrow="off", border=True)  # fmt: skip
    k[2].metric("Redacted", f"{s.get('redacted', 0):,}", border=True)
    k[3].metric("High-risk events", f"{metrics['high_risk_events']:,}", border=True)
    k[4].metric("Gateway latency p50", _ms(metrics["latency_ms"]["gateway"]["p50"]), border=True)

    holdout = bench.get("holdout", {})
    st.subheader("Security effectiveness (holdout split, never used for tuning)")
    if not holdout:
        st.info(
            "No benchmark stored yet. Run `uv run python scripts/evaluate_security.py` or use the Security benchmark tab."
        )
    else:
        best = "protected_ml" if "protected_ml" in holdout else "protected"
        m, rules, base = (
            holdout[best]["metrics"],
            holdout.get("protected", holdout[best])["metrics"],
            holdout["baseline"]["metrics"],
        )
        k = st.columns(5)
        k[0].metric("Attack recall", _pct(m["attack_detection"]["recall"]),
                    f"{(m['attack_detection']['recall'] - rules['attack_detection']['recall']) * 100:+.0f} pts vs rules only"
                    if best == "protected_ml" else None, border=True)  # fmt: skip
        k[1].metric("Precision", _pct(m["attack_detection"]["precision"]), border=True)
        k[2].metric("Benign FPR", _pct(m["attack_detection"]["false_positive_rate"]), border=True)
        k[3].metric("Secrets leaked", f"{m['secret_leakage']['leaked']}/{m['secret_leakage']['n']}",
                    f"baseline {base['secret_leakage']['leaked']}/{base['secret_leakage']['n']}", delta_color="off", delta_arrow="off", border=True)  # fmt: skip
        k[4].metric("PII reaching model", f"{m['pii_exposure']['reached_model']}/{m['pii_exposure']['n']}",
                    f"baseline {base['pii_exposure']['reached_model']}/{base['pii_exposure']['n']}", delta_color="off", delta_arrow="off", border=True)  # fmt: skip

        left, right = st.columns([3, 2])
        with left:
            box = _card("Controls disabled vs enabled", "Holdout split · deterministic vulnerable mock model")
            configs = [c for c in CONFIG_ORDER if c in holdout]
            names = ["Attack recall", "Precision", "Benign FPR", "Secret leakage", "PII to model"]
            series = {CONFIG_LABELS[c]: [holdout[c]["metrics"]["attack_detection"]["recall"],
                                         holdout[c]["metrics"]["attack_detection"]["precision"],
                                         holdout[c]["metrics"]["attack_detection"]["false_positive_rate"],
                                         holdout[c]["metrics"]["secret_leakage"]["rate"],
                                         holdout[c]["metrics"]["pii_exposure"]["reached_model_rate"]] for c in configs}  # fmt: skip
            show(grouped_bars(names, series, {CONFIG_LABELS[c]: config_color(c) for c in configs}, percent=True), box)
        with right:
            box = _card("Project targets", "Measured on the holdout split · ✅ met · ❌ missed")
            rows = []
            for name, target, get, ok in TARGETS:
                row = {"Metric": name, "Target": target}
                for config in ("protected", "protected_ml"):
                    if config in holdout:
                        v = get(holdout[config]["metrics"])
                        shown = str(v) if name == "Secret leakage" else _pct(v)
                        row[CONFIG_LABELS[config]] = f"{'✅' if ok(v) else '❌'} {shown}"
                rows.append(row)
            box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        box = _card("Real local LLM: secrets leaked to the user", "Live holdout attacks through Ollama")
        if live:
            labels = [x for x in ("baseline", "protected") if x in live]
            first = live[labels[0]]
            names = {x: _live_name(x) for x in labels}
            box.caption(f"`{first['model']}` on {first.get('device') or 'n/a'} · {first['metrics']['n_malicious']} attacks + "
                        f"{first['metrics']['n_benign']} benign · {first['ts']}")  # fmt: skip
            series = {names[x]: [live[x]["metrics"]["returned_secret"], live[x]["metrics"]["blocked_before_model"]]
                      for x in labels}  # fmt: skip
            show(grouped_bars(["Secrets returned to user", "Blocked before the model"], series,
                              {names[x]: config_color(_live_config(x)) for x in labels}, height=280), box)  # fmt: skip
        else:
            box.info("Run `uv run python scripts/evaluate_security.py --live 20` with Ollama running.")
    with c2:
        box = _card("Inference throughput: GPU vs CPU", "Tokens per second (p50), full pipeline enabled")
        devices = [d for d in DEVICE_ORDER if d in inference]
        if devices:
            show(hbars({DEVICE_LABELS[d]: inference[d]["metrics"]["tokens_per_s_p50"] for d in devices},
                       {DEVICE_LABELS[d]: device_color(d) for d in devices}, value_format="{:.0f} tok/s"), box)  # fmt: skip
            if "gpu" in inference and "cpu" in inference:
                speed = inference["gpu"]["metrics"]["tokens_per_s_p50"] / inference["cpu"]["metrics"]["tokens_per_s_p50"]
                box.caption(f"GPU is **{speed:.1f}×** faster than CPU on `{inference['gpu']['model']}`.")
        else:
            box.info("Run `uv run python scripts/benchmark_inference.py` with Ollama running.")


# -------------------------------------------------------------------------------- live traffic
def _traffic() -> None:
    top = st.columns([2, 2, 1])
    window = top[0].selectbox("Time range", list(RANGES), key="traffic_range")
    auto = top[2].toggle("Auto-refresh (5 s)", key="traffic_auto")
    # A fragment re-runs only this tab on its own timer, leaving the rest of the page untouched.
    st.fragment(_traffic_body, run_every=5 if auto else None)(window)


def _traffic_body(window: str) -> None:
    since = time.time() - RANGES[window] if RANGES[window] else None
    try:
        m = _get("/api/v1/metrics", (("since", since),) if since else ())
    except client.ApiError as exc:
        st.error(str(exc))
        return
    s = m["by_status"]
    total = m["total_requests"]
    k = st.columns(6)
    k[0].metric("Requests", total, border=True)
    k[1].metric("Allowed", s.get("allowed", 0) + s.get("allowed_with_warning", 0), border=True)
    k[2].metric("Blocked (input)", s.get("blocked", 0) + s.get("review", 0), border=True)
    k[3].metric("Blocked (output)", s.get("output_blocked", 0), border=True)
    k[4].metric("Redacted", s.get("redacted", 0), border=True)
    k[5].metric("Rejected / errors", s.get("rejected", 0) + s.get("inference_error", 0), border=True)
    if not total:
        st.info("No traffic in this range. Send prompts from the **Chat** page or `POST /api/v1/chat`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            box = _card("Outcome by status")
            order = [x for x in STATUS_LABELS if x in s]
            show(hbars({STATUS_LABELS[x]: s[x] for x in order}, {STATUS_LABELS[x]: STATUS_COLORS[x] for x in order},
                       order=[STATUS_LABELS[x] for x in order]), box)  # fmt: skip
        with c2:
            box = _card("Requests over time", "Hourly buckets, stacked by outcome")
            tl = pd.DataFrame(m["timeline"])
            fig = go.Figure()
            for status in [x for x in STATUS_LABELS if x in set(tl["status"])]:
                part = tl[tl["status"] == status]
                fig.add_bar(x=part["bucket"], y=part["count"], name=STATUS_LABELS[status], marker=dict(color=STATUS_COLORS[status]),
                            hovertemplate=f"{STATUS_LABELS[status]}: %{{y}}<extra>%{{x}}</extra>")  # fmt: skip
            show(finish(fig, barmode="stack", height=300), box)

        c1, c2, c3 = st.columns(3)
        with c1:
            box = _card("Risk level")
            risk = {r: m["by_risk_level"][r] for r in RISK_ORDER if r in m["by_risk_level"]}
            show(hbars(risk, RISK_COLORS, order=[r for r in RISK_ORDER if r in risk]), box)
        with c2:
            box = _card("Detection categories", "Events containing each finding category")
            if m["categories"]:
                show(hbars(m["categories"], series_colors()[0]), box)
            else:
                box.caption("No detections.")
        with c3:
            box = _card("PII entities detected")
            if m["pii_entities"]:
                show(hbars(m["pii_entities"], series_colors()[0]), box)
            else:
                box.caption("No PII detected.")

        lat = m["latency_ms"]
        box = _card("Latency", f"Gateway = security processing excluding the model (n={lat['gateway']['n']}); "
                               f"model = inference only (n={lat['inference']['n']})")  # fmt: skip
        cols = box.columns(4)
        cols[0].metric("Gateway p50", _ms(lat["gateway"]["p50"]))
        cols[1].metric("Gateway p95", _ms(lat["gateway"]["p95"]))
        cols[2].metric("Model p50", _ms(lat["inference"]["p50"]))
        cols[3].metric("Model p95", _ms(lat["inference"]["p95"]))

    box = _card("Recent high-risk events")
    try:
        recent = []
        for level in ("critical", "high"):
            params = (("risk_level", level), ("limit", 15)) + ((("since", since),) if since else ())
            recent += _get("/api/v1/events", params)["items"]
    except client.ApiError as exc:
        box.error(str(exc))
        recent = []
    if recent:
        df = pd.DataFrame(sorted(recent, key=lambda e: e["ts_epoch"], reverse=True)[:15])
        box.dataframe(df[["ts", "status", "risk_level", "risk_score", "categories", "input_action", "output_action", "gateway_ms"]],
                      hide_index=True, width="stretch",
                      column_config={"risk_score": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f")})  # fmt: skip
    else:
        box.caption("None in this range.")


# ---------------------------------------------------------------------------------- benchmark
def _benchmark(bench: dict, status: dict | None) -> None:
    top = st.columns([2, 3, 2])
    if not bench:
        st.info("No benchmark stored yet.")
    splits = [x for x in ("holdout", "dev") if x in bench]
    split = (
        top[0].segmented_control("Split", splits, default=splits[0] if splits else None, key="bench_split")
        if splits
        else None
    )
    token = st.session_state.get("admin_token")
    if top[2].button("Run benchmark now", icon=":material/play_arrow:", disabled=not token, width="stretch",
                     help=None if token else "Enter the admin token on the Settings page first"):  # fmt: skip
        with st.spinner("Running all splits and configurations…"):
            try:
                client.post("/api/v1/evaluate", {"include_classifier": True}, admin_token=token, timeout=600)
                _get.clear()
                st.rerun()
            except client.ApiError as exc:
                st.error(str(exc))
    if not split:
        return
    runs = bench[split]
    configs = [c for c in CONFIG_ORDER if c in runs]
    first = runs[configs[0]]
    top[1].caption(f"Dataset `{first['dataset_version']}` · {first['ts']}  \n"
                   + ("Written after the rules were frozen; never used for tuning." if split == "holdout"
                      else "Rules were debugged against this split: optimistic numbers."))  # fmt: skip

    k = st.columns(len(configs))
    for col, c in zip(k, configs, strict=True):
        m = runs[c]["metrics"]
        a = m["attack_detection"]
        box = col.container(border=True)
        box.markdown(f"**{CONFIG_LABELS[c]}**")
        sub = box.columns(3)
        sub[0].metric("Recall", _pct(a["recall"], 0))
        sub[1].metric("Precision", _pct(a["precision"], 0))
        sub[2].metric("FPR", _pct(a["false_positive_rate"], 0))
        box.caption(f"F1 {a['f1'] if a['f1'] is not None else '—'} · block recall {_pct(m['attack_blocking']['recall'], 0)} · "
                    f"secrets leaked {m['secret_leakage']['leaked']}/{m['secret_leakage']['n']} · "
                    f"PII to model {m['pii_exposure']['reached_model']}/{m['pii_exposure']['n']}")  # fmt: skip

    c1, c2 = st.columns([3, 2])
    with c1:
        box = _card("Recall by attack category", "Share of attacks in each category that the gateway flagged")
        cats = list(runs[configs[-1]]["metrics"]["per_attack_category"])
        active = [c for c in configs if c != "baseline"]
        series = {
            CONFIG_LABELS[c]: [runs[c]["metrics"]["per_attack_category"][cat]["recall"] for cat in cats] for c in active
        }
        labels = [
            f"{cat.replace('_', ' ')} (n={runs[active[0]]['metrics']['per_attack_category'][cat]['n']})" for cat in cats
        ]
        show(grouped_hbars(labels, series, {CONFIG_LABELS[c]: config_color(c) for c in active}), box)
    with c2:
        box = _card("Confusion matrices", "Rows: actual label · columns: gateway decision")
        for c in [c for c in configs if c != "baseline"]:
            show(confusion_matrix(runs[c]["metrics"]["attack_detection"]["confusion_matrix"], CONFIG_LABELS[c]), box)

    box = _card(
        "Risk-score separation",
        "Each dot is one prompt, jittered vertically so identical scores stay visible. Attacks should sit right of the "
        "block line. Benign dots at 0.40 are PII prompts: redacted, not flagged as attacks.",
    )
    policy = (status or {}).get("policy", {})
    for c in [c for c in configs if c != "baseline"]:
        cases = runs[c]["metrics"].get("cases")
        if not cases:
            box.caption("Per-case scores are stored from the next benchmark run onwards.")
            break
        box.markdown(f"<span style='font-size:0.85rem'>{CONFIG_LABELS[c]}</span>", unsafe_allow_html=True)
        show(risk_strip(cases, policy.get("injection_block_threshold", 0.7), policy.get("injection_warn_threshold", 0.35),
                        config_color(c)), box)  # fmt: skip

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c1:
        box = _card("Output suite", "Benign prompt, scripted leaky model response")
        rows = [{"configuration": CONFIG_LABELS[c], "expected action matched": _pct(o["expected_action_accuracy"], 0),
                 "secret cases leaked": f"{o['secret_leaks']}/{o['secret_cases']}"}
                for c in configs if (o := runs[c]["metrics"]["output_suite"])["n"]]  # fmt: skip
        box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch") if rows else box.caption(
            "Not part of this split."
        )
    with c2:
        box = _card("Edge cases & operations")
        rows = [{"configuration": CONFIG_LABELS[c],
                 "edge cases": f"{e['passed']}/{e['n']}" if (e := runs[c]["metrics"]["edge_cases"])["n"] else "—",
                 "blocked → model": runs[c]["metrics"]["operational"]["blocked_but_model_invoked"],
                 "benign answered": _pct(runs[c]["metrics"]["benign_answer_rate"], 0)} for c in configs]  # fmt: skip
        box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with c3:
        box = _card("Gateway latency in benchmark")
        rows = [{"configuration": CONFIG_LABELS[c], "p50": _ms(runs[c]["metrics"]["latency_ms"]["gateway_p50"]),
                 "p95": _ms(runs[c]["metrics"]["latency_ms"]["gateway_p95"])} for c in configs]  # fmt: skip
        box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Misclassified and missed cases"):
        for c in [c for c in configs if c != "baseline"]:
            m = runs[c]["metrics"]
            st.markdown(f"**{CONFIG_LABELS[c]}** — false negatives `{m['false_negatives'] or 'none'}` · false positives "
                        f"`{m['false_positives'] or 'none'}` · PII reaching model `{m['pii_exposure']['reached_model_ids'] or 'none'}` · "
                        f"spurious redactions `{m.get('spurious_redaction', {}).get('ids') or 'none'}`")  # fmt: skip


# ---------------------------------------------------------------------------------- classifier
def _classifier(status: dict | None) -> None:
    clf = (status or {}).get("classifier", {})
    if not clf.get("available"):
        st.info(
            f"No trained classifier. {clf.get('detail') or ''}  \nTrain it with `uv run python scripts/train_classifier.py`."
        )
        return
    ds = clf.get("dataset") or {}
    k = st.columns(4)
    k[0].metric("Block threshold", f"{clf['thresholds']['block']:.2f}", border=True)
    k[1].metric("Warn threshold", f"{clf['thresholds']['warn']:.2f}", border=True)
    k[2].metric("Test ROC-AUC", f"{clf['test_metrics']['roc_auc']:.3f}", border=True)
    sp = clf.get("splits") or {}
    split_note = " · ".join(f"{v['rows']} {k}" for k, v in sp.items())
    k[3].metric("Corpus rows", f"{ds.get('rows_after_dedup', '—'):,}",
                f"{split_note} · {ds.get('dropped_near_duplicates', 0)} near-duplicates removed",
                delta_color="off", delta_arrow="off", border=True)  # fmt: skip
    st.caption(f"`{clf['version']}` · {clf.get('algorithm', '')} · trained {clf.get('trained_at', '—')} · "
               f"{'enabled' if clf['enabled'] else 'disabled'} in the active policy")  # fmt: skip

    sets = {"Validation (grouped CV)": clf.get("validation"), "Test families (held out)": clf["test_metrics"]}
    for name, d in (clf.get("external_evaluation") or {}).items():
        sets[f"Hand-written {'dev' if name == 'evaluation' else name}"] = d
    sets = {k2: v for k2, v in sets.items() if v}
    c1, c2 = st.columns([3, 2])
    with c1:
        box = _card("Classifier alone, at the warn threshold", "Before combining with rules")
        palette = series_colors()
        series = {n: [d["at_warn"]["precision"], d["at_warn"]["recall"], d["at_warn"]["f1"], d["at_warn"]["false_positive_rate"]]
                  for n, d in sets.items()}  # fmt: skip
        # Three colour slots validate all-pairs, so the chart shows the three held-out sets; validation is in the table.
        shown = {n: v for n, v in series.items() if not n.startswith("Validation")}
        colors = {n: palette[i % 3] for i, n in enumerate(shown)}
        show(grouped_bars(["Precision", "Recall", "F1", "FPR"], shown, colors, percent=True), box)
        rows = []
        for n, d in sets.items():
            for which in ("at_block", "at_warn"):
                x = d[which]
                cm = x["confusion_matrix"]
                rows.append({"data": n, "threshold": f"{which[3:]} {x['threshold']}",
                             "TP/FP/TN/FN": f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}", "precision": x["precision"], "recall": x["recall"],
                             "F1": x["f1"], "FPR": x["false_positive_rate"], "AUC": d.get("roc_auc")})  # fmt: skip
        box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=_RATIO_COLS)
    with c2:
        box = _card("Test families, at the block threshold")
        show(confusion_matrix(clf["test_metrics"]["at_block"]["confusion_matrix"], "Held-out template families"), box)
        splits = clf.get("splits") or {}
        if splits:
            box.markdown("**Family split** (whole families per split, so test styles are unseen)")
            for name, sp in splits.items():
                fams = ", ".join(f.split(":", 1)[1] for f in sp["families"])
                box.caption(f"**{name}** · {sp['rows']} rows ({sp['malicious']} malicious): {fams}")


# ----------------------------------------------------------------------------------- inference
def _inference(status: dict | None, inference_runs: list[dict], live: dict[str, dict]) -> None:
    gpus = (status or {}).get("hardware", {}).get("nvidia_gpus", [])
    c1, c2 = st.columns([2, 3])
    with c1:
        box = _card("GPU (live)")
        if gpus:
            g = gpus[0]
            box.markdown(f"{g['name']} · driver {g['driver']}")
            used, total = g["memory_used_mib"] or 0, g["memory_total_mib"] or 1
            box.caption(f"VRAM {used:,} / {total:,} MiB ({used / total:.0%})")
            show(gauge_bar(used, total, "VRAM"), box)
            m = box.columns(2)
            m[0].metric("Utilisation", f"{g.get('utilization_pct', '—')}%")
            m[1].metric("Temperature", f"{g.get('temperature_c', '—')} °C")
        else:
            box.info("No NVIDIA GPU detected; local inference runs on CPU.")
        model = (status or {}).get("model", {})
        box.caption(f"Active model `{model.get('model_id', '—')}` ({model.get('backend', '—')}) · placement "
                    f"{model.get('device') or 'not loaded'} · {model.get('detail', '')}")  # fmt: skip

    latest = _latest_by_label([r for r in inference_runs if r["kind"] == "inference"])
    devices = [d for d in DEVICE_ORDER if d in latest]
    with c2:
        box = _card("GPU vs CPU", "Same prompts through the full security pipeline; first request cold-loads the model")
        if not devices:
            box.info("Run `uv run python scripts/benchmark_inference.py` with Ollama running.")
        else:
            k = box.columns(len(devices) + (1 if {"gpu", "cpu"} <= set(devices) else 0))
            for col, d in zip(k, devices, strict=False):
                mm = latest[d]["metrics"]
                col.metric(f"{DEVICE_LABELS[d]} tokens/s", f"{mm['tokens_per_s_p50']:.0f}", f"placement: {latest[d]['device']}",
                           delta_color="off", delta_arrow="off")  # fmt: skip
            if {"gpu", "cpu"} <= set(devices):
                speed = latest["gpu"]["metrics"]["tokens_per_s_p50"] / latest["cpu"]["metrics"]["tokens_per_s_p50"]
                k[-1].metric("GPU speed-up", f"{speed:.1f}×")
            colors = {DEVICE_LABELS[d]: device_color(d) for d in devices}
            show(grouped_bars(["Inference p50", "Inference p95", "Cold first request"],
                              {DEVICE_LABELS[d]: [latest[d]["metrics"]["inference_p50_ms"], latest[d]["metrics"]["inference_p95_ms"],
                                                  latest[d]["metrics"]["cold_request_ms"]] for d in devices},
                              colors, height=300, value_format="{:,.0f} ms"), box)  # fmt: skip
    if devices:
        rows = [{"device": DEVICE_LABELS[d], "placement": latest[d]["device"], "model": latest[d]["model"],
                 "warm requests": latest[d]["metrics"]["n_warm"], "tokens/s p50": round(latest[d]["metrics"]["tokens_per_s_p50"], 1),
                 "inference p50": _ms(latest[d]["metrics"]["inference_p50_ms"]), "inference p95": _ms(latest[d]["metrics"]["inference_p95_ms"]),
                 "gateway p50": _ms(latest[d]["metrics"]["gateway_p50_ms"]), "gateway p95": _ms(latest[d]["metrics"]["gateway_p95_ms"]),
                 "VRAM for model": f"{latest[d]['metrics']['vram_model_mib']} MiB" if latest[d]["metrics"]["vram_model_mib"] is not None else "—",
                 "run at": latest[d]["ts"][:16].replace("T", " ")} for d in devices]  # fmt: skip
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Gateway time is the security pipeline (Presidio, rules, classifier), which always runs on CPU; "
                   "the device only changes model time.")  # fmt: skip

    box = _card("Live security run on the real model", "Holdout prompts sent to the local LLM with controls off vs on")
    if live:
        labels = [x for x in ("baseline", "protected") if x in live]
        first = live[labels[0]]
        box.caption(f"`{first['model']}` · {first['metrics']['n_malicious']} attacks + {first['metrics']['n_benign']} benign "
                    f"prompts · {first['ts'][:16].replace('T', ' ')} UTC")  # fmt: skip
        rows = [{"configuration": _live_name(x), "device": live[x].get("device"),
                 "model output contained a secret": live[x]["metrics"]["model_leaked_secret"],
                 "secret returned to user": live[x]["metrics"]["returned_secret"],
                 "blocked before model": live[x]["metrics"]["blocked_before_model"],
                 "output blocked": live[x]["metrics"]["output_blocked"],
                 "model p50": _ms(live[x]["metrics"]["inference_p50_ms"])} for x in labels]  # fmt: skip
        box.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        box.caption(
            "Real LLM output varies with hardware and drivers; treat these as indicative. The mock-model benchmark is the reproducible one."
        )
    else:
        box.info("Run `uv run python scripts/evaluate_security.py --live 20` with Ollama running.")


# --------------------------------------------------------------------------------------- page
def render() -> None:
    head = st.columns([5, 1])
    head[0].title("Security dashboard")
    if head[1].button("Refresh", icon=":material/refresh:", width="stretch"):
        _get.clear()
    try:
        status = _get("/api/v1/status")
    except client.ApiError as exc:
        st.error(str(exc))
        return
    _status_strip(status)
    try:
        metrics = _get("/api/v1/metrics")
        bench = _latest_benchmark(_get("/api/v1/evaluations", (("limit", 100),)))
        inference_runs = _get("/api/v1/inference-runs", (("limit", 100),))
    except client.ApiError as exc:
        st.error(str(exc))
        return
    live = _latest_by_label([r for r in inference_runs if r["kind"] == "live_security"])
    inference = _latest_by_label([r for r in inference_runs if r["kind"] == "inference"])

    tabs = st.tabs([":material/dashboard: Overview", ":material/timeline: Live traffic", ":material/science: Security benchmark",
                    ":material/psychology: ML classifier", ":material/memory: Inference & hardware"])  # fmt: skip
    with tabs[0]:
        _overview(status, metrics, bench, inference, live)
    with tabs[1]:
        _traffic()
    with tabs[2]:
        _benchmark(bench, status)
    with tabs[3]:
        _classifier(status)
    with tabs[4]:
        _inference(status, inference_runs, live)
