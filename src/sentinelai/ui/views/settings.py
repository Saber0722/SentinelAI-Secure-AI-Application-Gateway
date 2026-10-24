"""Model selection and security policy. Changes go through the authenticated admin API."""

from __future__ import annotations

import streamlit as st

from sentinelai.ui import client

ACTIONS = ["ALLOW", "REDACT", "BLOCK", "REVIEW"]


def render() -> None:
    st.title("Settings")
    try:
        status = client.get("/api/v1/status")
    except client.ApiError as exc:
        st.error(str(exc))
        return

    model, clf = status["model"], status["classifier"]
    st.subheader("Runtime status")
    c = st.columns(3)
    c[0].metric("Model", f"{model['backend']}")
    c[0].caption(f"`{model['model_id']}` · {'available' if model['available'] else 'unavailable'}")
    c[1].metric("Inference device", model.get("device") or "n/a")
    gpus = status["hardware"]["nvidia_gpus"]
    c[1].caption(
        ", ".join(f"{g['name']} ({g['memory_used_mib']}/{g['memory_total_mib']} MiB)" for g in gpus)
        or "No NVIDIA GPU detected"
    )
    c[2].metric("ML classifier", "enabled" if clf["enabled"] else ("available" if clf["available"] else "not trained"))
    c[2].caption(clf.get("version") or clf.get("detail") or "")
    st.caption(model["detail"])
    with st.expander("Active policy (read-only view)"):
        st.json(status["policy"])
        st.caption("Detectors: " + ", ".join(f"`{d}`" for d in status["detectors"]))

    st.subheader("Change settings")
    if not status["admin_enabled"]:
        st.warning(
            "The admin API is disabled. Set `SENTINEL_ADMIN_TOKEN` in `.env` and restart the API to edit settings here."
        )
        return
    token = st.text_input(
        "Admin token",
        type="password",
        value=st.session_state.get("admin_token", ""),
        help="Sent as X-Admin-Token. The gateway, not this page, enforces every policy.",
    )
    st.session_state.admin_token = token

    policy = status["policy"]
    with st.form("policy"):
        st.markdown("**Security policy**")
        a, b = st.columns(2)
        rules = a.toggle("Injection / jailbreak rules", policy["input_rules_enabled"])
        classifier = a.toggle("ML classifier", policy["classifier_enabled"], disabled=not clf["available"])
        output = a.toggle("Output scanning", policy["output_scanning_enabled"])
        block = b.slider("Block threshold", 0.0, 1.0, policy["injection_block_threshold"], 0.05)
        warn = b.slider("Warn threshold", 0.0, 1.0, policy["injection_warn_threshold"], 0.05)
        p1, p2, p3 = st.columns(3)
        pii_in = p1.selectbox("PII in prompts", ACTIONS, ACTIONS.index(policy["pii_input_action"]))
        pii_out = p2.selectbox("PII in responses", ACTIONS, ACTIONS.index(policy["pii_output_action"]))
        secret_out = p3.selectbox(
            "Credentials in responses", ["BLOCK", "REDACT"], ["BLOCK", "REDACT"].index(policy["output_secret_action"])
        )
        entities = st.multiselect("PII entities", status["supported_pii_entities"], policy["pii_entities"])
        if st.form_submit_button("Apply policy", type="primary"):
            patch = {
                "input_rules_enabled": rules,
                "classifier_enabled": classifier,
                "output_scanning_enabled": output,
                "injection_block_threshold": block,
                "injection_warn_threshold": warn,
                "pii_input_action": pii_in,
                "pii_output_action": pii_out,
                "output_secret_action": secret_out,
                "pii_entities": entities,
            }
            try:
                client.put("/api/v1/settings", {"policy": patch}, admin_token=token)
                st.success("Policy applied and recorded in the audit log.")
            except client.ApiError as exc:
                st.error(str(exc))

    with st.form("model"):
        st.markdown("**Model**")
        try:
            available = client.get("/api/v1/models")["available"]
        except client.ApiError:
            available = []
        m1, m2, m3 = st.columns(3)
        backend = m1.selectbox(
            "Backend",
            ["mock", "ollama", "openai_compatible"],
            ["mock", "ollama", "openai_compatible"].index(model["backend"])
            if model["backend"] in ("mock", "ollama", "openai_compatible")
            else 0,
        )
        name = m2.text_input(
            "Model name",
            model["model_id"] if backend != "mock" else "qwen2.5:1.5b",
            help=("Installed: " + ", ".join(available)) if available else None,
        )
        device = m3.selectbox("Device", ["auto", "gpu", "cpu"], help="auto = GPU if available, CPU fallback on GPU failure")
        if st.form_submit_button("Switch model"):
            try:
                client.put(
                    "/api/v1/settings", {"model": {"backend": backend, "name": name, "device": device}}, admin_token=token
                )
                st.success("Model switched.")
            except client.ApiError as exc:
                st.error(str(exc))
