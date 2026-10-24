"""Inspect the local environment and report what SentinelAI can use.

uv run python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.util
import json
import sys

import httpx

from sentinelai.config import PROJECT_ROOT, get_settings
from sentinelai.detectors.ml_classifier import ClassifierUnavailable, MlClassifier
from sentinelai.environment import summarize_environment


def check(label: str, ok: bool, detail: str = "", hint: str = "") -> bool:
    print(f"[{'OK' if ok else '--'}] {label}{': ' + detail if detail else ''}")
    if not ok and hint:
        print(f"      -> {hint}")
    return ok


def main() -> int:
    s = get_settings()
    env = summarize_environment(PROJECT_ROOT)
    print(json.dumps(env, indent=2))
    print()
    check("Python >= 3.12", sys.version_info >= (3, 12), env["python"])
    check("Running inside the project venv", sys.prefix != sys.base_prefix, sys.prefix, "use `uv run ...`")
    spacy_ok = importlib.util.find_spec(s.spacy_model) is not None
    check(f"spaCy model {s.spacy_model}", spacy_ok, hint="run `uv sync` (the model is a pinned dependency)")
    gpus = env["nvidia_gpus"]
    check(
        "NVIDIA GPU",
        bool(gpus),
        ", ".join(f"{g['name']} {g['memory_total_mib']} MiB" for g in gpus) or env["nvidia_smi"],
        "optional: local LLM inference falls back to CPU (set SENTINEL_INFERENCE_DEVICE=cpu); "
        "a 'Driver/library version mismatch' after a driver update is fixed by rebooting",
    )
    try:
        MlClassifier.load(s.classifier_dir)
        check("ML classifier artifact", True, str(s.classifier_dir))
    except ClassifierUnavailable as exc:
        check("ML classifier artifact", False, str(exc), "optional: `uv run python scripts/train_classifier.py`")
    try:
        with httpx.Client(base_url=s.model_base_url, timeout=3, trust_env=False) as c:
            version = c.get("/api/version").json()["version"]
            models = sorted(m["name"] for m in c.get("/api/tags").json().get("models", []))
        check("Ollama server", True, f"v{version} at {s.model_base_url}")
        check(
            f"Model {s.model_name} installed",
            s.model_name in models,
            ", ".join(models),
            f"optional: `ollama pull {s.model_name}` (~1 GB); the mock model works without it",
        )
    except (httpx.HTTPError, ValueError, KeyError):
        check(
            "Ollama server",
            False,
            s.model_base_url,
            "optional: install Ollama for real local inference; mock model works without it",
        )
    print(f"\nConfigured backend: {s.model_backend} (device setting: {s.inference_device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
