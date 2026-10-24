"""Measure real local-LLM inference through the gateway on GPU and on CPU.

    uv run python scripts/benchmark_inference.py                  # gpu (if detected) + cpu, 20 prompts each
    uv run python scripts/benchmark_inference.py --devices cpu --n 10

For each device the model is unloaded first, so the first request measures a cold load with fresh device
placement. The remaining requests are warm. Benign evaluation prompts are used, so every request reaches
the model. Outputs: results/inference_benchmark.json, docs/inference_report.md, and one row per device in
the SQLite database (shown on the dashboard).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sentinelai.audit import AuditLogger, NullAuditLogger, percentile
from sentinelai.config import SecurityPolicy, get_settings
from sentinelai.detectors.ml_classifier import ClassifierUnavailable, MlClassifier
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.environment import detect_nvidia_gpus, nvidia_smi_diagnostic
from sentinelai.evaluation.dataset import load_dataset
from sentinelai.gateway.pipeline import SecurityGateway, Status
from sentinelai.models import GenerationConfig
from sentinelai.models.local_model import OllamaModel
from sentinelai.storage.database import Database

ROOT = Path(__file__).resolve().parents[1]


def gpu_memory_used() -> int | None:
    detect_nvidia_gpus.cache_clear()
    gpus = detect_nvidia_gpus()
    return gpus[0]["memory_used_mib"] if gpus else None


def run_device(device: str, prompts: list[str], gateway: SecurityGateway, model: OllamaModel) -> dict:
    model.unload()
    time.sleep(1.0)
    vram_before = gpu_memory_used()
    t0 = time.perf_counter()
    cold = gateway.chat(prompts[0])
    cold_ms = (time.perf_counter() - t0) * 1000
    vram_loaded = gpu_memory_used()
    rows = []
    for text in prompts[1:]:
        r = gateway.chat(text)
        if r.status is Status.INFERENCE_ERROR:
            rows.append({"error": r.error})
            continue
        tps = r.output_tokens / (r.inference_ms / 1000) if r.output_tokens and r.inference_ms else None
        rows.append(
            {
                "inference_ms": r.inference_ms,
                "gateway_ms": r.gateway_ms,
                "output_tokens": r.output_tokens,
                "tokens_per_s": tps,
                "status": r.status.value,
            }
        )
        print(f"  {device}: {r.inference_ms:7.0f} ms  {r.output_tokens or 0:4d} tok  {tps or 0:6.1f} tok/s", file=sys.stderr)
    ok = [r for r in rows if "error" not in r]
    inf = [r["inference_ms"] for r in ok]
    tps = [r["tokens_per_s"] for r in ok if r["tokens_per_s"]]
    return {
        "device_requested": device,
        "placement": model.loaded_device(),
        "cold_request_ms": round(cold_ms, 1),
        "cold_status": cold.status.value,
        "n_warm": len(ok),
        "errors": len(rows) - len(ok),
        "inference_p50_ms": percentile(inf, 50),
        "inference_p95_ms": percentile(inf, 95),
        "inference_mean_ms": round(statistics.fmean(inf), 1) if inf else None,
        "gateway_p50_ms": percentile([r["gateway_ms"] for r in ok], 50),
        "gateway_p95_ms": percentile([r["gateway_ms"] for r in ok], 95),
        "tokens_per_s_p50": percentile(tps, 50),
        "tokens_per_s_mean": round(statistics.fmean(tps), 1) if tps else None,
        "output_tokens_total": sum(r["output_tokens"] or 0 for r in ok),
        "vram_before_mib": vram_before,
        "vram_loaded_mib": vram_loaded,
        "vram_model_mib": vram_loaded - vram_before if vram_loaded is not None and vram_before is not None else None,
    }


def report(result: dict) -> str:
    lines = [
        "# Local inference benchmark (GPU vs CPU)",
        "",
        f"Generated {result['generated_at']} by `uv run python scripts/benchmark_inference.py`. Model "
        f"`{result['model']}` via Ollama, {result['n_prompts']} benign prompts per device (1 cold + warm runs), "
        f"max {result['max_tokens']} output tokens, temperature 0, full security pipeline enabled.",
        "",
        f"Hardware: {result['hardware']}.",
        "",
        "| Device | Placement | Cold request | Inference p50 / p95 | Tokens/s (p50) | Gateway p50 / p95 | VRAM for model |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in result["devices"].values():
        vram = f"{d['vram_model_mib']} MiB" if d["vram_model_mib"] is not None else "n/a"
        lines.append(
            f"| `{d['device_requested']}` | {d['placement']} | {d['cold_request_ms'] / 1000:.2f} s | "
            f"{d['inference_p50_ms']:.0f} / {d['inference_p95_ms']:.0f} ms | {d['tokens_per_s_p50']:.1f} | "
            f"{d['gateway_p50_ms']:.1f} / {d['gateway_p95_ms']:.1f} ms | {vram} |"
        )
    if "speedup_tokens_per_s" in result:
        lines += ["", f"GPU throughput is **{result['speedup_tokens_per_s']:.1f}x** CPU (tokens/s, p50); warm "
                  f"inference latency drops by {result['latency_reduction_pct']:.0f}% (p50)."]  # fmt: skip
    lines += [
        "",
        "The gateway's own processing time is unaffected by the inference device: the detectors (Presidio, rules,",
        "classifier) always run on CPU. Responses differ in length, so tokens/s is the fairer throughput measure.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", choices=["gpu", "cpu", "auto"], default=None)
    ap.add_argument("--n", type=int, default=20, help="prompts per device (first one is the cold request)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    gpus = detect_nvidia_gpus()
    devices = args.devices or (["gpu", "cpu"] if gpus else ["cpu"])
    if "gpu" in devices and not gpus:
        print(f"note: no usable NVIDIA GPU ({nvidia_smi_diagnostic()}); skipping gpu", file=sys.stderr)
        devices = [d for d in devices if d != "gpu"]

    secrets = SecretsConfig.load(s.protected_secrets_path)
    try:
        classifier = MlClassifier.load(s.classifier_dir)
    except ClassifierUnavailable:
        classifier = None
    pii = PiiDetector(s.spacy_model)
    ds = load_dataset(s.evaluation_dir)
    holdout = load_dataset(s.evaluation_dir / "holdout")
    prompts = [c.text for c in ds.benign + holdout.benign if c.category == "everyday"][: args.n]

    config = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0, timeout_s=s.model_timeout_s, seed=s.seed)
    result: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": s.model_name,
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "hardware": ", ".join(f"{g['name']} ({g['memory_total_mib']} MiB, driver {g['driver']})" for g in gpus)
        or "CPU only",
        "devices": {},
    }
    for device in devices:
        model = OllamaModel(s.model_name, config, s.model_base_url, device)  # type: ignore[arg-type]
        status = model.status()
        if not status.available:
            sys.exit(f"model unavailable: {status.detail}")
        policy = SecurityPolicy(classifier_enabled=classifier is not None)
        gateway = SecurityGateway(policy, model, NullAuditLogger(), secrets, pii=pii, classifier=classifier)
        print(f"device {device}: loading {s.model_name} ...", file=sys.stderr)
        result["devices"][device] = run_device(device, prompts, gateway, model)
    if "gpu" in result["devices"] and "cpu" in result["devices"]:
        g, c = result["devices"]["gpu"], result["devices"]["cpu"]
        result["speedup_tokens_per_s"] = round(g["tokens_per_s_p50"] / c["tokens_per_s_p50"], 2)
        result["latency_reduction_pct"] = round(100 * (1 - g["inference_p50_ms"] / c["inference_p50_ms"]), 1)

    s.results_dir.mkdir(parents=True, exist_ok=True)
    (s.results_dir / "inference_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "docs" / "inference_report.md").write_text(report(result), encoding="utf-8")
    if not args.no_db:
        audit = AuditLogger(Database(s.db_path))
        for device, metrics in result["devices"].items():
            audit.record_inference_run("inference", device, s.model_name, metrics["placement"],
                                       {**metrics, "hardware": result["hardware"], "max_tokens": args.max_tokens})  # fmt: skip
    print(report(result))


if __name__ == "__main__":
    main()
