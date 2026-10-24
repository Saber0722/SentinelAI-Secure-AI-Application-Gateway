"""Run the reproducible security benchmark and write the report.

    uv run python scripts/evaluate_security.py                 # mock model (deterministic)
    uv run python scripts/evaluate_security.py --live 20       # + real local LLM run (Ollama)

Outputs: results/benchmark_latest.json (all per-case rows), docs/benchmark_report.md, and one
eval_runs row per split/configuration in the SQLite database (shown on the dashboard).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentinelai.audit import AuditLogger, NullAuditLogger, percentile
from sentinelai.config import SecurityPolicy, get_settings
from sentinelai.detectors.ml_classifier import ClassifierUnavailable, MlClassifier
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.environment import detect_nvidia_gpus
from sentinelai.evaluation.dataset import load_dataset
from sentinelai.evaluation.report import compact, record_runs, render_markdown, run_suite
from sentinelai.evaluation.runner import leaks
from sentinelai.gateway.pipeline import SecurityGateway, Status
from sentinelai.models import GenerationConfig, ModelError, build_model
from sentinelai.storage.database import Database

ROOT = Path(__file__).resolve().parents[1]


def live_run(n: int, secrets: SecretsConfig, pii: PiiDetector, classifier: MlClassifier | None) -> dict:
    """Send the first n malicious + n benign holdout prompts to the real local model."""
    s = get_settings()
    config = GenerationConfig(max_tokens=160, temperature=0.0, timeout_s=s.model_timeout_s, seed=s.seed)
    # Pin the device explicitly: "auto" may reuse an instance Ollama already loaded on CPU.
    device = "gpu" if detect_nvidia_gpus() else "cpu"
    model = build_model("ollama", s.model_name, s.model_base_url, device, config)
    status = model.status()
    if not status.available:
        return {"skipped": status.detail}
    ds = load_dataset(ROOT / "data" / "evaluation" / "holdout")
    cases = ds.malicious[:n] + ds.benign[:n]
    protected = [v for _, v in secrets.all_values]
    out: dict = {"model": s.model_name, "n_malicious": len(ds.malicious[:n]), "n_benign": len(ds.benign[:n]), "configs": {}}
    policies = {
        "baseline": SecurityPolicy.disabled(),
        "protected": SecurityPolicy(classifier_enabled=classifier is not None),
    }
    for label, policy in policies.items():
        gw = SecurityGateway(policy, model, NullAuditLogger(), secrets, pii=pii, classifier=classifier)
        rows = []
        for case in cases:
            try:
                r = gw.chat(case.text)
            except ModelError as exc:  # pragma: no cover - live failure
                rows.append({"id": case.id, "error": str(exc)})
                continue
            rows.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "status": r.status.value,
                    "raw_output_leaked": r.raw_model_output is not None and leaks(r.raw_model_output, protected),
                    "returned_leaked": leaks(r.response, protected),
                    "gateway_ms": r.gateway_ms,
                    "inference_ms": r.inference_ms,
                }
            )
            print(f"  live {label:9} {case.id:7} {r.status.value}", file=sys.stderr)
        inf = [r["inference_ms"] for r in rows if r.get("inference_ms") is not None]
        out["configs"][label] = {
            "model_leaked_secret": sum(r.get("raw_output_leaked", False) for r in rows),
            "returned_secret": sum(r.get("returned_leaked", False) for r in rows),
            "blocked_before_model": sum(r.get("status") in (Status.BLOCKED, Status.REVIEW) for r in rows),
            "output_blocked": sum(r.get("status") == Status.OUTPUT_BLOCKED for r in rows),
            "errors": sum("error" in r for r in rows),
            "inference_p50_ms": percentile(inf, 50),
            "inference_p95_ms": percentile(inf, 95),
            "n_inference": len(inf),
            "gateway_p50_ms": percentile([r["gateway_ms"] for r in rows if "gateway_ms" in r], 50),
            "gateway_p95_ms": percentile([r["gateway_ms"] for r in rows if "gateway_ms" in r], 95),
            "device": model.loaded_device() if hasattr(model, "loaded_device") else None,
            "rows": rows,
        }
    return out


def live_section(live: dict) -> str:
    if "skipped" in live:
        return f"## Live local-LLM run\n\nSkipped: {live['skipped']}"
    lines = [
        "## Live local-LLM run",
        "",
        f"Model `{live['model']}` via Ollama (temperature 0, seed fixed, max 160 tokens), first {live['n_malicious']} "
        f"malicious + {live['n_benign']} benign holdout prompts. Real LLM output is not perfectly reproducible "
        "across hardware/driver versions, so these numbers are indicative; the mock-model tables above are the "
        "reproducible benchmark.",
        "",
        "| Configuration | Model output contained a secret | Secret returned to user | Blocked before model | Output blocked | Model p50 / p95 ms | Gateway p50 / p95 ms | Device |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, c in live["configs"].items():
        lines.append(
            f"| `{label}` | {c['model_leaked_secret']} | {c['returned_secret']} | {c['blocked_before_model']} | "
            f"{c['output_blocked']} | {c['inference_p50_ms']} / {c['inference_p95_ms']} | {c['gateway_p50_ms']:.2f} / "
            f"{c['gateway_p95_ms']:.2f} | {c['device']} |"
        )
    return "\n".join(lines)


def classifier_section(clf: MlClassifier | None) -> str:
    if clf is None:
        return "## ML classifier\n\nNo trained classifier found; `protected_ml` was not evaluated."
    m = clf.metadata
    t = m["test"]
    lines = [
        "## ML classifier (held-out test families)",
        "",
        f"`{m['model_version']}` — {m['algorithm']}. Thresholds chosen by {m['thresholds']['selection']}: "
        f"block {m['thresholds']['block']}, warn {m['thresholds']['warn']}. Test families "
        f"(never seen in training): {m['splits']['test']['families']}.",
        "",
        "| Data | Threshold | Precision | Recall | F1 | FPR | ROC-AUC | TP/FP/TN/FN |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [("test families", t)] + [(f"hand-written `{k}`", v) for k, v in m["external_evaluation"].items()]
    for name, d in rows:
        for which in ("at_block", "at_warn"):
            s = d[which]
            cm = s["confusion_matrix"]
            lines.append(
                f"| {name} | {which[3:]} {s['threshold']} | {s['precision']} | {s['recall']} | {s['f1']} | "
                f"{s['false_positive_rate']} | {d['roc_auc']} | {cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=int, default=0, metavar="N", help="also run N+N holdout prompts through Ollama")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--no-db", action="store_true", help="do not store the runs in the SQLite database")
    args = ap.parse_args()

    settings = get_settings()
    secrets = SecretsConfig.load(settings.protected_secrets_path)
    pii = PiiDetector(settings.spacy_model)
    classifier = None
    if not args.no_classifier:
        try:
            classifier = MlClassifier.load(settings.classifier_dir)
        except ClassifierUnavailable as exc:
            print(f"note: {exc}", file=sys.stderr)

    suite = run_suite(settings.evaluation_dir, secrets, pii=pii, classifier=classifier)
    sections = [classifier_section(classifier)]
    if args.live:
        suite["live"] = live_run(args.live, secrets, pii, classifier)
        sections.append(live_section(suite["live"]))

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "benchmark_latest.json").write_text(json.dumps(suite, indent=1, default=str), encoding="utf-8")
    (ROOT / "docs" / "benchmark_report.md").write_text(render_markdown(suite, sections), encoding="utf-8")
    if not args.no_db:
        audit = AuditLogger(Database(settings.db_path))
        record_runs(audit, suite)
        live = suite.get("live", {})
        for label, cfg in live.get("configs", {}).items():
            metrics = {k: v for k, v in cfg.items() if k != "rows"}
            metrics.update(n_malicious=live["n_malicious"], n_benign=live["n_benign"])
            audit.record_inference_run("live_security", label, live["model"], cfg.get("device"), metrics)

    for split, res in compact(suite)["splits"].items():
        for label, cfg in res["configurations"].items():
            a = cfg["attack_detection"]
            print(
                f"{split:8} {label:13} recall={a['recall']} precision={a['precision']} fpr={a['false_positive_rate']} "
                f"leaks={cfg['secret_leakage']['leaked']}/{cfg['secret_leakage']['n']} "
                f"pii_to_model={cfg['pii_exposure']['reached_model']}/{cfg['pii_exposure']['n']}"
            )
    print("wrote results/benchmark_latest.json and docs/benchmark_report.md")


if __name__ == "__main__":
    main()
