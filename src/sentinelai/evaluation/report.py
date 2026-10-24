"""Run the benchmark on every split, persist runs, and render the Markdown report."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinelai.audit import AuditLogger
from sentinelai.detectors.ml_classifier import MlClassifier
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.evaluation.dataset import load_dataset
from sentinelai.evaluation.runner import run_benchmark

SPLITS = {
    "dev": ("", "Development split: written first; rules were debugged against it, so results are optimistic."),
    "holdout": ("holdout", "Holdout split: written after the rules were frozen; never used for tuning."),
}


def run_suite(
    evaluation_dir: Path,
    secrets: SecretsConfig,
    pii: PiiDetector | None = None,
    classifier: MlClassifier | None = None,
    include_classifier: bool = True,
) -> dict[str, Any]:
    pii = pii or PiiDetector()
    suite: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "splits": {}}
    for name, (sub, description) in SPLITS.items():
        dataset = load_dataset(evaluation_dir / sub if sub else evaluation_dir, name=f"sentinelai-eval-{name}")
        result = run_benchmark(dataset, secrets, pii=pii, classifier=classifier, include_classifier=include_classifier)
        result["description"] = description
        suite["splits"][name] = result
    return suite


def compact(suite: dict[str, Any]) -> dict[str, Any]:
    """Drop per-case rows (kept in the JSON file) for API responses and the database."""
    out = copy.deepcopy(suite)
    for split in out["splits"].values():
        for cfg in split["configurations"].values():
            cfg.pop("rows", None)
            cfg.get("output_suite", {}).pop("rows", None)
            cfg.get("edge_cases", {}).pop("rows", None)
    return out


# Per-case fields kept in the database (enough for score-distribution charts; no prompt text).
CASE_FIELDS = ("id", "label", "category", "status", "risk_score", "flagged")


def record_runs(audit: AuditLogger, suite: dict[str, Any]) -> list[str]:
    ids = []
    small = compact(suite)
    for split_name, split in small["splits"].items():
        for label, cfg in split["configurations"].items():
            full_rows = suite["splits"][split_name]["configurations"][label].get("rows", [])
            cfg["cases"] = [{k: r[k] for k in CASE_FIELDS} for r in full_rows]
            ids.append(
                audit.record_eval_run(
                    label=f"{split_name}/{label}",
                    dataset_version=split["dataset_version"],
                    config={"policy": cfg.get("policy"), "classifier": (split.get("classifier") or {}).get("model_version")},
                    metrics={k: v for k, v in cfg.items() if k != "policy"},
                )
            )
    return ids


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def render_markdown(suite: dict[str, Any], extra_sections: list[str] | None = None) -> str:
    lines = [
        "# SentinelAI benchmark report",
        "",
        f"Generated {suite['generated_at']} by `uv run python scripts/evaluate_security.py`. "
        "All numbers below are measured by that script; nothing is hand-edited.",
        "",
        "The model is the deterministic `MockModel`, which simulates a *vulnerable* LLM (it obeys "
        "extraction requests and echoes its input), so every configuration sees identical model behaviour.",
        "",
        "Definitions: *flagged* = the attack signals alone lead to WARN, REVIEW or BLOCK; *recall* = "
        "flagged malicious / all malicious; *precision* = malicious among flagged; *FPR* = flagged benign / "
        "all benign; *secret leakage* = malicious prompts whose returned response contains a protected "
        "synthetic secret (verbatim or with separators changed); *PII to model* = PII prompts where a labelled "
        "synthetic PII value reached the model.",
        "",
    ]
    for split_name, split in suite["splits"].items():
        counts = split["counts"]
        lines += [
            f"## Split: {split_name}",
            "",
            f"{split['description']} Dataset `{split['dataset_version']}`: {counts['malicious']} malicious, "
            f"{counts['benign']} benign prompts, {counts['output_cases']} output cases, {counts['edge_cases']} edge cases.",
            "",
            "| Configuration | Attack recall | Block recall | Precision | Benign FPR | F1 | Secret leakage | PII to model | Benign answered | Gateway p50 / p95 ms |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label, cfg in split["configurations"].items():
            a, b, lat = cfg["attack_detection"], cfg["attack_blocking"], cfg["latency_ms"]
            leak = cfg["secret_leakage"]
            pii = cfg["pii_exposure"]
            lines.append(
                f"| `{label}` | {_pct(a['recall'])} | {_pct(b['recall'])} | {_pct(a['precision'])} | "
                f"{_pct(a['false_positive_rate'])} | {a['f1'] if a['f1'] is not None else 'n/a'} | "
                f"{leak['leaked']}/{leak['n']} | {pii['reached_model']}/{pii['n']} | {_pct(cfg['benign_answer_rate'])} | "
                f"{_ms(lat['gateway_p50'])} / {_ms(lat['gateway_p95'])} |"
            )
        lines += ["", "Confusion matrices (flagged):", ""]
        for label, cfg in split["configurations"].items():
            cm = cfg["attack_detection"]["confusion_matrix"]
            lines.append(f"- `{label}`: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
        lines += [
            "",
            "Per attack category (flagged / n):",
            "",
            "| Category | " + " | ".join(f"`{k}`" for k in split["configurations"]) + " |",
            "|---|" + "---|" * len(split["configurations"]),
        ]
        cats = next(iter(split["configurations"].values()))["per_attack_category"].keys()
        for cat in cats:
            cells = [
                f"{c['per_attack_category'][cat]['flagged']}/{c['per_attack_category'][cat]['n']}"
                for c in split["configurations"].values()
            ]
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
        lines.append("")
        for label, cfg in split["configurations"].items():
            if label == "baseline":
                continue
            lines.append(
                f"Misses for `{label}`: false negatives {cfg['false_negatives'] or 'none'}; "
                f"false positives {cfg['false_positives'] or 'none'}; benign prompts not answered "
                f"{cfg['benign_not_answered'] or 'none'}; PII reaching the model {cfg['pii_exposure']['reached_model_ids'] or 'none'}."
            )
            lines.append("")
        first = next(iter(split["configurations"].values()))
        if first["output_suite"]["n"]:
            lines += [
                "Output suite (benign prompt, scripted model response, end to end):",
                "",
                "| Configuration | Expected action matched | Secret/credential cases leaked | Any sensitive value shown |",
                "|---|---|---|---|",
            ]
            for label, cfg in split["configurations"].items():
                o = cfg["output_suite"]
                lines.append(
                    f"| `{label}` | {_pct(o['expected_action_accuracy'])} | {o['secret_leaks']}/{o['secret_cases']} | "
                    f"{o['sensitive_leaks_all']}/{o['n']} |"
                )
            for label, cfg in split["configurations"].items():
                if label != "baseline" and cfg["output_suite"]["mismatches"]:
                    lines.append(f"\nOutput mismatches for `{label}`: {cfg['output_suite']['mismatches']}")
            lines.append("")
        if first["edge_cases"]["n"]:
            lines.append(
                "Edge cases passed: "
                + ", ".join(
                    f"`{label}` {cfg['edge_cases']['passed']}/{cfg['edge_cases']['n']}"
                    for label, cfg in split["configurations"].items()
                )
            )
            lines.append("")
        lines.append(
            "Operational check, blocked requests that still reached the model: "
            + ", ".join(
                f"`{label}` {cfg['operational']['blocked_but_model_invoked']}"
                for label, cfg in split["configurations"].items()
            )
        )
        lines.append("")
    for section in extra_sections or []:
        lines += [section, ""]
    return "\n".join(lines).rstrip() + "\n"
