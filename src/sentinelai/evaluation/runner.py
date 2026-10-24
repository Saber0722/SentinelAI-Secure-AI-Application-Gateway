"""Reproducible security benchmark: the same cases with controls disabled vs enabled.

The deterministic ``MockModel`` (a simulated *vulnerable* model) is used so runs are repeatable.
Leak checks are implemented independently of the detectors (plain substring tests) so the
detectors are not graded by themselves.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from sentinelai.audit import NullAuditLogger, percentile
from sentinelai.config import SecurityPolicy
from sentinelai.detectors.ml_classifier import MlClassifier
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.evaluation.dataset import EvalDataset, PromptCase
from sentinelai.evaluation.metrics import Confusion, ratio
from sentinelai.gateway.decisions import ATTACK_CATEGORIES, Action, Category
from sentinelai.gateway.pipeline import InputValidationError, SecurityGateway, Status
from sentinelai.gateway.policy_engine import decide_input
from sentinelai.models.mock_model import MockModel, ScriptedModel

_ATTACK_OR_ML = ATTACK_CATEGORIES | {Category.ML_SUSPICIOUS}


def _compact(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", text.casefold())


def leaks(text: str, values: list[str]) -> bool:
    """True if any value appears verbatim or with separators/case changed."""
    compact = _compact(text)
    return any(v in text or (len(_compact(v)) >= 6 and _compact(v) in compact) for v in values)


@dataclass(frozen=True, slots=True)
class Configuration:
    label: str
    policy: SecurityPolicy
    description: str


def configurations(classifier: MlClassifier | None, include_classifier: bool = True) -> list[Configuration]:
    configs = [
        Configuration("baseline", SecurityPolicy.disabled(), "All gateway controls disabled (unprotected)"),
        Configuration("protected", SecurityPolicy(), "Rules + secrets + PII + output scanning"),
    ]
    if classifier is not None and include_classifier:
        configs.append(
            Configuration("protected_ml", SecurityPolicy(classifier_enabled=True), "Protected configuration + ML classifier")
        )
    return configs


def _attack_flag(gateway: SecurityGateway, result_findings: list[Any], policy: SecurityPolicy) -> Action:
    """The action attributable to attack signals alone (PII/secret handling excluded)."""
    attack = [f for f in result_findings if f.category in _ATTACK_OR_ML]
    ml_block = ml_warn = None
    if policy.classifier_enabled and gateway.classifier is not None:
        ml_block = policy.classifier_block_threshold or gateway.classifier.block_threshold
        ml_warn = policy.classifier_warn_threshold or gateway.classifier.warn_threshold
    return decide_input(attack, policy, ml_block, ml_warn).action


def _run_prompts(
    gateway: SecurityGateway, model: MockModel, cases: list[PromptCase], policy: SecurityPolicy, protected_values: list[str]
) -> dict[str, Any]:
    rows = []
    for case in cases:
        calls_before = model.call_count
        r = gateway.chat(case.text, policy=policy)
        attack_action = _attack_flag(gateway, r.input_decision.findings, policy)
        model_saw = model.received[-1] if model.call_count > calls_before else None
        rows.append(
            {
                "id": case.id,
                "label": case.label,
                "category": case.category,
                "status": r.status.value,
                "input_action": r.input_decision.action.value,
                "attack_action": attack_action.value,
                "flagged": attack_action is not Action.ALLOW,
                "blocked": r.status in (Status.BLOCKED, Status.REVIEW),
                "model_invoked": model_saw is not None,
                "inference_attempted": r.inference_attempted,
                "secret_leaked": leaks(r.response, protected_values),
                "pii_reached_model": bool(case.sensitive_values)
                and model_saw is not None
                and leaks(model_saw, list(case.sensitive_values)),
                "pii_reached_user": bool(case.sensitive_values) and leaks(r.response, list(case.sensitive_values)),
                "redacted_input": r.input_decision.action is Action.REDACT_AND_ALLOW,
                "useful_response": r.status in (Status.ALLOWED, Status.ALLOWED_WITH_WARNING, Status.REDACTED),
                "gateway_ms": round(r.gateway_ms, 3),
                "inference_ms": round(r.inference_ms, 3) if r.inference_ms is not None else None,
                "risk_score": r.input_decision.risk_score,
                "categories": r.input_decision.categories,
            }
        )
    return {"rows": rows}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = Confusion.from_pairs([(r["label"] == "malicious", r["flagged"]) for r in rows])
    blocked = Confusion.from_pairs([(r["label"] == "malicious", r["blocked"]) for r in rows])
    malicious = [r for r in rows if r["label"] == "malicious"]
    benign = [r for r in rows if r["label"] == "benign"]
    pii_cases = [r for r in benign if r["category"] == "pii"]
    per_category: dict[str, dict[str, Any]] = {}
    for r in malicious:
        c = per_category.setdefault(r["category"], {"n": 0, "flagged": 0, "blocked": 0})
        c["n"] += 1
        c["flagged"] += r["flagged"]
        c["blocked"] += r["blocked"]
    for c in per_category.values():
        c["recall"] = ratio(c["flagged"], c["n"])
    benign_by_cat: dict[str, dict[str, Any]] = {}
    for r in benign:
        c = benign_by_cat.setdefault(r["category"], {"n": 0, "flagged": 0})
        c["n"] += 1
        c["flagged"] += r["flagged"]
    return {
        "attack_detection": flagged.summary(),
        "attack_blocking": blocked.summary(),
        "per_attack_category": per_category,
        "benign_by_category": benign_by_cat,
        "false_negatives": [r["id"] for r in malicious if not r["flagged"]],
        "false_positives": [r["id"] for r in benign if r["flagged"]],
        "benign_not_answered": [r["id"] for r in benign if not r["useful_response"]],
        "secret_leakage": {
            "leaked": sum(r["secret_leaked"] for r in malicious),
            "n": len(malicious),
            "rate": ratio(sum(r["secret_leaked"] for r in malicious), len(malicious)),
            "leaked_ids": [r["id"] for r in malicious if r["secret_leaked"]],
        },
        "pii_exposure": {
            "n": len(pii_cases),
            "reached_model": sum(r["pii_reached_model"] for r in pii_cases),
            "reached_model_rate": ratio(sum(r["pii_reached_model"] for r in pii_cases), len(pii_cases)),
            "reached_user": sum(r["pii_reached_user"] for r in pii_cases),
            "reached_model_ids": [r["id"] for r in pii_cases if r["pii_reached_model"]],
        },
        "benign_answer_rate": ratio(sum(r["useful_response"] for r in benign), len(benign)),
        "spurious_redaction": {
            "n": len(non_pii := [r for r in benign if r["category"] != "pii"]),
            "redacted": sum(r["redacted_input"] for r in non_pii),
            "rate": ratio(sum(r["redacted_input"] for r in non_pii), len(non_pii)),
            "ids": [r["id"] for r in non_pii if r["redacted_input"]],
        },
        "operational": {
            "blocked_but_model_invoked": sum(1 for r in rows if r["blocked"] and r["model_invoked"]),
        },
        "latency_ms": {
            "gateway_p50": percentile([r["gateway_ms"] for r in rows], 50),
            "gateway_p95": percentile([r["gateway_ms"] for r in rows], 95),
            "inference_p50": percentile([r["inference_ms"] for r in rows if r["inference_ms"] is not None], 50),
            "inference_p95": percentile([r["inference_ms"] for r in rows if r["inference_ms"] is not None], 95),
            "n": len(rows),
        },
    }


def _run_outputs(
    gateway: SecurityGateway, model: ScriptedModel, dataset: EvalDataset, policy: SecurityPolicy, protected_values: list[str]
) -> dict[str, Any]:
    """End-to-end: benign prompt in, scripted (possibly leaky) model response out."""
    rows = []
    for case in dataset.outputs:
        model.response = case.model_response
        r = gateway.chat(case.prompt, policy=policy)
        action = r.output_decision.action.value if r.output_decision else f"input:{r.input_decision.action.value}"
        sensitive = list(case.sensitive_values)
        rows.append(
            {
                "id": case.id,
                "category": case.category,
                "expected": case.expected,
                "action": action,
                "status": r.status.value,
                "correct": action == case.expected,
                "protected_secret_leaked": leaks(r.response, protected_values),
                "sensitive_leaked": bool(sensitive) and leaks(r.response, sensitive),
            }
        )
    secret_cases = [r for r in rows if r["category"] in {"protected_secret", "credential", "system_prompt"}]
    secret_leaks = sum(r["protected_secret_leaked"] or r["sensitive_leaked"] for r in secret_cases)
    return {
        "n": len(rows),
        "expected_action_accuracy": ratio(sum(r["correct"] for r in rows), len(rows)),
        "secret_cases": len(secret_cases),
        "secret_leaks": secret_leaks,
        "secret_leakage_rate": ratio(secret_leaks, len(secret_cases)),
        "sensitive_leaks_all": sum(r["sensitive_leaked"] or r["protected_secret_leaked"] for r in rows),
        "mismatches": [{"id": r["id"], "expected": r["expected"], "got": r["action"]} for r in rows if not r["correct"]],
        "rows": rows,
    }


def _run_edges(gateway: SecurityGateway, dataset: EvalDataset, policy: SecurityPolicy) -> dict[str, Any]:
    rows = []
    for case in dataset.edges:
        try:
            r = gateway.chat(case.payload, policy=policy)
            outcome = "blocked" if r.status in (Status.BLOCKED, Status.REVIEW) else "handled"
            detail = r.status.value
        except InputValidationError as exc:
            outcome, detail = "reject", str(exc)
        except Exception as exc:
            outcome, detail = "crash", type(exc).__name__
        rows.append(
            {
                "id": case.id,
                "kind": case.kind,
                "expect": case.expect,
                "outcome": outcome,
                "detail": detail,
                "pass": outcome == case.expect,
            }
        )
    return {"n": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}


def run_benchmark(
    dataset: EvalDataset,
    secrets: SecretsConfig,
    pii: PiiDetector | None = None,
    classifier: MlClassifier | None = None,
    include_classifier: bool = True,
) -> dict[str, Any]:
    pii = pii or PiiDetector()
    protected_values = [v for _, v in secrets.all_values]
    results: dict[str, Any] = {"dataset_version": dataset.version, "configurations": {}}
    started = time.time()
    for config in configurations(classifier, include_classifier):
        model = MockModel()
        gateway = SecurityGateway(config.policy, model, NullAuditLogger(), secrets, pii=pii, classifier=classifier)
        prompt_run = _run_prompts(gateway, model, dataset.prompts, config.policy, protected_values)
        summary = _summarize(prompt_run["rows"])
        scripted = ScriptedModel()
        out_gateway = SecurityGateway(config.policy, scripted, NullAuditLogger(), secrets, pii=pii, classifier=classifier)
        summary["output_suite"] = _run_outputs(out_gateway, scripted, dataset, config.policy, protected_values)
        summary["edge_cases"] = _run_edges(gateway, dataset, config.policy)
        summary["description"] = config.description
        summary["policy"] = config.policy.model_dump(mode="json")
        summary["rows"] = prompt_run["rows"]
        results["configurations"][config.label] = summary
    results["counts"] = {
        "benign": len(dataset.benign),
        "malicious": len(dataset.malicious),
        "output_cases": len(dataset.outputs),
        "edge_cases": len(dataset.edges),
    }
    results["classifier"] = classifier.metadata if classifier else None
    results["duration_s"] = round(time.time() - started, 2)
    return results
