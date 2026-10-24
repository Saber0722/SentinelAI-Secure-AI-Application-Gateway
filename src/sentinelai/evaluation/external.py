"""External (real-world, third-party) benchmark: public datasets and open-source guard models.

None of these datasets was used to build SentinelAI's rules or to train its classifier. Some guard
models *were* trained on some of them; that is recorded in ``GuardModel.trained_on`` so results can be
marked as in-distribution rather than silently compared.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinelai.audit import percentile
from sentinelai.config import SecurityPolicy
from sentinelai.gateway.decisions import ATTACK_CATEGORIES, Action, Category, combine_scores
from sentinelai.gateway.pipeline import InputValidationError, SecurityGateway
from sentinelai.gateway.policy_engine import decide_input

SAMPLE_SEED = 20261023


@dataclass(frozen=True, slots=True)
class ExternalCase:
    id: str
    text: str
    label: int  # 1 = attack, 0 = benign


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    repo: str
    files: tuple[str, ...]
    license: str
    description: str
    note: str
    loader: Callable[[Path], list[ExternalCase]]
    sample: int | None = None  # stratified random sample size (None = all rows)


@dataclass(frozen=True, slots=True)
class GuardModel:
    key: str
    repo: str
    license: str
    params: str
    trained_on: tuple[str, ...] = field(default=())  # dataset keys listed in the model card


def _read(path: Path):
    import pandas as pd

    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _cases(prefix: str, texts: list[str], labels: list[int]) -> list[ExternalCase]:
    return [ExternalCase(f"{prefix}-{i:05d}", str(t), int(y)) for i, (t, y) in enumerate(zip(texts, labels, strict=True))]


def _deepset(d: Path) -> list[ExternalCase]:
    df = _read(next(d.glob("data/test-*.parquet")))
    return _cases("deepset", df.text.tolist(), df.label.tolist())


def _jackhhao(d: Path) -> list[ExternalCase]:
    df = _read(d / "default" / "jailbreak_dataset_test.csv")
    return _cases("jackhhao", df.prompt.tolist(), (df.type == "jailbreak").astype(int).tolist())


def _gandalf(d: Path) -> list[ExternalCase]:
    df = _read(next(d.glob("data/test-*.parquet")))
    return _cases("gandalf", df.text.tolist(), [1] * len(df))


def _safeguard(d: Path) -> list[ExternalCase]:
    df = _read(next(d.glob("data/test-*.parquet")))
    return _cases("safeguard", df.text.tolist(), df.label.tolist())


def _spml(d: Path) -> list[ExternalCase]:
    df = _read(d / "spml_prompt_injection.csv").dropna(subset=["User Prompt"])
    return _cases("spml", df["User Prompt"].str.strip().tolist(), df["Prompt injection"].astype(int).tolist())


def _notinject(d: Path) -> list[ExternalCase]:
    import pandas as pd

    df = pd.concat([_read(p) for p in sorted(d.glob("data/NotInject_*.parquet"))], ignore_index=True)
    return _cases("notinject", df.prompt.tolist(), [0] * len(df))


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("deepset", "deepset/prompt-injections", ("data/test-00000-of-00001-701d16158af87368.parquet",),
                "Apache-2.0", "Prompt injections and benign prompts, English and German (test split)",
                "Labels are broad: some role-play requests ('act as an interviewer') count as injections.", _deepset),
    DatasetSpec("jackhhao", "jackhhao/jailbreak-classification", ("default/jailbreak_dataset_test.csv",),
                "Apache-2.0", "Real jailbreak prompts (DAN etc.) vs benign prompts (test split)",
                "Benign side contains many persona/role-play prompts: hard negatives.", _jackhhao),
    DatasetSpec("safeguard", "xTRam1/safe-guard-prompt-injection", ("data/test-00000-of-00001.parquet",),
                "not stated", "Synthetic + collected injection prompts vs instruction-tuning prompts (test split)",
                "Licence not stated on the card: used for evaluation only, not redistributed.", _safeguard),
    DatasetSpec("spml", "reshabhs/SPML_Chatbot_Prompt_Injection", ("spml_prompt_injection.csv",),
                "MIT", "User prompts that violate a chatbot's system prompt, graded by degree",
                "Labels depend on the system prompt, which detectors here do not see; sampled 2,000 rows.",
                _spml, sample=2000),
    DatasetSpec("gandalf", "Lakera/gandalf_ignore_instructions", ("data/test-00000-of-00001-bc92128b9288a6d1.parquet",),
                "MIT", "Real user attacks on Lakera's Gandalf game (test split)",
                "Attacks only: measures recall.", _gandalf),
    DatasetSpec("notinject", "leolee99/NotInject",
                ("data/NotInject_one-00000-of-00001.parquet", "data/NotInject_two-00000-of-00001.parquet",
                 "data/NotInject_three-00000-of-00001.parquet"),
                "MIT", "Benign prompts deliberately containing injection trigger words (InjecGuard paper)",
                "Benign only: measures over-defence (false-positive rate). Includes Chinese prompts.", _notinject),
)  # fmt: skip

GUARD_MODELS: tuple[GuardModel, ...] = (
    GuardModel("protectai_v2", "protectai/deberta-v3-base-prompt-injection-v2", "Apache-2.0", "184M", ("jackhhao",)),
    GuardModel("deepset", "deepset/deberta-v3-base-injection", "MIT", "184M", ("deepset",)),
    GuardModel("preamble", "PreambleAI/prompt-injection-defense", "Apache-2.0", "149M"),
)


def load_dataset(spec: DatasetSpec, root: Path) -> list[ExternalCase]:
    cases = spec.loader(root / spec.key)
    if spec.sample and len(cases) > spec.sample:
        rng = random.Random(SAMPLE_SEED)
        pos = [c for c in cases if c.label == 1]
        neg = [c for c in cases if c.label == 0]
        k_pos = round(spec.sample * len(pos) / len(cases))
        cases = rng.sample(pos, k_pos) + rng.sample(neg, spec.sample - k_pos)
    return cases


# ------------------------------------------------------------------------------------------ scoring
@dataclass
class Scores:
    """Per-case score in [0, 1] (higher = more malicious) and binary decision at the system's default."""

    scores: list[float]
    flagged: list[bool]
    latency_ms: list[float]
    device: str


def sentinel_scorer(gateway: SecurityGateway, use_rules: bool, use_ml: bool) -> Callable[[list[str]], Scores]:
    """SentinelAI attack detection only (PII/secret handling is not an attack verdict)."""
    policy = SecurityPolicy(
        input_rules_enabled=use_rules, classifier_enabled=use_ml, pii_input_enabled=False,
        input_secret_action="ALLOW",
    )  # fmt: skip
    clf = gateway.classifier
    ml_block = clf.block_threshold if (use_ml and clf) else None
    ml_warn = clf.warn_threshold if (use_ml and clf) else None

    def run(texts: list[str]) -> Scores:
        scores, flagged, lat = [], [], []
        for text in texts:
            t0 = time.perf_counter()
            try:
                analysis = gateway.analyze_input(text, policy)
            except InputValidationError:  # empty/invisible-only rows: rejected, i.e. not flagged as an attack
                scores.append(0.0)
                flagged.append(False)
                lat.append((time.perf_counter() - t0) * 1000)
                continue
            findings = analysis.decision.findings
            attack = [f for f in findings if f.category in ATTACK_CATEGORIES]
            rule_score = combine_scores([f.score for f in attack]) if attack else 0.0
            # The classifier only emits a finding above its warn threshold; score the raw probability for AUC.
            p = clf.probability(analysis.normalized) if (use_ml and clf) else 0.0
            relevant = attack + [f for f in findings if f.category is Category.ML_SUSPICIOUS]
            action = decide_input(relevant, policy, ml_block, ml_warn).action
            lat.append((time.perf_counter() - t0) * 1000)
            scores.append(max(rule_score if use_rules else 0.0, p))
            flagged.append(action is not Action.ALLOW)
        return Scores(scores, flagged, lat, "cpu")

    return run


def guard_scorer(model_dir: Path, device: str, batch_size: int = 32) -> tuple[Callable[[list[str]], Scores], dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    labels = {int(k): v for k, v in model.config.id2label.items()}
    malicious = [
        i
        for i, name in labels.items()
        if any(w in name.lower() for w in ("inject", "jailbreak", "malicious", "unsafe", "untrusted"))
    ]
    if len(malicious) != 1:
        raise ValueError(f"cannot identify the malicious label in {labels}")
    idx = malicious[0]
    info = {"labels": labels, "malicious_label": labels[idx], "max_length": min(512, tok.model_max_length)}

    @torch.inference_mode()
    def probs(batch: list[str]) -> list[float]:
        enc = tok(batch, truncation=True, max_length=info["max_length"], padding=True, return_tensors="pt").to(device)
        return torch.softmax(model(**enc).logits.float(), dim=-1)[:, idx].tolist()

    def run(texts: list[str]) -> Scores:
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            out += probs(texts[i : i + batch_size])
        # Per-request latency (batch of 1), measured on a fixed subset so it is comparable with SentinelAI.
        lat = []
        for text in texts[:200]:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs([text])
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        return Scores(out, [p >= 0.5 for p in out], lat, device)

    return run, info


def metrics(labels: list[int], s: Scores) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    tp = sum(1 for y, f in zip(labels, s.flagged, strict=True) if y and f)
    fp = sum(1 for y, f in zip(labels, s.flagged, strict=True) if not y and f)
    fn = sum(1 for y, f in zip(labels, s.flagged, strict=True) if y and not f)
    tn = sum(1 for y, f in zip(labels, s.flagged, strict=True) if not y and not f)
    pos, neg = tp + fn, fp + tn
    recall = tp / pos if pos else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if pos and (tp + fp) else None)
    auc = roc_auc_score(labels, s.scores) if pos and neg else None
    return {
        "n": len(labels), "positives": pos, "negatives": neg,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "precision": precision, "f1": f1,
        "fpr": fp / neg if neg else None, "accuracy": (tp + tn) / len(labels),
        "roc_auc": auc,
        "latency_p50_ms": percentile(s.latency_ms, 50), "latency_p95_ms": percentile(s.latency_ms, 95),
        "device": s.device,
    }  # fmt: skip
