"""Train the optional TF-IDF + logistic-regression prompt classifier (CPU; takes seconds).

Protocol
  1. Load data/training/prompts.jsonl (see generate_training_data.py).
  2. Drop training rows that duplicate or nearly duplicate any evaluation prompt (dev + holdout).
  3. Hold out whole template families as the test split (stratified by label).
  4. Validation = grouped 5-fold cross-validation over the remaining families: each family is
     predicted by a model that never saw it. C and the thresholds are chosen from these
     out-of-fold predictions; the final model is then fit on the training families only.
  5. Evaluate once on the held-out test families, and on the hand-written evaluation sets.
  6. Save the artifact, its SHA-256 and all versions/seeds to models/classifier/.

    uv run python scripts/train_classifier.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from sentinelai.detectors.ml_classifier import METADATA_FILE, MODEL_FILE, featurize, sha256_file
from sentinelai.evaluation.dataset import read_jsonl

ROOT = Path(__file__).resolve().parents[1]

C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]
NEAR_DUP_JACCARD = 0.8


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", featurize(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def remove_eval_overlap(rows: list[dict], eval_texts: list[str]) -> tuple[list[dict], int]:
    eval_tokens = [tokens(t) for t in eval_texts]
    kept, dropped = [], 0
    for r in rows:
        t = tokens(r["text"])
        if any(jaccard(t, e) >= NEAR_DUP_JACCARD for e in eval_tokens):
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def split_families(rows: list[dict], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    splits: dict[str, list[str]] = {"train": [], "test": []}
    for label in ("malicious", "benign"):
        fams = sorted({r["group"] for r in rows if r["label"] == label})
        rng.shuffle(fams)
        n_test = max(2, round(len(fams) * 0.2))
        splits["test"] += fams[:n_test]
        splits["train"] += fams[n_test:]
    return splits


def out_of_fold(X: list[str], y: np.ndarray, groups: list[str], c: float, seed: int) -> np.ndarray:
    """Validation predictions: every family is scored by a model trained without it."""
    oof = np.zeros(len(y))
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(X, y, groups):
        pipe = build_pipeline(c, seed).fit([X[i] for i in fit_idx], y[fit_idx])
        oof[val_idx] = pipe.predict_proba([X[i] for i in val_idx])[:, 1]
    return oof


def build_pipeline(c: float, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def scores_at(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, object]:
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": t,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "training" / "prompts.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "classifier")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    t_start = time.time()

    rows = read_jsonl(args.data)
    eval_dirs = [ROOT / "data" / "evaluation", ROOT / "data" / "evaluation" / "holdout"]
    eval_rows = [r for d in eval_dirs for f in ("benign.jsonl", "malicious.jsonl") for r in read_jsonl(d / f)]
    rows, dropped = remove_eval_overlap(rows, [r["text"] for r in eval_rows])
    print(f"training corpus: {len(rows)} rows ({dropped} dropped as near-duplicates of evaluation prompts)")

    families = split_families(rows, args.seed)
    part = {name: [r for r in rows if r["group"] in set(fams)] for name, fams in families.items()}
    X = {k: [featurize(r["text"]) for r in v] for k, v in part.items()}
    y = {k: np.array([1 if r["label"] == "malicious" else 0 for r in v]) for k, v in part.items()}
    for k in part:
        print(f"  {k:<10} {len(part[k]):4d} rows, {int(y[k].sum()):4d} malicious, families: {sorted(families[k])}")

    # Model selection on out-of-fold (validation) predictions only.
    groups = [r["group"] for r in part["train"]]
    best = None
    for c in C_GRID:
        p_oof = out_of_fold(X["train"], y["train"], groups, c, args.seed)
        auc = roc_auc_score(y["train"], p_oof)
        print(f"  C={c:<5} validation (grouped CV) ROC-AUC={auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, c, p_oof)
    _, best_c, p_val = best
    y_val = y["train"]

    # Block threshold: lowest threshold with validation FPR <= 2% (precision first, since a block
    # stops a user); warn threshold: lowest threshold with validation FPR <= 5%.
    def lowest_with_fpr(limit: float) -> float:
        return next(t for t in THRESHOLDS if (scores_at(y_val, p_val, t)["false_positive_rate"] or 0) <= limit)

    block = (
        lowest_with_fpr(0.02)
        if any((scores_at(y_val, p_val, t)["false_positive_rate"] or 0) <= 0.02 for t in THRESHOLDS)
        else THRESHOLDS[-1]
    )
    warn = min(lowest_with_fpr(0.05), block)
    pipe = build_pipeline(best_c, args.seed).fit(X["train"], y["train"])
    print(f"  selected C={best_c}, block threshold={block}, warn threshold={warn}")

    # Single evaluation on held-out test families.
    p_test = pipe.predict_proba(X["test"])[:, 1]
    test = {
        "at_block": scores_at(y["test"], p_test, block),
        "at_warn": scores_at(y["test"], p_test, warn),
        "roc_auc": round(float(roc_auc_score(y["test"], p_test)), 4),
        "n": len(y["test"]),
    }
    # External check on the hand-written evaluation prompts (never used for any choice above).
    external = {}
    for d in eval_dirs:
        ev = [r for f in ("benign.jsonl", "malicious.jsonl") for r in read_jsonl(d / f)]
        ye = np.array([1 if r["label"] == "malicious" else 0 for r in ev])
        pe = pipe.predict_proba([featurize(r["text"]) for r in ev])[:, 1]
        external[d.name] = {
            "at_block": scores_at(ye, pe, block),
            "at_warn": scores_at(ye, pe, warn),
            "roc_auc": round(float(roc_auc_score(ye, pe)), 4),
            "n": len(ye),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / MODEL_FILE
    joblib.dump(pipe, model_path)
    digest = sha256_file(model_path)
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    metadata = {
        "model_version": f"tfidf-lr-{data_sha[:8]}-s{args.seed}",
        "algorithm": "TF-IDF (word 1-2 + char_wb 2-5 n-grams) + LogisticRegression(class_weight=balanced)",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifact_sha256": digest,
        "dataset": {
            "path": str(args.data.relative_to(ROOT)),
            "sha256": data_sha,
            "version": rows[0].get("version"),
            "generator_seed": rows[0].get("seed"),
            "rows_after_dedup": len(rows),
            "dropped_near_duplicates": dropped,
        },
        "split_seed": args.seed,
        "splits": {
            k: {"families": sorted(v), "rows": len(part[k]), "malicious": int(y[k].sum())} for k, v in families.items()
        },
        "hyperparameters": {"C": best_c, "C_grid": C_GRID},
        "thresholds": {
            "block": block,
            "warn": warn,
            "selection": "grouped 5-fold CV over training families; block = lowest t with validation FPR <= 2%, warn = lowest t with FPR <= 5%",
        },
        "validation": {
            "method": "GroupKFold(5) out-of-fold",
            "at_block": scores_at(y_val, p_val, block),
            "at_warn": scores_at(y_val, p_val, warn),
            "roc_auc": round(float(roc_auc_score(y_val, p_val)), 4),
        },
        "test": test,
        "external_evaluation": external,
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": np.__version__},
        "device": "cpu (GPU gives no benefit for a linear model on ~1k examples)",
        "train_seconds": round(time.time() - t_start, 2),
    }
    (args.out / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "classifier_metrics.json").write_text(
        json.dumps({k: v for k, v in metadata.items()}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test": test, "external": external}, indent=2))
    print(f"saved {model_path} (sha256 {digest[:12]}...) in {metadata['train_seconds']}s")


if __name__ == "__main__":
    main()
