"""Stacked fusion on out-of-source predictions. Uses the training pool only; benchmark data is not read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import FOLDS, OUT, ROOT, load_pool  # noqa: E402

df = load_pool()
comp = np.load(OUT / "components_pool.npz")
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
rows = []
for fi, fold in enumerate(FOLDS):
    d = np.load(OUT / f"val_dsmall_{fold}.npz")
    t = np.load(OUT / f"val_tfidf_{fold}.npz")
    assert (d["idx"] == t["idx"]).all()
    idx = d["idx"]
    rows.append(np.column_stack([idx, np.full(len(idx), fi), comp["rule"][idx], t["score"], d["score"], comp["protectai"][idx]]))
M = np.vstack(rows)
idx, fold_id = M[:, 0].astype(int), M[:, 1].astype(int)
raw = {"rule": M[:, 2], "tfidf": M[:, 3], "dsmall": M[:, 4], "protectai": M[:, 5]}
feat = {"rule": raw["rule"], "tfidf": logit(raw["tfidf"]), "dsmall": logit(raw["dsmall"]), "protectai": logit(raw["protectai"])}
y = df.label.to_numpy()[idx]
src = df.source.to_numpy()[idx]
hn = src == "hardneg"
w = np.where(hn, 3.0, 1.0)
BENIGN = ["inthewild", "wambosec", "dolly", "alespalla", "llmail"]


def macro_fpr(s, t):
    return float(np.mean([(s[(src == b) & (y == 0)] >= t).mean() for b in BENIGN]))


def pick(s, fpr, hn_fpr):
    """Lowest threshold meeting both limits: macro benign FPR across sources and hard-negative FPR."""
    for t in np.quantile(s, np.linspace(0, 1, 1501)):
        if macro_fpr(s, t) <= fpr and (s[hn] >= t).mean() <= hn_fpr:
            return float(t)
    return float(s.max())


def report(name, s):
    warn, block = pick(s, 0.05, 0.05), pick(s, 0.01, 0.01)
    atk = {a: float((s[(src == a) & (y == 1)] >= warn).mean()) for a in ["wambosec", "inthewild", "llmail", "yanismiraoui", "mosscap"]}
    r = {"recall_macro@warn": float(np.mean(list(atk.values()))), "recall_macro@block": float(np.mean([(s[(src == a) & (y == 1)] >= block).mean() for a in atk])), "macro_fpr@warn": macro_fpr(s, warn), "hardneg_fpr@warn": float((s[hn] >= warn).mean()), "warn": warn, "block": block, "by_source_recall@warn": atk}
    print(f"{name:26} recall={r['recall_macro@warn']:.3f} (block {r['recall_macro@block']:.3f}) fpr={r['macro_fpr@warn']:.3f} hardneg={r['hardneg_fpr@warn']:.3f}  {({k: round(v, 2) for k, v in atk.items()})}", flush=True)
    return r


def loso(names):
    X = np.column_stack([feat[n] for n in names])
    out = np.zeros(len(y))
    for f in range(len(FOLDS)):
        tr = fold_id != f
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr], sample_weight=w[tr])
        out[fold_id == f] = m.predict_proba(X[fold_id == f])[:, 1]
    return out, X


res = {}
print("Single components (out-of-source):")
for n in ["rule", "tfidf", "dsmall", "protectai"]:
    res[n] = report(n, raw[n])
print("\nStacked fusion (leave-fold-out meta-model):")
VARIANTS = {"own": ["rule", "tfidf", "dsmall"], "with_guard": ["rule", "tfidf", "dsmall", "protectai"]}
d = ROOT / "models/research"
for name, names in VARIANTS.items():
    s, X = loso(names)
    res[name] = report(name, s)
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y, sample_weight=w)
    res[name]["coef"] = dict(zip(names, map(float, m.coef_[0]), strict=True))
    (d / f"fusion_{name}.json").write_text(json.dumps({"features": names, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "warn": res[name]["warn"], "block": res[name]["block"]}, indent=1))
    print("   coef", {k: round(v, 2) for k, v in res[name]["coef"].items()}, "warn", round(res[name]["warn"], 4), "block", round(res[name]["block"], 4))
(OUT / "fusion_selection.json").write_text(json.dumps(res, indent=1))
