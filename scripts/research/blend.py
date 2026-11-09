"""Holdout blending on the 15% calibration set. Fixes fusion weights and thresholds before the benchmark is opened."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_pool  # noqa: E402

df = load_pool()
comp = np.load(OUT / "components_pool.npz")
d, t = np.load(OUT / "val_dsmall_final_calib.npz"), np.load(OUT / "val_tfidf_calib.npz")
assert (d["idx"] == t["idx"]).all()
idx = d["idx"]
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
raw = {"rule": comp["rule"][idx], "tfidf": t["score"], "dsmall": d["score"], "protectai": comp["protectai"][idx]}
feat = {"rule": raw["rule"], "tfidf": logit(raw["tfidf"]), "dsmall": logit(raw["dsmall"]), "protectai": logit(raw["protectai"])}
y, src = df.label.to_numpy()[idx], df.source.to_numpy()[idx]
hn = src == "hardneg"
w = np.where(hn, 3.0, 1.0)
BENIGN = ["inthewild", "wambosec", "dolly", "alespalla", "llmail", "own_templates"]
ATTACK = ["wambosec", "inthewild", "llmail", "yanismiraoui", "mosscap", "own_templates"]
macro_fpr = lambda s, th: float(np.mean([(s[(src == b) & (y == 0)] >= th).mean() for b in BENIGN]))


def pick(s, fpr, hn_fpr):
    for th in np.quantile(s, np.linspace(0, 1, 2001)):
        if macro_fpr(s, th) <= fpr and (s[hn] >= th).mean() <= hn_fpr:
            return float(th)
    return float(s.max())


def report(name, s):
    warn, block = pick(s, 0.02, 0.05), pick(s, 0.005, 0.01)
    atk = {a: round(float((s[(src == a) & (y == 1)] >= warn).mean()), 3) for a in ATTACK}
    r = {"auc": float(roc_auc_score(y, s)), "recall_macro": float(np.mean(list(atk.values()))), "macro_fpr": macro_fpr(s, warn), "hardneg_fpr": float((s[hn] >= warn).mean()), "warn": warn, "block": block, "by_source": atk}
    print(f"{name:12} auc={r['auc']:.4f} recall={r['recall_macro']:.3f} fpr={r['macro_fpr']:.3f} hardneg={r['hardneg_fpr']:.3f} warn={warn:.3f} {atk}", flush=True)
    return r


res = {n: report(n, raw[n]) for n in raw}
for name, names in {"own": ["rule", "tfidf", "dsmall"], "with_guard": ["rule", "tfidf", "dsmall", "protectai"]}.items():
    X = np.column_stack([feat[n] for n in names])
    cv = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=3).split(X, y):
        cv[b] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X[a], y[a], sample_weight=w[a]).predict_proba(X[b])[:, 1]
    res[name] = report(name, cv)
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y, sample_weight=w)
    res[name]["coef"] = dict(zip(names, map(float, m.coef_[0]), strict=True))
    print("   coef", {k: round(v, 2) for k, v in res[name]["coef"].items()})
    (ROOT / f"models/research/fusion_{name}.json").write_text(json.dumps({"features": names, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "warn": res[name]["warn"], "block": res[name]["block"]}, indent=1))
(OUT / "blend_selection.json").write_text(json.dumps(res, indent=1))
