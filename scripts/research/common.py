"""Shared helpers for the detector research. The six benchmark datasets are only ever opened by name here
to remove near-duplicates from the training pool, and by final_eval.py for the one-time test."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data" / "external" / "pool"
POOL_FILE = POOL_DIR / "pool.parquet"
HARDNEG_FILE = POOL_DIR / "hardneg.parquet"
OUT = ROOT / "results" / "research"
TOK = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> frozenset[str]:
    return frozenset(TOK.findall(text.lower()))


def load_pool(with_hardneg: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(POOL_FILE)
    if with_hardneg and HARDNEG_FILE.exists():
        df = pd.concat([df, pd.read_parquet(HARDNEG_FILE)], ignore_index=True)
    short = POOL_DIR / "short.parquet"
    if with_hardneg and short.exists():  # appended last so earlier fold indices stay valid
        df = pd.concat([df, pd.read_parquet(short)], ignore_index=True)
    return df.reset_index(drop=True)


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


# Leave-source-out folds: each fold's validation set is whole public sources the model does not train on.
FOLDS = {
    "A_wambosec": ["wambosec"],
    "B_inthewild": ["inthewild"],
    "C_llmail_yanis_dolly": ["llmail", "yanismiraoui", "dolly"],
    "D_mosscap_alespalla": ["mosscap", "alespalla"],
}


def split(df: pd.DataFrame, fold: str):
    """Return (train_idx, val_idx). 20% of hard negatives always sit in validation; own templates stay in train."""
    import numpy as np

    if fold == "calib":  # final model: random 15% of every source held out for blending and thresholds
        mask = np.zeros(len(df), dtype=bool)
        rng = np.random.default_rng(2026)
        for _, g in df.groupby(["source", "label"]):
            idx = g.index.to_numpy()
            mask[rng.choice(idx, max(1, int(0.15 * len(idx))), replace=False)] = True
        return np.where(~mask)[0], np.where(mask)[0]
    held = FOLDS[fold]
    hn = (df.source == "hardneg").to_numpy()
    rng = np.random.default_rng(99)
    hn_val = hn & (rng.random(len(df)) < 0.2)
    val = df.source.isin(held).to_numpy() | hn_val
    train = ~val
    return np.where(train)[0], np.where(val)[0]


def val_metrics(df_val: pd.DataFrame, scores) -> dict:
    """Recall at fixed benign false-positive rates, and over-defence on held-out hard negatives."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    s = np.asarray(scores)
    y = df_val.label.to_numpy()
    hn = (df_val.source == "hardneg").to_numpy()
    normal = ~hn
    neg = s[normal & (y == 0)]
    out: dict = {"auc": float(roc_auc_score(y[normal], s[normal])) if len(set(y[normal])) == 2 else None, "n_val": int(normal.sum())}
    for fpr in (0.01, 0.05):
        t = float(np.quantile(neg, 1 - fpr)) if len(neg) else 0.5
        out[f"recall@fpr{int(fpr * 100)}"] = float((s[normal & (y == 1)] >= t).mean())
        out[f"hardneg_fpr@fpr{int(fpr * 100)}"] = float((s[hn] >= t).mean()) if hn.any() else None
    return out
