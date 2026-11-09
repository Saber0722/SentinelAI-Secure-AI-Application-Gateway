"""Leave-source-out baseline: TF-IDF + logistic regression retrained on the public pool."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

sys.path.insert(0, str(Path(__file__).parent))
from common import FOLDS, OUT, dump, load_pool, split, val_metrics  # noqa: E402

from sentinelai.detectors.ml_classifier import featurize  # noqa: E402

df = load_pool()
texts = np.array([featurize(t) for t in df.text], dtype=object)
y = df.label.to_numpy()
res = {}
for fold in FOLDS:
    tr, va = split(df, fold)
    m = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300000, sublinear_tf=True)), LogisticRegression(C=10, max_iter=400, class_weight="balanced"))
    m.fit(texts[tr], y[tr])
    s = m.predict_proba(texts[va])[:, 1]
    res[fold] = val_metrics(df.iloc[va], s)
    np.savez(OUT / f"val_tfidf_{fold}.npz", idx=va, score=s)
    print(fold, json.dumps(res[fold]), flush=True)
dump("baseline_tfidf.json", res)
