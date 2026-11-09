"""Fit the TF-IDF component on the 85% training part and score the 15% calibration part."""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_pool, split  # noqa: E402

from sentinelai.detectors.ml_classifier import featurize  # noqa: E402

df = load_pool()
tr, va = split(df, "calib")
texts = np.array([featurize(t) for t in df.text], dtype=object)
m = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300000, sublinear_tf=True)), LogisticRegression(C=10, max_iter=400, class_weight="balanced"))
m.fit(texts[tr], df.label.to_numpy()[tr])
np.savez(OUT / "val_tfidf_calib.npz", idx=va, score=m.predict_proba(texts[va])[:, 1])
d = ROOT / "models/research/tfidf_final"
d.mkdir(parents=True, exist_ok=True)
joblib.dump(m, d / "tfidf.joblib")
print("saved", len(tr), "calib", len(va))
