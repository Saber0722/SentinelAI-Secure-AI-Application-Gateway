"""Optional TF-IDF + logistic-regression prompt classifier (trained by scripts/train_classifier.py).

The artifact is a joblib (pickle) file. Pickles can execute code when loaded, so only load
artifacts you trained locally; the loader verifies the SHA-256 recorded at training time to catch
accidental corruption or swapping, which is an integrity check, not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelai.gateway.decisions import Category, Finding, severity_for
from sentinelai.gateway.normalize import NormalizedText, make_skeleton, normalize

DETECTOR_NAME = "ml_classifier"
MODEL_FILE = "classifier.joblib"
METADATA_FILE = "metadata.json"

log = logging.getLogger(__name__)


def classifier_text(nt: NormalizedText) -> str:
    """Feature text shared by training and inference: de-obfuscated skeleton + decoded payloads."""
    return " ".join([nt.skeleton, *(make_skeleton(s) for s in nt.decoded_segments)])


def featurize(raw: str) -> str:
    return classifier_text(normalize(raw))


class ClassifierUnavailable(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(slots=True)
class MlClassifier:
    pipeline: Any
    metadata: dict[str, Any]
    name: str = DETECTOR_NAME

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def block_threshold(self) -> float:
        return float(self.metadata["thresholds"]["block"])

    @property
    def warn_threshold(self) -> float:
        return float(self.metadata["thresholds"]["warn"])

    @classmethod
    def load(cls, directory: Path) -> MlClassifier:
        model_path = directory / MODEL_FILE
        meta_path = directory / METADATA_FILE
        if not model_path.exists() or not meta_path.exists():
            raise ClassifierUnavailable(
                f"no trained classifier in {directory}; run `uv run python scripts/train_classifier.py`"
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        digest = sha256_file(model_path)
        if digest != metadata.get("artifact_sha256"):
            raise ClassifierUnavailable("classifier artifact checksum does not match metadata.json")
        import joblib

        return cls(pipeline=joblib.load(model_path), metadata=metadata)

    def probability(self, nt: NormalizedText) -> float:
        return float(self.pipeline.predict_proba([classifier_text(nt)])[0][1])

    def analyze(self, nt: NormalizedText, warn_threshold: float | None = None) -> list[Finding]:
        p = self.probability(nt)
        warn = self.warn_threshold if warn_threshold is None else warn_threshold
        if p < warn:
            return []
        return [
            Finding(
                detector=self.name,
                detector_version=self.version,
                category=Category.ML_SUSPICIOUS,
                score=p,
                severity=severity_for(p),
                evidence=f"classifier probability {p:.2f} that the prompt is malicious",
                rule_ids=("ML:malicious",),
            )
        ]
