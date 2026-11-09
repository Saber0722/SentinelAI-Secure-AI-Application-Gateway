"""Transformer cascade: a DeBERTa model trained for this project, plus the ProtectAI guard model.

A prompt is flagged when our model is confident, or when the guard model fires and our model does not
veto it. Our model was trained on benign prompts that use attack vocabulary, which is why it may overrule
the guard. Both agreeing is a BLOCK; one detector alone is a WARN.

The cascade exposes the same interface as ``MlClassifier``. Its score is a decision score, not a probability:
0.95 and above means both models fired, 0.5 and above means one fired, and below 0.5 means neither did.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinelai.detectors.ml_classifier import ClassifierUnavailable, sha256_file
from sentinelai.gateway.decisions import Category, Finding, severity_for
from sentinelai.gateway.normalize import NormalizedText

DETECTOR_NAME = "transformer_cascade"
METADATA_FILE = "metadata.json"
BLOCK_SCORE = 0.95
WARN_SCORE = 0.5
_WINDOW_STEP = 200
_BATCH = 16

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CascadePolicy:
    own_threshold: float
    guard_threshold: float
    veto_threshold: float

    def score(self, own: float, guard: float) -> float:
        """Combine both model probabilities into one decision score."""
        own_fires = own >= self.own_threshold
        guard_fires = guard >= self.guard_threshold and own >= self.veto_threshold
        if own_fires and guard >= self.guard_threshold:
            return BLOCK_SCORE + (1 - BLOCK_SCORE) * min(own, guard)
        if own_fires or guard_fires:
            strength = max(own if own_fires else 0.0, guard if guard_fires else 0.0)
            return WARN_SCORE + (BLOCK_SCORE - WARN_SCORE) * min(strength, 0.999)
        vetoed_guard = min(guard, self.guard_threshold - 1e-6) if own < self.veto_threshold else guard
        return 0.49 * max(own, min(vetoed_guard, 0.999))


def load_detector(settings: Any) -> Any:
    """Load the classifier selected by settings; raises ClassifierUnavailable if it cannot be used."""
    from sentinelai.detectors.ml_classifier import MlClassifier

    if settings.classifier_kind == "cascade":
        return TransformerCascade.load(settings.cascade_dir, settings.torch_device, settings.torch_threads)
    return MlClassifier.load(settings.classifier_dir)


class TransformerCascade:
    name = DETECTOR_NAME

    def __init__(
        self, own: tuple[Any, Any], guard: tuple[Any, Any], policy: CascadePolicy, metadata: dict[str, Any], device: str
    ) -> None:
        self._own_tok, self._own = own
        self._guard_tok, self._guard = guard
        self.policy = policy
        self.metadata = metadata
        self.device = device
        self._guard_index = int(metadata["guard_malicious_index"])
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def block_threshold(self) -> float:
        return BLOCK_SCORE

    @property
    def warn_threshold(self) -> float:
        return WARN_SCORE

    @classmethod
    def load(cls, directory: Path, device: str = "cpu", threads: int = 0) -> TransformerCascade:
        meta_path = directory / METADATA_FILE
        if not meta_path.exists():
            raise ClassifierUnavailable(f"no cascade model in {directory}; run `python scripts/build_cascade.py`")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ClassifierUnavailable(f"cascade needs torch and transformers: {exc}") from None
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        for sub, expected in metadata["weights_sha256"].items():
            if sha256_file(directory / sub / "model.safetensors") != expected:
                raise ClassifierUnavailable(f"cascade weights in {sub}/ do not match metadata.json")
        if threads > 0:
            torch.set_num_threads(threads)

        def part(sub: str) -> tuple[Any, Any]:
            return AutoTokenizer.from_pretrained(directory / sub), AutoModelForSequenceClassification.from_pretrained(
                directory / sub
            ).to(device).eval()

        t = metadata["thresholds"]
        return cls(part("own"), part("guard"), CascadePolicy(t["own"], t["guard"], t["veto"]), metadata, device)

    def _own_probability(self, text: str) -> float:
        """Long prompts are scanned in overlapping windows and the most suspicious window counts."""
        import torch

        tok = self._own_tok
        body = int(self.metadata["own_max_length"]) - 2
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"] or [tok.unk_token_id]
        starts = range(0, max(1, len(ids) - body + _WINDOW_STEP), _WINDOW_STEP) if len(ids) > body else [0]
        windows = [[tok.cls_token_id, *ids[s : s + body], tok.sep_token_id] for s in starts]
        best = 0.0
        for i in range(0, len(windows), _BATCH):
            chunk = windows[i : i + _BATCH]
            longest = max(map(len, chunk))
            x = torch.tensor([w + [tok.pad_token_id] * (longest - len(w)) for w in chunk]).to(self.device)
            logits = self._own(input_ids=x, attention_mask=(x != tok.pad_token_id).long()).logits
            best = max(best, float(torch.softmax(logits.float(), -1)[:, 1].max()))
        return best

    def _guard_probability(self, text: str) -> float:
        import torch

        enc = self._guard_tok(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        return float(torch.softmax(self._guard(**enc).logits.float(), -1)[0, self._guard_index])

    def scores(self, nt: NormalizedText) -> tuple[float, float, float]:
        """Return (decision score, own probability, guard probability)."""
        import torch

        with self._lock, torch.inference_mode():
            own, guard = self._own_probability(nt.text), self._guard_probability(nt.text)
        return self.policy.score(own, guard), own, guard

    def probability(self, nt: NormalizedText) -> float:
        return self.scores(nt)[0]

    def analyze(self, nt: NormalizedText, warn_threshold: float | None = None) -> list[Finding]:
        score, own, guard = self.scores(nt)
        if score < (WARN_SCORE if warn_threshold is None else warn_threshold):
            return []
        return [
            Finding(
                detector=self.name,
                detector_version=self.version,
                category=Category.ML_SUSPICIOUS,
                score=score,
                severity=severity_for(score),
                evidence=f"cascade: own model {own:.2f}, guard model {guard:.2f}",
                rule_ids=("ML:cascade",),
            )
        ]
