"""PII detection with Microsoft Presidio (spaCy ``en_core_web_sm``, runs fully offline).

Presidio's results are probabilistic: NER-based entities (PERSON, LOCATION) depend on the small
spaCy model and English text, and pattern recognizers miss unusual formats. Treat the output as
a strong signal, not a guarantee. See docs/limitations.md.
"""

from __future__ import annotations

import logging
import re
import threading
from importlib.metadata import version as pkg_version

from sentinelai.gateway.decisions import Category, Finding, Span, severity_for

DETECTOR_NAME = "presidio_pii"
SPACY_MODEL = "en_core_web_sm"
# spaCy's small model tags many capitalised non-names ("OAuth", "Convert 25 degrees") as PERSON.
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z]+(?:['’-][A-Za-z]+)*|[A-Z]['’][A-Z][a-z]+|[A-Z]\.)$")

log = logging.getLogger(__name__)


class PiiDetector:
    name = DETECTOR_NAME

    def __init__(self, spacy_model: str = SPACY_MODEL) -> None:
        self.spacy_model = spacy_model
        self.version = f"presidio-{pkg_version('presidio-analyzer')}/{spacy_model}"
        self._analyzer = None
        self._lock = threading.Lock()

    def _engine(self):
        if self._analyzer is None:
            with self._lock:
                if self._analyzer is None:
                    self._analyzer = _build_analyzer(self.spacy_model)
        return self._analyzer

    def warm_up(self) -> None:
        self.analyze("warm up", ["PERSON"], 0.5)

    def analyze(self, text: str, entities: list[str], min_score: float) -> list[Finding]:
        if not text or not entities:
            return []
        engine = self._engine()
        with self._lock:  # spaCy pipelines are not guaranteed thread-safe
            results = engine.analyze(text=text, language="en", entities=list(entities))
        by_type: dict[str, list[tuple[Span, float]]] = {}
        for r in results:
            if r.score < min_score:
                continue
            if r.entity_type == "PERSON" and not all(_NAME_TOKEN.match(t) for t in text[r.start : r.end].split()):
                continue
            by_type.setdefault(r.entity_type, []).append((Span(r.start, r.end, r.entity_type), r.score))

        findings = []
        for entity, items in sorted(by_type.items()):
            score = max(s for _, s in items)
            findings.append(
                Finding(
                    detector=self.name,
                    detector_version=self.version,
                    category=Category.PII,
                    score=score,
                    severity=severity_for(score),
                    evidence=f"{len(items)} {entity} entit{'y' if len(items) == 1 else 'ies'} (max score {score:.2f})",
                    rule_ids=(f"PII:{entity}",),
                    spans=tuple(span for span, _ in items),
                    entity_type=entity,
                )
            )
        return findings


def _build_analyzer(spacy_model: str):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer, InPanRecognizer

    # Presidio logs every recognizer it loads at INFO; keep startup output readable.
    logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    analyzer.registry.add_recognizer(InPanRecognizer())
    analyzer.registry.add_recognizer(InAadhaarRecognizer())
    log.info("Presidio analyzer ready (spaCy model %s)", spacy_model)
    return analyzer
