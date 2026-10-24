"""Process-wide application state, built once at startup from ``Settings``."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from sentinelai.audit import AuditLogger
from sentinelai.config import Settings
from sentinelai.detectors.ml_classifier import ClassifierUnavailable, MlClassifier
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.evaluation.report import compact, record_runs, run_suite
from sentinelai.gateway.pipeline import SecurityGateway
from sentinelai.models import GenerationConfig, ModelAdapter, UnavailableModel, build_model
from sentinelai.storage.database import Database

log = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    db: Database
    audit: AuditLogger
    gateway: SecurityGateway
    classifier_error: str | None = None
    started_at: float = field(default_factory=time.time)
    _eval_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def build(cls, settings: Settings, model: ModelAdapter | None = None) -> AppState:
        db = Database(settings.db_path)
        audit = AuditLogger(db, settings.var_dir / "audit_hmac.key", settings.audit_store_redacted_prompt)
        secrets = SecretsConfig.load(settings.protected_secrets_path)
        if not secrets.secrets:
            log.warning("no protected secrets configured at %s", settings.protected_secrets_path)

        classifier, classifier_error = None, None
        try:
            classifier = MlClassifier.load(settings.classifier_dir)
        except ClassifierUnavailable as exc:
            classifier_error = str(exc)
        policy = settings.initial_policy()
        if policy.classifier_enabled and classifier is None:
            log.info("classifier unavailable (%s); continuing with rules only", classifier_error)
            policy = policy.model_copy(update={"classifier_enabled": False})

        if model is None:
            config = GenerationConfig(
                settings.max_output_tokens, settings.temperature, settings.model_timeout_s, settings.seed
            )
            try:
                model = build_model(
                    settings.model_backend, settings.model_name, settings.model_base_url, settings.inference_device, config
                )
            except ValueError as exc:
                model = UnavailableModel(settings.model_name, str(exc))

        pii = PiiDetector(settings.spacy_model)
        pii.warm_up()
        gateway = SecurityGateway(
            policy, model, audit, secrets, pii=pii, classifier=classifier, max_input_chars=settings.max_input_chars
        )
        return cls(settings=settings, db=db, audit=audit, gateway=gateway, classifier_error=classifier_error)

    def classifier_status(self) -> dict[str, Any]:
        clf = self.gateway.classifier
        if clf is None:
            return {"available": False, "enabled": False, "detail": self.classifier_error}
        policy = self.gateway.policy
        return {
            "available": True,
            "enabled": policy.classifier_enabled,
            "version": clf.version,
            "thresholds": {
                "block": policy.classifier_block_threshold or clf.block_threshold,
                "warn": policy.classifier_warn_threshold or clf.warn_threshold,
            },
            "test_metrics": clf.metadata.get("test"),
            "validation": clf.metadata.get("validation"),
            "external_evaluation": clf.metadata.get("external_evaluation"),
            "splits": clf.metadata.get("splits"),
            "algorithm": clf.metadata.get("algorithm"),
            "dataset": clf.metadata.get("dataset"),
            "trained_at": clf.metadata.get("trained_at"),
        }

    def run_evaluation(self, include_classifier: bool = True) -> dict[str, Any]:
        """Run the reproducible benchmark (mock model, no audit pollution) and store the runs."""
        with self._eval_lock:
            suite = run_suite(
                self.settings.evaluation_dir,
                self.gateway.secrets_config,
                pii=self.gateway.pii,
                classifier=self.gateway.classifier,
                include_classifier=include_classifier,
            )
            record_runs(self.audit, suite)
            return compact(suite)
