"""Application settings (environment / .env) and the runtime security policy."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class PiiAction(StrEnum):
    ALLOW = "ALLOW"
    REDACT = "REDACT"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class OutputSecretAction(StrEnum):
    BLOCK = "BLOCK"
    REDACT = "REDACT"


DEFAULT_PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IN_PAN",
    "IN_AADHAAR",
    "PERSON",
]


# Entities the configured Presidio analyzer can detect (see detectors/pii.py).
SUPPORTED_PII_ENTITIES = sorted(
    {
        *DEFAULT_PII_ENTITIES,
        "LOCATION",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "US_BANK_NUMBER",
        "UK_NHS",
        "MEDICAL_LICENSE",
        "CRYPTO",
        "URL",
        "DATE_TIME",
        "NRP",
        "ORGANIZATION",
    }
)


class SecurityPolicy(BaseModel):
    """Server-side security policy. Clients can never supply or override this per request."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    input_rules_enabled: bool = True
    injection_block_threshold: float = Field(0.70, ge=0.0, le=1.0)
    injection_warn_threshold: float = Field(0.35, ge=0.0, le=1.0)

    pii_input_enabled: bool = True
    pii_input_action: PiiAction = PiiAction.REDACT
    pii_output_action: PiiAction = PiiAction.REDACT
    pii_entities: list[str] = Field(default_factory=lambda: list(DEFAULT_PII_ENTITIES))
    # 0.4 admits phone numbers validated by libphonenumber but lacking a context word ("phone").
    pii_min_score: float = Field(0.4, ge=0.0, le=1.0)

    input_secret_action: PiiAction = PiiAction.REDACT

    classifier_enabled: bool = False
    # None means "use the threshold selected on validation data and stored with the model".
    classifier_block_threshold: float | None = Field(None, ge=0.0, le=1.0)
    classifier_warn_threshold: float | None = Field(None, ge=0.0, le=1.0)

    output_scanning_enabled: bool = True
    output_secret_action: OutputSecretAction = OutputSecretAction.BLOCK
    output_denylist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> SecurityPolicy:
        if self.injection_warn_threshold > self.injection_block_threshold:
            raise ValueError("injection_warn_threshold must not exceed injection_block_threshold")
        return self

    @field_validator("pii_entities")
    @classmethod
    def _known_entities(cls, entities: list[str]) -> list[str]:
        unknown = sorted(set(entities) - set(SUPPORTED_PII_ENTITIES))
        if unknown:
            raise ValueError(f"unsupported PII entities: {unknown}")
        return entities

    @field_validator("output_denylist")
    @classmethod
    def _valid_regexes(cls, patterns: list[str]) -> list[str]:
        import re

        for p in patterns:
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"invalid denylist regex: {exc}") from None
        return patterns

    @classmethod
    def disabled(cls) -> SecurityPolicy:
        """Every control off. Used only by the evaluation runner as the unprotected baseline."""
        return cls(
            input_rules_enabled=False,
            pii_input_enabled=False,
            pii_input_action=PiiAction.ALLOW,
            pii_output_action=PiiAction.ALLOW,
            input_secret_action=PiiAction.ALLOW,
            classifier_enabled=False,
            output_scanning_enabled=False,
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    var_dir: Path = PROJECT_ROOT / "var"
    db_path: Path = PROJECT_ROOT / "var" / "sentinel.db"
    protected_secrets_path: Path = PROJECT_ROOT / "data" / "protected_secrets.json"
    classifier_dir: Path = PROJECT_ROOT / "models" / "classifier"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    results_dir: Path = PROJECT_ROOT / "results"

    model_backend: Literal["mock", "ollama", "openai_compatible"] = "mock"
    model_name: str = "qwen2.5:1.5b"
    model_base_url: str = "http://127.0.0.1:11434"
    allow_remote_model: bool = False
    inference_device: Literal["auto", "gpu", "cpu"] = "auto"
    model_timeout_s: float = Field(60.0, gt=0)
    max_output_tokens: int = Field(256, gt=0, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    seed: int = 7

    max_input_chars: int = Field(8000, gt=0)
    # Larger spaCy models (en_core_web_lg) give better PERSON/LOCATION recall; install explicitly.
    spacy_model: str = "en_core_web_sm"
    # Used only if a trained artifact exists in classifier_dir; otherwise rules-only.
    classifier_enabled: bool = True
    audit_store_redacted_prompt: bool = False
    admin_token: SecretStr | None = None

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @model_validator(mode="after")
    def _model_url_is_local(self) -> Settings:
        check_model_url(self.model_base_url, self.allow_remote_model)
        return self

    def initial_policy(self) -> SecurityPolicy:
        return SecurityPolicy(classifier_enabled=self.classifier_enabled)


def check_model_url(url: str, allow_remote: bool) -> None:
    """Prompts must not leave the machine unless the operator explicitly allows it."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("model_base_url must be an http(s) URL")
    if parsed.hostname not in LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(
            "model_base_url must point at a loopback address; set SENTINEL_ALLOW_REMOTE_MODEL=true "
            "to explicitly allow a non-local model server"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
