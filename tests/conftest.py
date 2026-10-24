from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinelai.api.app import create_app
from sentinelai.audit import AuditLogger, NullAuditLogger
from sentinelai.config import PROJECT_ROOT, SecurityPolicy, Settings
from sentinelai.detectors.pii import PiiDetector
from sentinelai.detectors.secrets import SecretsConfig
from sentinelai.evaluation.dataset import load_dataset
from sentinelai.gateway.pipeline import SecurityGateway
from sentinelai.models import MockModel, ModelAdapter
from sentinelai.storage.database import Database

ADMIN_TOKEN = "test-admin-token-0123456789"


@pytest.fixture(scope="session")
def secrets_config() -> SecretsConfig:
    return SecretsConfig.load(PROJECT_ROOT / "data" / "protected_secrets.json")


@pytest.fixture(scope="session")
def protected_values(secrets_config: SecretsConfig) -> list[str]:
    return [v for _, v in secrets_config.all_values]


@pytest.fixture(scope="session")
def pii_detector() -> PiiDetector:
    detector = PiiDetector()
    detector.warm_up()
    return detector


@pytest.fixture(scope="session")
def dev_dataset():
    return load_dataset(PROJECT_ROOT / "data" / "evaluation")


@pytest.fixture(scope="session")
def holdout_dataset():
    return load_dataset(PROJECT_ROOT / "data" / "evaluation" / "holdout")


@pytest.fixture
def make_gateway(secrets_config: SecretsConfig, pii_detector: PiiDetector) -> Callable[..., SecurityGateway]:
    def factory(
        policy: SecurityPolicy | None = None,
        model: ModelAdapter | None = None,
        audit: AuditLogger | None = None,
        **kwargs,
    ) -> SecurityGateway:
        return SecurityGateway(
            policy or SecurityPolicy(),
            model or MockModel(),
            audit or NullAuditLogger(),
            secrets_config,
            pii=pii_detector,
            **kwargs,
        )

    return factory


@pytest.fixture
def memory_audit() -> AuditLogger:
    return AuditLogger(Database(":memory:"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        var_dir=tmp_path,
        db_path=tmp_path / "sentinel.db",
        admin_token=ADMIN_TOKEN,
        classifier_dir=tmp_path / "no-classifier",
        _env_file=None,
    )


@pytest.fixture
def client_factory(settings: Settings) -> Callable[..., TestClient]:
    clients: list[TestClient] = []

    def factory(model: ModelAdapter | None = None, **overrides) -> TestClient:
        s = settings.model_copy(update=overrides) if overrides else settings
        client = TestClient(create_app(s, model=model), raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(client_factory) -> Iterator[TestClient]:
    yield client_factory()


@pytest.fixture
def admin_headers() -> dict[str, str]:
    return {"X-Admin-Token": ADMIN_TOKEN}
