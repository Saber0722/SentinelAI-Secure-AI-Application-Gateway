"""Model adapters and a factory that builds one from settings."""

from __future__ import annotations

from sentinelai.models.base import GenerationConfig, ModelAdapter, ModelError, ModelResponse, ModelStatus
from sentinelai.models.mock_model import MockModel, ScriptedModel, UnavailableModel

__all__ = [
    "GenerationConfig",
    "MockModel",
    "ModelAdapter",
    "ModelError",
    "ModelResponse",
    "ModelStatus",
    "ScriptedModel",
    "UnavailableModel",
    "build_model",
]


def build_model(backend: str, model_name: str, base_url: str, device: str, config: GenerationConfig) -> ModelAdapter:
    if backend == "mock":
        return MockModel(config)
    from sentinelai.models.local_model import OllamaModel, OpenAICompatibleModel

    if backend == "ollama":
        return OllamaModel(model_name, config, base_url, device)  # type: ignore[arg-type]
    if backend == "openai_compatible":
        return OpenAICompatibleModel(model_name, config, base_url, device)  # type: ignore[arg-type]
    raise ValueError(f"unknown model backend {backend!r}")
