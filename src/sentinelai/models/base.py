"""Model-adapter interface. The gateway depends only on this, never on a concrete backend.

The model is *not* a security control: whatever it returns still goes through output scanning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ModelError(RuntimeError):
    """Inference failed. The message must be safe to show to users (no prompt content)."""


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model_id: str
    latency_ms: float
    device: str | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelStatus:
    backend: str
    model_id: str
    available: bool
    detail: str
    device: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.2
    timeout_s: float = 60.0
    seed: int = 7


class ModelAdapter(ABC):
    backend: str = "abstract"

    def __init__(self, model_id: str, config: GenerationConfig) -> None:
        self.model_id = model_id
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, prompt: str) -> ModelResponse: ...

    @abstractmethod
    def status(self) -> ModelStatus: ...

    def list_models(self) -> list[str]:
        return [self.model_id]
