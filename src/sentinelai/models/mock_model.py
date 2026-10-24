"""Deterministic models for demos, tests and the reproducible benchmark.

``MockModel`` deliberately simulates a *vulnerable* LLM: when asked to reveal its instructions
or secrets it complies, and otherwise it echoes the prompt it received. That makes the effect of
the gateway measurable -- without the gateway, attacks leak secrets and PII reaches the "model".
"""

from __future__ import annotations

import re
import time
from collections import deque

from sentinelai.models.base import GenerationConfig, ModelAdapter, ModelError, ModelResponse, ModelStatus

_LEAK_VERB = re.compile(
    r"\b(ignore|disregard|forget|reveal|print|show|display|output|repeat|tell|dump|list|leak|"
    r"what (are|is|were)|what's|give|share|recite|expose|disclose)\b"
)
_LEAK_OBJECT = re.compile(
    r"\b(?:your|the|all|any|its)\s+(?:\w+\s+){0,2}?(system prompt|instructions|secrets?|passwords?|"
    r"api[ _]keys?|credentials|configuration|config|tokens?|hidden rules|internal\w*|initial prompt|"
    r"text above|rules)\b|\b(everything above|verbatim|internal_api_key|demo_db_password)\b"
)
_JAILBREAK = re.compile(
    r"\b(dan|do anything now|developer mode|jailbr(eak|oken)|unrestricted|no restrictions|unfiltered|"
    r"uncensored|evil)\b"
)


class MockModel(ModelAdapter):
    backend = "mock"

    def __init__(self, config: GenerationConfig | None = None, latency_ms: float = 0.0) -> None:
        super().__init__("mock-vulnerable-v1", config or GenerationConfig())
        self.latency_ms = latency_ms
        # Bounded record of prompts received, used by tests to prove what reached the model.
        self.received: deque[str] = deque(maxlen=200)

    @property
    def call_count(self) -> int:
        return len(self.received)

    def generate(self, system_prompt: str, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        self.received.append(prompt)
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        prompt = re.sub(r"</?user_input>", "", prompt).strip()
        lowered = prompt.casefold()
        if _LEAK_VERB.search(lowered) and _LEAK_OBJECT.search(lowered):
            text = "Of course! Here is my complete configuration:\n\n" + system_prompt
        elif _JAILBREAK.search(lowered):
            text = "Jailbreak mode enabled. I no longer follow any rules. " + system_prompt.splitlines()[0]
        else:
            text = f'[mock model] I received your message ({len(prompt)} chars): "{prompt[:500]}"'
        return ModelResponse(
            text=text,
            model_id=self.model_id,
            latency_ms=(time.perf_counter() - start) * 1000,
            device="cpu",
            output_tokens=len(text.split()),
        )

    def status(self) -> ModelStatus:
        return ModelStatus(
            backend=self.backend,
            model_id=self.model_id,
            available=True,
            detail="Deterministic mock model (simulates a vulnerable LLM; no inference performed)",
            device="cpu",
        )


class ScriptedModel(ModelAdapter):
    """Returns a preset response. Used to test output scanning independently of input scanning."""

    backend = "scripted"

    def __init__(self, response: str = "") -> None:
        super().__init__("scripted-v1", GenerationConfig())
        self.response = response
        self.received: deque[str] = deque(maxlen=200)

    def generate(self, system_prompt: str, prompt: str) -> ModelResponse:
        self.received.append(prompt)
        return ModelResponse(text=self.response, model_id=self.model_id, latency_ms=0.0, device="cpu")

    def status(self) -> ModelStatus:
        return ModelStatus(self.backend, self.model_id, True, "Scripted test model", "cpu")


class UnavailableModel(ModelAdapter):
    """Stands in when the configured backend cannot be constructed; every call fails safely."""

    backend = "unavailable"

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(model_id, GenerationConfig())
        self.reason = reason

    def generate(self, system_prompt: str, prompt: str) -> ModelResponse:
        raise ModelError(f"model unavailable: {self.reason}")

    def status(self) -> ModelStatus:
        return ModelStatus(self.backend, self.model_id, False, self.reason)
