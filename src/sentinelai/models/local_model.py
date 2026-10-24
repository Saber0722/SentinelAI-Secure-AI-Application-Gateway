"""Adapters for locally hosted LLM servers.

* ``OllamaModel`` -- Ollama (llama.cpp-based; manages GGUF quantized models and GPU offload).
* ``OpenAICompatibleModel`` -- any local server exposing ``/v1/chat/completions``, e.g.
  llama.cpp's ``llama-server``.

Both talk HTTP to a loopback address only (enforced in config) and never download models; pull
models explicitly during setup (``ollama pull qwen2.5:1.5b``). Error messages never include
server response bodies, which could echo prompt content.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Literal

import httpx

from sentinelai.environment import detect_nvidia_gpus
from sentinelai.models.base import GenerationConfig, ModelAdapter, ModelError, ModelResponse, ModelStatus

Device = Literal["auto", "gpu", "cpu"]


def _hardware_note() -> str:
    gpus = detect_nvidia_gpus()
    if not gpus:
        return "no NVIDIA GPU detected"
    g = gpus[0]
    return f"{g['name']} ({g['memory_used_mib']}/{g['memory_total_mib']} MiB used)"


class _HttpModel(ModelAdapter):
    def __init__(
        self,
        model_id: str,
        config: GenerationConfig,
        base_url: str,
        device: Device = "auto",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(model_id, config)
        self.base_url = base_url.rstrip("/")
        self.device = device
        self._client = httpx.Client(base_url=self.base_url, transport=transport, trust_env=False)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload, timeout=self.config.timeout_s)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            raise ModelError(f"model did not answer within {self.config.timeout_s:.0f}s") from None
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"model server returned HTTP {exc.response.status_code}") from None
        except httpx.HTTPError:
            raise ModelError(f"model server not reachable at {self.base_url}") from None
        except ValueError:
            raise ModelError("model server returned invalid JSON") from None

    def _get(self, path: str) -> dict[str, Any]:
        resp = self._client.get(path, timeout=3.0)
        resp.raise_for_status()
        return resp.json()


class OllamaModel(_HttpModel):
    backend = "ollama"

    # Ollama caps num_gpu at the model's layer count, so a large value means "every layer on the GPU".
    ALL_LAYERS = 999

    def _options(self, cpu_only: bool) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "num_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
        }
        if cpu_only:
            opts["num_gpu"] = 0  # offload zero layers -> pure CPU inference
        elif self.device == "gpu":
            opts["num_gpu"] = self.ALL_LAYERS
        # "auto" leaves placement to Ollama, which may reuse an instance already loaded on CPU.
        return opts

    def unload(self) -> None:
        """Ask Ollama to drop the model so the next request loads it with fresh device placement."""
        # Nothing loaded or server down: either way the next call reloads.
        with contextlib.suppress(httpx.HTTPError):
            self._client.post("/api/generate", json={"model": self.model_id, "keep_alive": 0}, timeout=30)

    def generate(self, system_prompt: str, prompt: str) -> ModelResponse:
        payload = {
            "model": self.model_id,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": self._options(cpu_only=self.device == "cpu"),
        }
        start = time.perf_counter()
        try:
            data = self._post("/api/chat", payload)
        except ModelError as exc:
            # GPU allocation failures surface as HTTP 500; in auto mode retry once on CPU.
            if self.device != "auto" or "HTTP 500" not in str(exc):
                raise
            payload["options"] = self._options(cpu_only=True)
            data = self._post("/api/chat", payload)
        latency = (time.perf_counter() - start) * 1000
        text = (data.get("message") or {}).get("content")
        if not isinstance(text, str):
            raise ModelError("model server response had no message content")
        return ModelResponse(
            text=text,
            model_id=self.model_id,
            latency_ms=latency,
            device=self.loaded_device(),
            output_tokens=data.get("eval_count"),
        )

    def loaded_device(self) -> str | None:
        try:
            running = self._get("/api/ps").get("models", [])
        except (httpx.HTTPError, ValueError):
            return None
        for m in running:
            if m.get("name") == self.model_id or m.get("model") == self.model_id:
                size, vram = m.get("size") or 0, m.get("size_vram") or 0
                if not vram:
                    return "cpu"
                if size and vram >= size:
                    return "gpu"
                return f"gpu+cpu ({vram / size:.0%} in VRAM)" if size else "gpu"
        return None

    def list_models(self) -> list[str]:
        try:
            return sorted(m["name"] for m in self._get("/api/tags").get("models", []))
        except (httpx.HTTPError, ValueError, KeyError):
            return []

    def status(self) -> ModelStatus:
        try:
            installed = self.list_models() if self._get("/api/version") else []
        except (httpx.HTTPError, ValueError):
            return ModelStatus(
                self.backend,
                self.model_id,
                False,
                f"Ollama not reachable at {self.base_url}",
                extra={"hardware": _hardware_note(), "device_setting": self.device},
            )
        if self.model_id not in installed:
            return ModelStatus(
                self.backend,
                self.model_id,
                False,
                f"model not installed; run `ollama pull {self.model_id}`",
                extra={"installed_models": installed, "hardware": _hardware_note()},
            )
        device = self.loaded_device()
        return ModelStatus(
            self.backend,
            self.model_id,
            True,
            "Ollama model installed" + ("" if device else " (not loaded yet; device chosen at first request)"),
            device=device,
            extra={"hardware": _hardware_note(), "device_setting": self.device},
        )


class OpenAICompatibleModel(_HttpModel):
    backend = "openai_compatible"

    def generate(self, system_prompt: str, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        data = self._post(
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            },
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelError("model server response had no message content") from None
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            model_id=self.model_id,
            latency_ms=(time.perf_counter() - start) * 1000,
            output_tokens=usage.get("completion_tokens"),
        )

    def status(self) -> ModelStatus:
        try:
            models = [m.get("id") for m in self._get("/v1/models").get("data", [])]
        except (httpx.HTTPError, ValueError):
            return ModelStatus(self.backend, self.model_id, False, f"server not reachable at {self.base_url}")
        return ModelStatus(
            self.backend,
            self.model_id,
            True,
            f"server reachable; serving {models}",
            extra={"hardware": _hardware_note(), "device_setting": "configured on the server"},
        )
