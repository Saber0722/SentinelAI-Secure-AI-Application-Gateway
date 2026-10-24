# Local inference benchmark (GPU vs CPU)

Generated 2026-10-23T08:19:59+00:00 by `uv run python scripts/benchmark_inference.py`. Model `qwen2.5:1.5b` via Ollama, 20 benign prompts per device (1 cold + warm runs), max 128 output tokens, temperature 0, full security pipeline enabled.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, driver 615.78.08).

| Device | Placement | Cold request | Inference p50 / p95 | Tokens/s (p50) | Gateway p50 / p95 | VRAM for model |
|---|---|---|---|---|---|---|
| `gpu` | gpu | 2.40 s | 458 / 715 ms | 176.5 | 13.4 / 23.4 ms | 1238 MiB |
| `cpu` | cpu | 2.33 s | 2107 / 3948 ms | 32.1 | 17.2 / 29.8 ms | 0 MiB |

GPU throughput is **5.5x** CPU (tokens/s, p50); warm inference latency drops by 78% (p50).

The gateway's own processing time is unaffected by the inference device: the detectors (Presidio, rules,
classifier) always run on CPU. Responses differ in length, so tokens/s is the fairer throughput measure.
