# models/

Locally generated model artifacts. Nothing in this directory except this README is committed.

| Path | Produced by | Size | Notes |
|---|---|---|---|
| `classifier/classifier.joblib` | `uv run python scripts/train_classifier.py` | ~0.3 MB | TF-IDF + logistic regression; trains in ~1 s on CPU |
| `classifier/metadata.json` | same | ~5 KB | model version, dataset SHA-256, split seed, thresholds, metrics, artifact SHA-256 |

The gateway verifies the artifact's SHA-256 against `metadata.json` before loading it. joblib files are
pickles and can execute code when loaded: **only load artifacts you trained yourself**.

The local LLM is *not* stored here. It is managed by Ollama (`ollama pull qwen2.5:1.5b`, ~1 GB, stored in
`~/.ollama`). Any GGUF model served by llama.cpp's `llama-server` also works via
`SENTINEL_MODEL_BACKEND=openai_compatible`. No model is ever downloaded automatically.
