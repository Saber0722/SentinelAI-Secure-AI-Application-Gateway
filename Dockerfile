# SentinelAI gateway + UI image. The same image runs both services (see compose.yaml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# Project code and data. The install stays editable so PROJECT_ROOT resolves to /app.
COPY src ./src
COPY data ./data
COPY results ./results
COPY .streamlit ./.streamlit
COPY models/classifier ./models/classifier
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 sentinel \
    && mkdir -p /app/var \
    && chown -R sentinel:sentinel /app/var /app/results
USER sentinel

EXPOSE 8000 8501
CMD ["sentinelai", "serve", "--host", "0.0.0.0", "--port", "8000"]
