# ---------------------------------------------------------------------------
# Sentinel model server image.
# Used by docker-compose.yml and by the deploy workflow.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SENTINEL_ENV=production

WORKDIR /app

# System deps are minimal; scikit-learn/numpy wheels suffice on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

# Runtime dirs (mounted as volumes in compose so registry/routing/logs are
# shared with the controller process).
RUN mkdir -p data/raw data/processed data/synthetic models artifacts monitoring reports

# Non-root user
RUN useradd --create-home --uid 10001 sentinel \
    && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "sentinel.cli", "serve"]
