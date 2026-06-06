# Training image -- runs on AWS Batch (Linux x86_64). NOT subject to the macOS 11
# laptop constraint, so it installs current Linux wheels including the cloud extras
# (Metaflow for the flow). Multi-stage to keep the final image lean.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first for caching.
COPY pyproject.toml README.md ./
COPY src ./src
COPY flows ./flows

# Install core + orchestration extra (training needs Metaflow; not serving/UI).
RUN uv pip install --system --no-cache ".[orchestration]"

# Default: run the Metaflow training flow on Batch.
ENTRYPOINT ["python", "flows/training_flow.py"]
CMD ["run"]
