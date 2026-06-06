# FastAPI prediction API image. Loads the Production model from the MLflow
# registry (S3-backed) at startup. Linux image -- installs the serve extra.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system --no-cache ".[serve]"

EXPOSE 8000

CMD ["uvicorn", "nyc_taxi_demand.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
