# Streamlit UI image. Thin client over the FastAPI service (NTD_API_URL points at
# the in-cluster API service). Linux image -- installs the serve extra.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system --no-cache ".[serve]"

EXPOSE 8501

CMD ["streamlit", "run", "src/nyc_taxi_demand/serving/ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
