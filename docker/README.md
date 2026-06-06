# Docker images

All images are Linux x86_64 and install current Linux wheels — they are **not**
constrained by the macOS 11 laptop policy.

- `training.Dockerfile` — runs the Metaflow training flow on AWS Batch. Installs
  the `orchestration` extra.
- `api.Dockerfile` — FastAPI prediction service (`serve` extra). Loads the
  Production model at startup.
- `ui.Dockerfile` — Streamlit UI (`serve` extra). Thin client over the API.

Build all three with `make build`; push with `make push ECR_URL=<url> TAG=<tag>`.
