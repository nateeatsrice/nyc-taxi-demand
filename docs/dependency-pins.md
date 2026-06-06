# Dependency pins (laptop-safe ceilings)

Local development runs on an **Intel Mac on macOS 11 (Big Sur)**. Every dependency
that runs *locally* must install from a prebuilt wheel for that platform
(`macosx_11_0_x86_64`, `macosx_10_9_x86_64`, or `universal2`) — **no source builds**,
because the machine cannot reliably compile native extensions.

The ceiling is driven by wheel availability for this platform, **not by calendar
year**. Python 3.12 is fine on Big Sur, but some libraries' 3.12 wheels only
appeared in 2024 releases, so capping by date would both over- and under-constrain.

## Binding constraint: numpy / scipy

- **SciPy 1.13+ dropped old-macOS Intel wheels.** Pinned `scipy>=1.11,<1.13`.
- This transitively caps **scikit-learn** (`>=1.4,<1.5`), which in turn fixes the
  versions of xgboost / lightgbm / shap that resolve cleanly against it.
- **numpy** pinned `<2.0` for ABI compatibility with that scientific stack.

## MLflow

Pinned `>=2.9,<3` **on purpose**. This repo uses the legacy stage-based registry
API (`transition_model_version_stage`, stages None/Staging/Production). Stages are
deprecated in 2.9 but fully functional through all of 2.x; they are only removed in
3.x. The `<3` ceiling is what keeps the promotion code working. A targeted
`FutureWarning` filter (in `common/mlflow_utils.py`) silences just the stage
deprecation message — never a blanket filter.

## Local vs cloud split

The macOS-11 ceiling applies **only to the laptop**. Docker images, AWS Batch jobs,
and EKS workloads run on Linux x86_64 and use current Linux wheels. The split is
enforced in `pyproject.toml`:

- **Core `dependencies`** — laptop-safe; installed by `make setup`.
- **Optional extras** — `serve` (FastAPI/Streamlit), `orchestration` (Metaflow),
  `monitoring` (Evidently). These may need newer/source wheels and are installed
  only in Dockerfiles / CI, never locally.

## Before bumping any local pin

1. Check PyPI's "Download files" for the target version and confirm a
   `macosx_11_0_x86_64` (or `_10_9_` / `universal2`) wheel exists.
2. If only a source distribution (`.tar.gz`) is offered for macOS, do **not** bump —
   it will try to compile on Big Sur and likely fail.
3. Re-run `make setup` on the laptop and confirm no build step runs.
