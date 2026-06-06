"""MLflow setup helpers.

This repo runs MLflow with a **file/SQLite backend synced to S3** -- there is no
always-on tracking server and no RDS. Training writes runs to a local store
directory, and that store (plus artifacts) is synced to S3 under ``mlruns/`` and
``ml-artifacts/``. The ``make mlflow-ui`` target syncs the store back down before
launching the UI, so history persists in S3 across machines.

CONCURRENCY CAVEAT (accepted design hole #2): the SQLite/file store synced to S3
is last-write-wins. Run training **serially**. Two overlapping Batch runs writing
the same synced store can lose runs or corrupt the SQLite file. This is fine for a
single-operator portfolio platform; documented so it is a conscious choice.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow

from nyc_taxi_demand.common.config import Settings, get_settings

# The model-registry STAGE API (None/Staging/Production) is deprecated in MLflow
# 2.9 but used intentionally here (pin: mlflow<3). Silence only that specific
# FutureWarning -- never a blanket filter.
warnings.filterwarnings(
    "ignore",
    message=r".*Model registry stages will be removed.*",
    category=FutureWarning,
)


def configure_mlflow(settings: Settings | None = None) -> Settings:
    """Point MLflow at the local synced store + S3 artifact root and set the
    active experiment. Call once at the start of any training/registry process.
    """
    settings = settings or get_settings()

    store_dir = Path(settings.mlflow_local_store).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    # SQLite backend inside the synced store dir -> enables the model registry.
    tracking_uri = f"sqlite:///{store_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(
        experiment_name=settings.mlflow_experiment,
        # Artifacts go straight to S3; the DB only holds metadata.
    )
    return settings
