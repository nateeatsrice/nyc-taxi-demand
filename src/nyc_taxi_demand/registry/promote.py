"""Model registry promotion + Production model loading.

INTENTIONAL legacy stage API. This repo pins ``mlflow<3`` and uses
``transition_model_version_stage`` with the None/Staging/Production lifecycle.
Stages are deprecated in 2.9 but fully functional through 2.x; the targeted
FutureWarning filter in ``common.mlflow_utils`` keeps logs clean. Do NOT migrate
this to aliases without also lifting the version pin.

Promotion policy (simple, automatable):
  - register the best run's model -> creates a new version at stage "None"
  - move it to "Staging"
  - if it beats the current Production model's recorded RMSE (or none exists),
    move it to "Production" and archive the previous Production version.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from nyc_taxi_demand.common.config import Settings, get_settings
from nyc_taxi_demand.common.mlflow_utils import configure_mlflow


def register_run(run_id: str, settings: Settings | None = None) -> int:
    """Register a run's logged model and return the new version number."""
    settings = settings or get_settings()
    configure_mlflow(settings)
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=settings.registered_model_name,
    )
    return int(result.version)


def _production_rmse(client: MlflowClient, model_name: str) -> float | None:
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("rmse")


def promote_best_run(
    run_id: str,
    rmse: float,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Register + stage-promote the given run.

    Returns a dict describing the final stage of the new version.
    """
    settings = settings or get_settings()
    configure_mlflow(settings)
    client = MlflowClient()
    model_name = settings.registered_model_name

    version = register_run(run_id, settings)
    client.transition_model_version_stage(model_name, version, "Staging")

    incumbent = _production_rmse(client, model_name)
    if incumbent is None or rmse < incumbent:
        client.transition_model_version_stage(
            model_name,
            version,
            "Production",
            archive_existing_versions=True,
        )
        final_stage = "Production"
    else:
        final_stage = "Staging"

    return {
        "model": model_name,
        "version": str(version),
        "stage": final_stage,
        "rmse": str(rmse),
        "previous_production_rmse": str(incumbent),
    }


def load_production_model(settings: Settings | None = None):
    """Load the current Production model for serving.

    Raises if no Production version exists -- serving should fail loudly rather
    than silently serve a stale or missing model.
    """
    settings = settings or get_settings()
    configure_mlflow(settings)
    model_uri = f"models:/{settings.registered_model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)
