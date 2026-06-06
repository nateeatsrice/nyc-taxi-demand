"""Optuna-driven hyperparameter search, one study per model.

For a given model spec, run N trials. Each trial:
  1. samples hyperparameters from the model's search space,
  2. builds the pipeline, fits on the temporal-train split,
  3. evaluates validation RMSE,
  4. logs a NESTED MLflow run (params + metrics) under the study's parent run.

The study minimizes validation RMSE. The best trial's params are rebuilt, fit, and
returned so the orchestrator can log the headline run + servable model artifact.

Each model gets its OWN MLflow experiment (nyc-taxi-demand-<model>), so the UI
shows one experiment per model with its trials as runs -- exactly the structure you
asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import optuna
import pandas as pd

from nyc_taxi_demand.training.algorithms import ModelSpec
from nyc_taxi_demand.training.evaluate import regression_metrics

# Optuna is chatty; quiet it so the CLI output stays readable.
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TunedModel:
    model: str
    run_id: str  # the headline (best) run for this model
    rmse: float
    mae: float
    r2: float
    best_params: dict[str, Any]
    n_trials: int


def _objective(
    trial: optuna.Trial,
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    snapshot_params: dict[str, str],
) -> float:
    params = spec.suggest(trial)
    pipeline = spec.build(params)

    with mlflow.start_run(run_name=f"{spec.name}-trial-{trial.number}", nested=True):
        mlflow.set_tag("model", spec.name)
        mlflow.set_tag("model_family", spec.family)
        mlflow.set_tag("poisson", str(spec.poisson))
        mlflow.set_tag("trial_number", trial.number)
        # Log the snapshot (data provenance) + this trial's unique hyperparameters.
        mlflow.log_params(snapshot_params)
        mlflow.log_params({f"hp.{k}": v for k, v in params.items()})

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        metrics = regression_metrics(y_val, preds)
        mlflow.log_metrics(metrics)

    # Report to Optuna; also store run metrics for later inspection.
    trial.set_user_attr("rmse", metrics["rmse"])
    return metrics["rmse"]


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    snapshot_params: dict[str, str],
    *,
    experiment_name: str,
    n_trials: int,
    log_shap: bool = True,
) -> TunedModel:
    """Run an Optuna study for one model in its own MLflow experiment.

    Returns the best trial rebuilt + logged as the headline run (with the servable
    model artifact attached).
    """
    mlflow.set_experiment(experiment_name)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Parent run groups all trial runs for this model.
    with mlflow.start_run(run_name=f"{spec.name}-study"):
        mlflow.set_tag("model", spec.name)
        mlflow.set_tag("phase", "search")

        study.optimize(
            lambda t: _objective(t, spec, X_train, y_train, X_val, y_val, snapshot_params),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_rmse", study.best_value)

        # Rebuild + fit the best config; this run carries the servable artifact.
        best_pipeline = spec.build(best_params)
        best_pipeline.fit(X_train, y_train)
        preds = best_pipeline.predict(X_val)
        metrics = regression_metrics(y_val, preds)

        with mlflow.start_run(run_name=f"{spec.name}-best", nested=True) as best_run:
            mlflow.set_tag("model", spec.name)
            mlflow.set_tag("phase", "best")
            mlflow.log_params(snapshot_params)
            mlflow.log_params({f"hp.{k}": v for k, v in best_params.items()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                best_pipeline, artifact_path="model", input_example=X_val.head(3)
            )
            if log_shap and spec.family == "tree":
                from nyc_taxi_demand.training.evaluate import log_shap_summary

                log_shap_summary(best_pipeline, X_val.head(500), mlflow)
            best_run_id = best_run.info.run_id

    return TunedModel(
        model=spec.name,
        run_id=best_run_id,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        best_params=best_params,
        n_trials=n_trials,
    )
