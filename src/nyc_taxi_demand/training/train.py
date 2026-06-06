"""Orchestrate hyperparameter search across multiple models and pick a winner.

End-to-end:
  1. assemble training frame (gold hourly + daily weather)  [once]
  2. build leakage-free features (shared module)            [once]
  3. temporal split (past -> train, future -> val)          [once]
  4. for each requested model: run an Optuna study of N trials in its OWN MLflow
     experiment (nyc-taxi-demand-<model>), logging every trial as a nested run
  5. pick the global best across all models' best runs (lowest validation RMSE)
  6. caller registers + promotes that run

The data load + split happen once and are reused for every model so the comparison
is fair and we don't re-query Athena per model.

Runs identically on a laptop (file store) and on AWS Batch (synced store).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nyc_taxi_demand.common.config import PROJECT, Settings, get_settings
from nyc_taxi_demand.common.mlflow_utils import configure_mlflow
from nyc_taxi_demand.data.loader import load_training_frame
from nyc_taxi_demand.data.snapshot import describe_snapshot
from nyc_taxi_demand.data.split import temporal_split
from nyc_taxi_demand.features.transform import TARGET, build_features
from nyc_taxi_demand.training.algorithms import DEFAULT_MODELS, get_spec
from nyc_taxi_demand.training.tuning import TunedModel, tune_model


@dataclass
class TrainSummary:
    results: list[TunedModel]  # one per model, sorted best-first
    best: TunedModel


def _prepare_splits(
    settings: Settings, year: int | None, val_fraction: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, str]]:
    frame = load_training_frame(settings, year=year)
    snapshot = describe_snapshot(
        frame, source_table=settings.table_location_hourly, settings=settings
    )

    features = build_features(frame, derive_calendar=False)
    y = frame[TARGET].reset_index(drop=True)

    features = features.assign(pickup_date=frame["pickup_date"].values, _target=y.values)
    train_df, val_df = temporal_split(features, val_fraction=val_fraction)

    feat_cols = [c for c in features.columns if c not in ("pickup_date", "_target")]
    return (
        train_df[feat_cols],
        train_df["_target"],
        val_df[feat_cols],
        val_df["_target"],
        snapshot.as_mlflow_params(),
    )


def train_and_compare(
    settings: Settings | None = None,
    *,
    models: list[str] | None = None,
    n_trials: int = 50,
    year: int | None = None,
    val_fraction: float = 0.2,
) -> TrainSummary:
    """Tune each requested model with Optuna; return results sorted best-first.

    Parameters
    ----------
    models:
        Which models to tune. Defaults to all except the slow SVR. Each name must
        exist in the model registry (see algorithms.MODELS).
    n_trials:
        Optuna trials per model.
    """
    settings = settings or get_settings()
    configure_mlflow(settings)

    model_names = models or list(DEFAULT_MODELS)
    specs = [get_spec(n) for n in model_names]

    X_train, y_train, X_val, y_val, snapshot_params = _prepare_splits(settings, year, val_fraction)

    results: list[TunedModel] = []
    for spec in specs:
        tuned = tune_model(
            spec,
            X_train,
            y_train,
            X_val,
            y_val,
            snapshot_params,
            experiment_name=f"{PROJECT}-{spec.name}",
            n_trials=n_trials,
        )
        results.append(tuned)

    results.sort(key=lambda r: r.rmse)
    return TrainSummary(results=results, best=results[0])
