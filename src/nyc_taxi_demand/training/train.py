"""Train + compare all algorithms on the demand task and log to MLflow.

End-to-end flow:
  1. assemble training frame (gold hourly + daily weather)
  2. build leakage-free features (shared module)
  3. temporal split (past -> train, future -> val)
  4. for each algorithm: fit, evaluate on val, log params/metrics/SHAP to MLflow
  5. select best by validation RMSE; register that run's model
  6. return a summary so the caller (CLI / Metaflow flow) can promote it

This runs identically on a laptop (file store) and on AWS Batch (synced store).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd

from nyc_taxi_demand.common.config import Settings, get_settings
from nyc_taxi_demand.common.mlflow_utils import configure_mlflow
from nyc_taxi_demand.data.loader import load_training_frame
from nyc_taxi_demand.data.snapshot import describe_snapshot
from nyc_taxi_demand.data.split import temporal_split
from nyc_taxi_demand.features.transform import TARGET, build_features
from nyc_taxi_demand.training.algorithms import ALGORITHMS, build_estimator
from nyc_taxi_demand.training.evaluate import log_shap_summary, regression_metrics


@dataclass
class RunResult:
    algorithm: str
    run_id: str
    rmse: float
    mae: float
    r2: float


def _train_one(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    snapshot_params: dict[str, str],
) -> RunResult:
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("algorithm", name)
        mlflow.log_params(snapshot_params)

        pipeline = build_estimator(name)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        metrics = regression_metrics(y_val, preds)
        mlflow.log_metrics(metrics)

        # Log the fitted pipeline (preprocessing + model) as the servable artifact.
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=X_val.head(3),
        )
        log_shap_summary(pipeline, X_val.head(500), mlflow)

        return RunResult(
            algorithm=name,
            run_id=run.info.run_id,
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2=metrics["r2"],
        )


def train_and_compare(
    settings: Settings | None = None,
    *,
    year: int | None = None,
    val_fraction: float = 0.2,
) -> list[RunResult]:
    """Run the full compare. Returns results sorted best-first (lowest RMSE)."""
    settings = settings or get_settings()
    configure_mlflow(settings)

    frame = load_training_frame(settings, year=year)
    snapshot = describe_snapshot(
        frame, source_table=settings.table_location_hourly, settings=settings
    )

    features = build_features(frame, derive_calendar=False)
    y = frame[TARGET].reset_index(drop=True)

    # Re-attach date for the temporal split, then split both X and y consistently.
    features = features.assign(pickup_date=frame["pickup_date"].values, _target=y.values)
    train_df, val_df = temporal_split(features, val_fraction=val_fraction)

    feat_cols = [c for c in features.columns if c not in ("pickup_date", "_target")]
    X_train, y_train = train_df[feat_cols], train_df["_target"]
    X_val, y_val = val_df[feat_cols], val_df["_target"]

    results = [
        _train_one(name, X_train, y_train, X_val, y_val, snapshot.as_mlflow_params())
        for name in ALGORITHMS
    ]
    results.sort(key=lambda r: r.rmse)
    return results
