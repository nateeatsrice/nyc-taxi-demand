"""Evaluation metrics and SHAP explainability artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_shap_summary(pipeline, X_sample: pd.DataFrame, mlflow_module) -> None:
    """Compute SHAP values for the tree model and log a summary plot to MLflow.

    Uses TreeExplainer on the fitted model, transforming the input through the
    pipeline's preprocessor first so feature names line up. Best-effort: SHAP
    failures should not kill a training run, so they are caught and logged as a
    tag rather than raised.
    """
    try:
        import matplotlib
        import shap

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pre = pipeline.named_steps["pre"]
        model = pipeline.named_steps["model"]
        X_trans = pre.transform(X_sample)
        feature_names = list(pre.get_feature_names_out())

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=feature_names,
            show=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            mlflow_module.log_artifact(str(out), artifact_path="shap")
    except Exception as exc:  # noqa: BLE001 - explainability is non-fatal
        mlflow_module.set_tag("shap_error", str(exc)[:250])
