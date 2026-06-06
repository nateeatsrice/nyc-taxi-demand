"""The three algorithms compared on the demand task, each wrapped in an identical
preprocessing pipeline so the comparison is apples-to-apples.

Only ``time_of_day`` needs encoding (one-hot); the rest are already numeric or
boolean. The same ColumnTransformer is reused for every estimator so that the
feature space is identical across models.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nyc_taxi_demand.features.transform import PRE_KNOWABLE_FEATURES

_CATEGORICAL = ["time_of_day"]
_PASSTHROUGH = [c for c in PRE_KNOWABLE_FEATURES if c not in _CATEGORICAL]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def _xgb():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def _lgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
    )


# Registry of algorithm name -> factory. The flow/train loop iterates this.
ALGORITHMS: dict[str, Callable[[], object]] = {
    "xgboost": _xgb,
    "lightgbm": _lgbm,
    "hist_gradient_boosting": _hgb,
}


def build_estimator(name: str) -> Pipeline:
    """Build a full preprocessing+model pipeline for the named algorithm."""
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Options: {list(ALGORITHMS)}")
    return Pipeline(
        steps=[
            ("pre", _preprocessor()),
            ("model", ALGORITHMS[name]()),
        ]
    )
