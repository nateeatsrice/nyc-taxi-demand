"""Model registry with Optuna search spaces.

Each model is a ``ModelSpec``: a name, a family, whether it supports a Poisson
objective (trip counts are non-negative event counts, so Poisson is the natural
choice where available), and two callables:

  * ``suggest(trial)`` -> a dict of hyperparameters sampled from the model's search
    space for one Optuna trial.
  * ``build(params)``  -> a full sklearn Pipeline (shared preprocessing + estimator)
    configured with those params.

Keeping the preprocessing identical across every model makes the cross-model
comparison apples-to-apples. The only per-model variation is the estimator and its
hyperparameters.

POISSON NOTE: tree models (xgboost/lightgbm/hist_gradient_boosting) and the GLMs
(poisson_regression, and ridge/lasso/elastic_net via their objective) target
non-negative counts. Plain ``linear_regression`` (OLS) and ``svr`` minimize squared
error and CAN predict negatives -- these are kept as honest baselines, and the fact
that they can go negative is itself a useful comparison point. The serving layer
clamps at 0 regardless.

POLYNOMIAL FEATURES: "polynomial regression" is not a separate estimator -- it is a
PolynomialFeatures step in front of a linear model. Rather than a standalone model,
the linear specs expose a ``poly_degree`` hyperparameter (1 = plain) so Optuna can
decide whether polynomial expansion helps. This is the cleaner formulation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from nyc_taxi_demand.features.transform import PRE_KNOWABLE_FEATURES

_CATEGORICAL = ["time_of_day"]
_NUMERIC = [c for c in PRE_KNOWABLE_FEATURES if c not in _CATEGORICAL]

RANDOM_STATE = 42


# --- preprocessing ------------------------------------------------------------
def _preprocessor(scale: bool = False, poly_degree: int = 1) -> ColumnTransformer:
    """Shared preprocessing.

    One-hot encodes ``time_of_day``; passes numeric columns through. Linear/SVR
    models set ``scale=True`` (StandardScaler) and optionally ``poly_degree>1``
    (PolynomialFeatures) on the numeric block; tree models need neither.
    """
    numeric_steps: list[Any] = []
    if poly_degree > 1:
        numeric_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), _CATEGORICAL),
            ("num", numeric_transformer, _NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # "tree" | "linear" | "other"
    poisson: bool
    suggest: Callable[[optuna.Trial], dict[str, Any]]
    build: Callable[[dict[str, Any]], Pipeline]
    # Rough relative cost, for ordering / warning on slow models.
    slow: bool = False


# =============================================================================
# TREE MODELS (Poisson objective, no scaling needed)
# =============================================================================
def _suggest_xgb(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": t.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_xgb(p: dict[str, Any]) -> Pipeline:
    from xgboost import XGBRegressor

    return Pipeline(
        [
            ("pre", _preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="count:poisson",
                    # XGBoost 2.x defaults base_score=None, which collapses
                    # count:poisson to a constant huge value; set it explicitly.
                    # max_delta_step is the documented Poisson overflow safeguard.
                    base_score=0.5,
                    max_delta_step=0.7,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    **p,
                ),
            ),
        ]
    )


def _suggest_lgbm(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "num_leaves": t.suggest_int("num_leaves", 15, 255),
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 100),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_lgbm(p: dict[str, Any]) -> Pipeline:
    from lightgbm import LGBMRegressor

    return Pipeline(
        [
            ("pre", _preprocessor()),
            (
                "model",
                LGBMRegressor(
                    objective="poisson",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                    **p,
                ),
            ),
        ]
    )


def _suggest_hgb(t: optuna.Trial) -> dict[str, Any]:
    return {
        "max_iter": t.suggest_int("max_iter", 100, 800, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 255),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 100),
    }


def _build_hgb(p: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("pre", _preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(loss="poisson", random_state=RANDOM_STATE, **p),
            ),
        ]
    )


# =============================================================================
# LINEAR MODELS (scaled; poly_degree is a hyperparameter)
# =============================================================================
def _suggest_poisson_reg(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "poly_degree": t.suggest_int("poly_degree", 1, 3),
    }


def _build_poisson_reg(p: dict[str, Any]) -> Pipeline:
    p = dict(p)
    degree = p.pop("poly_degree", 1)
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True, poly_degree=degree)),
            ("model", PoissonRegressor(max_iter=1000, **p)),
        ]
    )


def _suggest_linreg(t: optuna.Trial) -> dict[str, Any]:
    return {"poly_degree": t.suggest_int("poly_degree", 1, 3)}


def _build_linreg(p: dict[str, Any]) -> Pipeline:
    degree = dict(p).get("poly_degree", 1)
    # OLS: squared-error, CAN predict negative -- kept as an honest baseline.
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True, poly_degree=degree)),
            ("model", LinearRegression()),
        ]
    )


def _suggest_ridge(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-3, 100.0, log=True),
        "poly_degree": t.suggest_int("poly_degree", 1, 3),
    }


def _build_ridge(p: dict[str, Any]) -> Pipeline:
    p = dict(p)
    degree = p.pop("poly_degree", 1)
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True, poly_degree=degree)),
            ("model", Ridge(random_state=RANDOM_STATE, **p)),
        ]
    )


def _suggest_lasso(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "poly_degree": t.suggest_int("poly_degree", 1, 3),
    }


def _build_lasso(p: dict[str, Any]) -> Pipeline:
    p = dict(p)
    degree = p.pop("poly_degree", 1)
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True, poly_degree=degree)),
            ("model", Lasso(max_iter=5000, random_state=RANDOM_STATE, **p)),
        ]
    )


def _suggest_enet(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
        "poly_degree": t.suggest_int("poly_degree", 1, 3),
    }


def _build_enet(p: dict[str, Any]) -> Pipeline:
    p = dict(p)
    degree = p.pop("poly_degree", 1)
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True, poly_degree=degree)),
            ("model", ElasticNet(max_iter=5000, random_state=RANDOM_STATE, **p)),
        ]
    )


# =============================================================================
# OTHER
# =============================================================================
def _suggest_rf(t: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": t.suggest_int("max_depth", 5, 30),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    }


def _build_rf(p: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("pre", _preprocessor()),
            ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE, **p)),
        ]
    )


def _suggest_svr(t: optuna.Trial) -> dict[str, Any]:
    return {
        "C": t.suggest_float("C", 1e-1, 100.0, log=True),
        "epsilon": t.suggest_float("epsilon", 1e-2, 5.0, log=True),
        "gamma": t.suggest_categorical("gamma", ["scale", "auto"]),
    }


def _build_svr(p: dict[str, Any]) -> Pipeline:
    # SVR: squared-epsilon-insensitive loss; CAN predict negative. Slow on 145k
    # rows -- skippable via --skip-svr / not selecting it.
    return Pipeline(
        [
            ("pre", _preprocessor(scale=True)),
            ("model", SVR(kernel="rbf", **p)),
        ]
    )


# =============================================================================
# REGISTRY
# =============================================================================
MODELS: dict[str, ModelSpec] = {
    # tree (Poisson)
    "xgboost": ModelSpec("xgboost", "tree", True, _suggest_xgb, _build_xgb),
    "lightgbm": ModelSpec("lightgbm", "tree", True, _suggest_lgbm, _build_lgbm),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting", "tree", True, _suggest_hgb, _build_hgb
    ),
    # linear
    "poisson_regression": ModelSpec(
        "poisson_regression", "linear", True, _suggest_poisson_reg, _build_poisson_reg
    ),
    "linear_regression": ModelSpec(
        "linear_regression", "linear", False, _suggest_linreg, _build_linreg
    ),
    "ridge": ModelSpec("ridge", "linear", False, _suggest_ridge, _build_ridge),
    "lasso": ModelSpec("lasso", "linear", False, _suggest_lasso, _build_lasso),
    "elastic_net": ModelSpec("elastic_net", "linear", False, _suggest_enet, _build_enet),
    # other
    "random_forest": ModelSpec("random_forest", "other", False, _suggest_rf, _build_rf, slow=True),
    "svr": ModelSpec("svr", "other", False, _suggest_svr, _build_svr, slow=True),
}

ALL_MODELS: list[str] = list(MODELS)
DEFAULT_MODELS: list[str] = [m for m in ALL_MODELS if m != "svr"]  # svr slow by default


def get_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Options: {ALL_MODELS}")
    return MODELS[name]
