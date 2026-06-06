"""Tests for the model registry + search spaces."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from nyc_taxi_demand.features.transform import PRE_KNOWABLE_FEATURES
from nyc_taxi_demand.training.algorithms import (
    ALL_MODELS,
    DEFAULT_MODELS,
    MODELS,
    get_spec,
)


def _synthetic(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "pickup_location_id": rng.integers(1, 266, n),
            "pickup_hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n).astype(bool),
            "time_of_day": rng.choice(["morning", "afternoon", "evening", "night"], n),
            "temp_avg_fahrenheit": rng.uniform(20, 90, n),
            "is_rainy": rng.integers(0, 2, n).astype(bool),
        }
    )[PRE_KNOWABLE_FEATURES]
    y = pd.Series(rng.poisson(50, n))
    return X, y


def test_registry_has_expected_models():
    assert len(MODELS) == 10
    assert "svr" in ALL_MODELS
    assert "svr" not in DEFAULT_MODELS  # excluded by default (slow)


def test_get_spec_rejects_unknown():
    with pytest.raises(KeyError):
        get_spec("not_a_model")


@pytest.mark.parametrize("name", DEFAULT_MODELS)
def test_each_model_fits_and_predicts(name):
    """Every default model builds, fits, and predicts on the feature contract."""
    X, y = _synthetic()
    spec = get_spec(name)

    # Sample params from the search space via a fixed-seed trial.
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    params = spec.suggest(study.ask())
    pipeline = spec.build(params)

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("name", [m for m, s in MODELS.items() if s.poisson])
def test_poisson_models_never_predict_negative(name):
    """Poisson-objective models must not produce negative counts."""
    X, y = _synthetic()
    spec = get_spec(name)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    pipeline = spec.build(spec.suggest(study.ask()))
    pipeline.fit(X, y)
    assert (pipeline.predict(X) >= 0).all()
