"""Tests for the shared feature module: leakage rules + bucket boundaries."""

from __future__ import annotations

import pandas as pd
import pytest

from nyc_taxi_demand.features.transform import (
    LEAKY_COLUMNS,
    PRE_KNOWABLE_FEATURES,
    build_features,
    compute_calendar_features,
    time_of_day_bucket,
)


@pytest.mark.parametrize(
    "hour,expected",
    [
        (0, "night"),
        (5, "night"),
        (6, "morning"),
        (11, "morning"),
        (12, "afternoon"),
        (16, "afternoon"),
        (17, "evening"),
        (20, "evening"),
        (21, "night"),
        (23, "night"),
    ],
)
def test_time_of_day_boundaries(hour, expected):
    assert time_of_day_bucket(hour) == expected


def test_time_of_day_rejects_bad_hour():
    with pytest.raises(ValueError):
        time_of_day_bucket(24)


def test_build_features_drops_leaky_columns(gold_hourly_frame):
    out = build_features(gold_hourly_frame, derive_calendar=False)
    for col in LEAKY_COLUMNS:
        assert col not in out.columns
    assert list(out.columns) == PRE_KNOWABLE_FEATURES


def test_build_features_requires_columns():
    with pytest.raises(KeyError):
        build_features(pd.DataFrame({"pickup_location_id": [1]}), derive_calendar=False)


def test_calendar_derivation_matches_known_dates():
    df = pd.DataFrame({"pickup_date": ["2024-01-06", "2024-01-08"], "pickup_hour": [8, 14]})
    out = compute_calendar_features(df)
    # 2024-01-06 is a Saturday (weekend), 2024-01-08 is a Monday (weekday).
    assert out.loc[0, "is_weekend"]
    assert not out.loc[1, "is_weekend"]
    assert out.loc[0, "time_of_day"] == "morning"
    assert out.loc[1, "time_of_day"] == "afternoon"
