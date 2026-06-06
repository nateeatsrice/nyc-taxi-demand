"""Tests for temporal split, snapshot descriptor, and the weather-join loader
(with awswrangler mocked)."""

from __future__ import annotations

from unittest.mock import patch

from nyc_taxi_demand.data.snapshot import describe_snapshot
from nyc_taxi_demand.data.split import temporal_split


def test_temporal_split_is_chronological(gold_hourly_frame):
    train, val = temporal_split(gold_hourly_frame, val_fraction=0.25)
    assert train["pickup_date"].max() < val["pickup_date"].min()
    assert len(train) > 0 and len(val) > 0


def test_temporal_split_no_shared_dates(gold_hourly_frame):
    train, val = temporal_split(gold_hourly_frame, val_fraction=0.3)
    assert set(train["pickup_date"]).isdisjoint(set(val["pickup_date"]))


def test_snapshot_hash_is_stable(gold_hourly_frame):
    s1 = describe_snapshot(gold_hourly_frame, source_table="location_hourly_features")
    s2 = describe_snapshot(
        gold_hourly_frame.sample(frac=1, random_state=1),  # reorder rows
        source_table="location_hourly_features",
    )
    # Hash is over the sorted natural key, so row order must not change it.
    assert s1.data_hash == s2.data_hash
    assert s1.row_count == len(gold_hourly_frame)


def test_loader_joins_weather(gold_hourly_frame):
    hourly = gold_hourly_frame.drop(columns=["temp_avg_fahrenheit", "is_rainy"])
    weather = (
        gold_hourly_frame[["pickup_date", "temp_avg_fahrenheit", "is_rainy"]]
        .drop_duplicates("pickup_date")
        .reset_index(drop=True)
    )

    with (
        patch("nyc_taxi_demand.data.loader.load_location_hourly", return_value=hourly),
        patch("nyc_taxi_demand.data.loader.load_weather_daily", return_value=weather),
    ):
        from nyc_taxi_demand.data.loader import load_training_frame

        merged = load_training_frame()

    assert "temp_avg_fahrenheit" in merged.columns
    assert "is_rainy" in merged.columns
    assert len(merged) == len(hourly)
    assert not merged["temp_avg_fahrenheit"].isna().any()
