"""Load gold tables from the shared data lake and assemble the training frame.

Reads are done through awswrangler against the Glue catalog (Athena), so we get
partition pruning for free and never hard-code S3 paths. The hourly features are
joined to daily weather by ``pickup_date``.

The weather table is daily; the demand table is hourly. We join weather context
DOWN onto each hour by date. At serve/batch time the same weather columns come
from a FORECAST for the target date (see inference module), which is why they are
legitimately pre-knowable.
"""

from __future__ import annotations

import awswrangler as wr
import pandas as pd

from nyc_taxi_demand.common.config import Settings, get_settings

# Weather columns we carry from the daily table onto the hourly rows.
_WEATHER_COLS = ["pickup_date", "temp_avg_fahrenheit", "is_rainy"]


def load_weather_daily(settings: Settings | None = None) -> pd.DataFrame:
    """Load the daily taxi+weather table (~600 rows)."""
    settings = settings or get_settings()
    df = wr.athena.read_sql_query(
        f"SELECT * FROM {settings.table_trip_weather_daily}",
        database=settings.glue_gold_database,
        ctas_approach=False,
    )
    df["pickup_date"] = pd.to_datetime(df["pickup_date"]).dt.date
    return df


def load_location_hourly(
    settings: Settings | None = None,
    *,
    year: int | None = None,
) -> pd.DataFrame:
    """Load the hourly per-location features (~145k rows, 20 monthly partitions).

    Optionally filter to a single ``year`` partition to keep local runs light.
    """
    settings = settings or get_settings()
    where = f"WHERE year = {year}" if year is not None else ""
    df = wr.athena.read_sql_query(
        f"SELECT * FROM {settings.table_location_hourly} {where}",
        database=settings.glue_gold_database,
        ctas_approach=False,
    )
    df["pickup_date"] = pd.to_datetime(df["pickup_date"]).dt.date
    return df


def load_training_frame(
    settings: Settings | None = None,
    *,
    year: int | None = None,
) -> pd.DataFrame:
    """Assemble the joined frame used for training.

    Returns the hourly rows with daily weather columns merged in by date. Leaky
    columns are still present here (they live in the gold table); they are dropped
    downstream by ``features.build_features`` -- this function's job is only to
    assemble, not to enforce the feature contract.
    """
    settings = settings or get_settings()
    hourly = load_location_hourly(settings, year=year)
    weather = load_weather_daily(settings)[_WEATHER_COLS]

    merged = hourly.merge(weather, on="pickup_date", how="left")
    # If a date has no weather row, fall back to sensible neutral values so the
    # join never silently drops demand rows.
    merged["temp_avg_fahrenheit"] = merged["temp_avg_fahrenheit"].fillna(
        merged["temp_avg_fahrenheit"].median()
    )
    merged["is_rainy"] = merged["is_rainy"].fillna(False).astype(bool)
    return merged
