"""Shared pytest fixtures: small synthetic frames shaped like the gold tables."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_demand.features.transform import time_of_day_bucket


@pytest.fixture
def gold_hourly_frame() -> pd.DataFrame:
    """A small frame shaped like location_hourly_features joined with weather,
    spanning several dates so temporal split has something to cut.
    """
    rng = np.random.default_rng(0)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=d) for d in range(20)]
    rows = []
    for d in dates:
        for hour in range(0, 24, 6):
            for zone in (100, 132, 161):
                rows.append(
                    {
                        "pickup_date": d,
                        "pickup_hour": hour,
                        "pickup_location_id": zone,
                        "trip_count": int(rng.integers(1, 200)),
                        # leaky columns (must be dropped by build_features)
                        "total_revenue": float(rng.uniform(100, 5000)),
                        "avg_tip": float(rng.uniform(0, 10)),
                        "avg_fare": float(rng.uniform(5, 60)),
                        "avg_distance": float(rng.uniform(0.5, 20)),
                        "avg_duration_min": float(rng.uniform(3, 45)),
                        "unique_destinations": int(rng.integers(1, 50)),
                        # calendar (present in gold)
                        "day_of_week": d.weekday(),
                        "is_weekend": d.weekday() >= 5,
                        "time_of_day": time_of_day_bucket(hour),
                        # weather (joined)
                        "temp_avg_fahrenheit": float(rng.uniform(20, 90)),
                        "is_rainy": bool(rng.integers(0, 2)),
                        # partitions
                        "year": d.year,
                        "month": d.month,
                    }
                )
    return pd.DataFrame(rows)
