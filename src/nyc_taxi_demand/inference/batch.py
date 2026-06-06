"""Batch inference: predict demand for all zones across a future time window.

This path is the practical proof that the feature design is leakage-free: it
constructs prediction rows from PRE-KNOWABLE INPUTS ALONE -- zone, target
date/hour calendar context, and a forecasted-weather input -- with no access to
any post-hoc gold columns. The same shared feature module and the same Production
model used by the live API produce the predictions.

WEATHER (accepted design hole #5): there is no live forecast provider wired in.
The caller passes a weather assumption per date (or a flat default). In a real
deployment this would come from a forecast API; here it is an explicit input so
the leakage boundary stays clean and obvious.

Output: one JSON file per run under ``batch-predictions/nyc-taxi-demand/``.
"""

from __future__ import annotations

import datetime as dt
from itertools import product

import pandas as pd

from nyc_taxi_demand.common.config import Settings, get_settings
from nyc_taxi_demand.common.s3 import put_json
from nyc_taxi_demand.features.transform import build_features
from nyc_taxi_demand.registry.promote import load_production_model


def _date_range(start: dt.date, days: int) -> list[dt.date]:
    return [start + dt.timedelta(days=d) for d in range(days)]


def run_batch_inference(
    *,
    start_date: dt.date,
    days: int = 1,
    zones: list[int] | None = None,
    hours: list[int] | None = None,
    weather_by_date: dict[dt.date, dict] | None = None,
    default_temp_f: float = 60.0,
    default_rainy: bool = False,
    settings: Settings | None = None,
) -> str:
    """Predict demand for the cartesian product of zones x dates x hours.

    Returns the S3 URI of the written predictions file.
    """
    settings = settings or get_settings()
    model = load_production_model(settings)

    zones = zones or list(range(1, 266))  # NYC taxi zone ids 1..265
    hours = hours or list(range(24))
    weather_by_date = weather_by_date or {}

    rows = []
    for d, zone, hour in product(_date_range(start_date, days), zones, hours):
        w = weather_by_date.get(d, {})
        rows.append(
            {
                "pickup_location_id": zone,
                "pickup_date": d,
                "pickup_hour": hour,
                "temp_avg_fahrenheit": w.get("temp_avg_fahrenheit", default_temp_f),
                "is_rainy": w.get("is_rainy", default_rainy),
            }
        )

    raw = pd.DataFrame(rows)
    features = build_features(raw, derive_calendar=True)  # pre-knowable only
    raw["predicted_trip_count"] = [max(0.0, float(p)) for p in model.predict(features)]

    generated_at = dt.datetime.utcnow().isoformat()
    key = f"{settings.batch_predictions_prefix}/run_date={start_date.isoformat()}/predictions.json"
    payload = {
        "generated_at": generated_at,
        "start_date": start_date.isoformat(),
        "days": days,
        "row_count": len(raw),
        "predictions": raw.assign(pickup_date=raw["pickup_date"].astype(str)).to_dict(
            orient="records"
        ),
    }
    return put_json(key, payload)
