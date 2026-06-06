"""SHARED feature-transformation module.

This module is imported by BOTH training and serving. It is the single place
where raw inputs become model features, which is what prevents **train/serve
skew**: if training derived ``time_of_day`` one way and the API derived it
another, the model would see different feature distributions at inference time
than it trained on, and predictions would silently degrade. By funneling both
paths through ``build_features`` (and the helpers below), the consistency test
in tests/test_train_serve_consistency.py can assert they are identical.

LEAKAGE (critical):
    The target is ``trip_count`` -- how many trips occurred in a zone-hour. Some
    gold columns only EXIST because those trips already happened (revenue, tips,
    distances, etc.). Feeding them to the model is target leakage: they would not
    be knowable at prediction time for a FUTURE hour. We restrict features to
    pre-knowable inputs only -- zone id, the target timestamp's calendar context,
    and (forecasted) weather -- and drop the leaky columns explicitly.

    The batch-inference path constructs rows from pre-knowable inputs ALONE, which
    is the practical proof that the feature design is leakage-free.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

TARGET = "trip_count"

# Columns that exist in the gold table ONLY because the trips already happened.
# Never features. Documented in README + docs/leakage.md.
LEAKY_COLUMNS: list[str] = [
    "total_revenue",
    "avg_tip",
    "avg_fare",
    "avg_distance",
    "avg_duration_min",
    "unique_destinations",
]

# The complete pre-knowable feature set the model is trained and served on.
# Anything here must be knowable BEFORE the target hour occurs.
PRE_KNOWABLE_FEATURES: list[str] = [
    "pickup_location_id",
    "pickup_hour",
    "day_of_week",
    "is_weekend",
    "time_of_day",  # categorical bucket, encoded downstream
    "temp_avg_fahrenheit",  # forecasted at serve/batch time
    "is_rainy",  # forecasted at serve/batch time
]

# Buckets for time_of_day. These boundaries MUST match how the gold table was
# produced upstream. If the data-pipeline definition changes, change it HERE in
# one place and the consistency test will flag any drift.
# Convention (inclusive lower, exclusive upper), 24h clock:
#   night:     00:00-05:59
#   morning:   06:00-11:59
#   afternoon: 12:00-16:59
#   evening:   17:00-20:59
#   night:     21:00-23:59  (wraps to night)
_TIME_OF_DAY_BOUNDS = [
    (6, 12, "morning"),
    (12, 17, "afternoon"),
    (17, 21, "evening"),
]


def time_of_day_bucket(hour: int) -> str:
    """Map an hour (0-23) to its time-of-day bucket.

    Single source of truth for the bucket boundaries used everywhere.
    """
    if not 0 <= int(hour) <= 23:
        raise ValueError(f"hour must be in 0..23, got {hour}")
    h = int(hour)
    for lo, hi, label in _TIME_OF_DAY_BOUNDS:
        if lo <= h < hi:
            return label
    return "night"  # 21-23 and 0-5


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from ``pickup_date`` + ``pickup_hour``.

    Used by the serving/batch path, where only a date + hour + zone are given and
    the calendar context must be reconstructed exactly as the gold table defines
    it. ``pickup_date`` may be a date, datetime, or ISO string.
    """
    out = df.copy()
    dates = pd.to_datetime(out["pickup_date"])
    # Monday=0 ... Sunday=6, matching the gold table's day_of_week convention.
    out["day_of_week"] = dates.dt.dayofweek.astype("int64")
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    out["time_of_day"] = out["pickup_hour"].map(time_of_day_bucket)
    return out


def build_features(
    df: pd.DataFrame,
    *,
    derive_calendar: bool = False,
) -> pd.DataFrame:
    """Produce the model-ready, leakage-free feature frame.

    Parameters
    ----------
    df:
        Input rows. For TRAINING this is the gold ``location_hourly_features``
        joined with daily weather (calendar cols already present, so
        ``derive_calendar=False``). For SERVING/BATCH the calendar cols are
        reconstructed from date+hour (``derive_calendar=True``), proving only
        pre-knowable inputs are used.
    derive_calendar:
        When True, (re)derive day_of_week / is_weekend / time_of_day from
        pickup_date + pickup_hour rather than trusting incoming columns.

    Returns
    -------
    A DataFrame containing exactly ``PRE_KNOWABLE_FEATURES`` (order preserved).
    Leaky columns are never present in the output regardless of input.
    """
    work = df.copy()

    if derive_calendar:
        work = compute_calendar_features(work)

    # Defensive: drop any leaky column that slipped in, so neither path can ever
    # train or serve on post-hoc information.
    work = work.drop(columns=[c for c in LEAKY_COLUMNS if c in work.columns])

    missing = [c for c in PRE_KNOWABLE_FEATURES if c not in work.columns]
    if missing:
        raise KeyError(
            f"Missing required pre-knowable feature columns: {missing}. "
            "Did the weather join run, and are calendar features derived?"
        )

    # Normalize dtypes the models expect.
    work["is_weekend"] = work["is_weekend"].astype(bool)
    work["time_of_day"] = work["time_of_day"].astype("category")

    return work[PRE_KNOWABLE_FEATURES].reset_index(drop=True)
