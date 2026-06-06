"""Train/serve consistency -- the critical anti-skew test.

Asserts that the SERVING path (derive calendar features from date+hour+zone) and
the TRAINING path (calendar features already in the gold row) produce IDENTICAL
feature rows for the same logical input. If this ever fails, the API would feed
the model a different distribution than it trained on.
"""

from __future__ import annotations

import pandas as pd

from nyc_taxi_demand.features.transform import build_features


def test_serving_reconstructs_training_features(gold_hourly_frame):
    # TRAINING path: calendar columns already present in the gold frame.
    train_feats = build_features(gold_hourly_frame, derive_calendar=False)

    # SERVING path: same rows but calendar features stripped, then re-derived
    # from only pre-knowable inputs (date + hour + zone + weather).
    serve_input = gold_hourly_frame.drop(columns=["day_of_week", "is_weekend", "time_of_day"])
    serve_feats = build_features(serve_input, derive_calendar=True)

    # Compare on a normalized basis (string dtypes) to ignore categorical vs object.
    pd.testing.assert_frame_equal(
        train_feats.astype(str).reset_index(drop=True),
        serve_feats.astype(str).reset_index(drop=True),
    )


def test_single_row_serving_matches_manual_expectation():
    raw = pd.DataFrame(
        [
            {
                "pickup_location_id": 132,
                "pickup_date": "2024-01-08",  # Monday
                "pickup_hour": 8,  # morning
                "temp_avg_fahrenheit": 45.0,
                "is_rainy": False,
            }
        ]
    )
    feats = build_features(raw, derive_calendar=True)
    row = feats.iloc[0]
    assert row["day_of_week"] == 0
    assert not row["is_weekend"]
    assert row["time_of_day"] == "morning"
