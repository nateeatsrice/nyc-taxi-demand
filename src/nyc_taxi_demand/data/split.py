"""Temporal train/validation split.

Demand forecasting must be evaluated the way it is used: train on the past,
validate on the future. A random shuffle split would leak future information into
training (rows from the same future days would appear in both sets) and produce
optimistic, meaningless metrics. We split strictly by date.
"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.2,
    date_col: str = "pickup_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, val) where every val date is strictly later than every
    train date.

    The cut point is chosen so that approximately ``val_fraction`` of ROWS fall in
    validation, but the boundary is placed on a date edge so no single day spans
    both sets.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    ordered = df.sort_values(date_col).reset_index(drop=True)
    unique_dates = pd.Series(sorted(ordered[date_col].unique()))

    # Walk dates from the end, accumulating rows until we reach val_fraction.
    counts = ordered.groupby(date_col).size()
    target_val_rows = int(len(ordered) * val_fraction)

    accumulated = 0
    cut_date = unique_dates.iloc[-1]
    for d in reversed(unique_dates.tolist()):
        accumulated += int(counts.loc[d])
        cut_date = d
        if accumulated >= target_val_rows:
            break

    train = ordered[ordered[date_col] < cut_date].reset_index(drop=True)
    val = ordered[ordered[date_col] >= cut_date].reset_index(drop=True)

    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            "Temporal split produced an empty side; check val_fraction and that "
            "the data spans multiple dates."
        )
    return train, val
