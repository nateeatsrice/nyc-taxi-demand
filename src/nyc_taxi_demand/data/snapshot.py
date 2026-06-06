"""Reproducibility: describe exactly which data a training run consumed.

Every run logs a snapshot descriptor as MLflow params so any model's training
data is reconstructable: the source table, the partitions read, the row count,
and a stable hash of the data's identifying content. If two runs share a hash,
they trained on the same data.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import pandas as pd

from nyc_taxi_demand.common.config import Settings, get_settings


@dataclass(frozen=True)
class SnapshotDescriptor:
    source_table: str
    glue_database: str
    s3_uri: str
    row_count: int
    partitions: str  # comma-joined "year=YYYY/month=MM" list
    date_min: str
    date_max: str
    data_hash: str  # sha256 over sorted (date,hour,zone) keys + rowcount

    def as_mlflow_params(self) -> dict[str, str]:
        return {f"data.{k}": str(v) for k, v in asdict(self).items()}


def _hash_frame(df: pd.DataFrame) -> str:
    """Stable content hash over the identifying keys of the frame.

    Uses the natural key (date, hour, zone) rather than full content so the hash
    is robust to column reordering but still uniquely pins the row set.
    """
    keys = ["pickup_date", "pickup_hour", "pickup_location_id"]
    present = [k for k in keys if k in df.columns]
    keyed = df[present].astype(str).agg("|".join, axis=1).sort_values()
    h = hashlib.sha256()
    for v in keyed:
        h.update(v.encode("utf-8"))
    h.update(str(len(df)).encode("utf-8"))
    return h.hexdigest()[:16]


def describe_snapshot(
    df: pd.DataFrame,
    *,
    source_table: str,
    settings: Settings | None = None,
) -> SnapshotDescriptor:
    settings = settings or get_settings()
    dates = pd.to_datetime(df["pickup_date"])
    if {"year", "month"}.issubset(df.columns):
        parts = (
            df[["year", "month"]]
            .drop_duplicates()
            .sort_values(["year", "month"])
            .apply(lambda r: f"year={int(r.year)}/month={int(r.month):02d}", axis=1)
            .tolist()
        )
    else:
        parts = sorted(dates.dt.strftime("%Y-%m").unique().tolist())

    return SnapshotDescriptor(
        source_table=source_table,
        glue_database=settings.glue_gold_database,
        s3_uri=settings.s3_uri(settings.gold_features_prefix, source_table),
        row_count=int(len(df)),
        partitions=",".join(parts),
        date_min=str(dates.min().date()),
        date_max=str(dates.max().date()),
        data_hash=_hash_frame(df),
    )
