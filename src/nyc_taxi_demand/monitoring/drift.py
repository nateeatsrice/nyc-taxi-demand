"""Evidently drift + performance reporting (deferrable phase).

ML-specific monitoring, separate from the CloudWatch infra/app layer. Compares a
"current" feature distribution (recent rows or a batch-inference feature set)
against the training baseline and writes an HTML report (for reading) plus a JSON
summary (for programmatic checks) to ``monitoring/nyc-taxi-demand/`` in S3.

Run as an on-demand / batch job, NOT an always-on service. Evidently lives in the
``monitoring`` optional-dependency group and is imported lazily so the core
laptop install never needs it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from nyc_taxi_demand.common.config import Settings, get_settings
from nyc_taxi_demand.common.s3 import put_json, put_text
from nyc_taxi_demand.features.transform import PRE_KNOWABLE_FEATURES


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    settings: Settings | None = None,
    label: str | None = None,
) -> dict[str, str]:
    """Generate a drift report comparing ``current`` to ``reference``.

    Both frames should contain the pre-knowable feature columns. Returns a dict
    of the written S3 URIs.
    """
    settings = settings or get_settings()
    # Lazy import: keeps Evidently out of the laptop-safe core install.
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    cols = [c for c in PRE_KNOWABLE_FEATURES if c in reference.columns and c in current.columns]
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])

    stamp = label or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base = f"{settings.monitoring_prefix}/drift/{stamp}"

    html_uri = put_text(f"{base}/report.html", report.get_html(), "text/html")
    json_uri = put_json(f"{base}/summary.json", report.as_dict())
    return {"html": html_uri, "json": json_uri}
