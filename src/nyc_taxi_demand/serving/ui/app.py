"""Streamlit demand-prediction UI.

Thin client over the FastAPI ``/predict`` endpoint. The UI holds no model.

Zone selection is bidirectional and stays consistent because there is ONE owner
per widget key, and each callback writes only the OTHER widgets' keys:
  * change Borough     -> reset Neighborhood to that borough's first zone,
                          then sync the id box.
  * change Neighborhood-> sync the id box.
  * change the id box  -> sync both dropdowns.

The key correctness rule (Streamlit): a widget with a ``key`` owns that slot in
session_state. We never pass ``value=``/``index=`` to these widgets; we seed their
keys in session_state ONCE before they render, and after that only callbacks mutate
them. Mixing ``value=`` with callback mutation is what caused the earlier desync.

Run: ``streamlit run src/nyc_taxi_demand/serving/ui/app.py``
Env ``NTD_API_URL`` overrides the API endpoint (default localhost:8000).
"""

from __future__ import annotations

import datetime as dt
import os

import requests
import streamlit as st

from nyc_taxi_demand.serving.ui.zones import (
    BOROUGH_INDEX,
    BOROUGHS,
    label_for_id,
)

API_URL = os.environ.get("NTD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="NYC Taxi Demand Forecast", page_icon="🚕")
st.title("🚕 NYC Taxi Demand Forecast")
st.caption(
    "Predict hourly trip demand for a zone using only pre-knowable inputs "
    "(zone, calendar, forecasted weather)."
)


# --- helpers ------------------------------------------------------------------
def _labels_for(borough: str) -> list[str]:
    return [label for label, _ in BOROUGH_INDEX.get(borough, [])]


def _id_for(borough: str, label: str) -> int | None:
    for lbl, lid in BOROUGH_INDEX.get(borough, []):
        if lbl == label:
            return lid
    return None


def _seed_state(default_id: int = 132) -> None:
    """Initialize the three widget keys ONCE, consistently, before they render."""
    if "id_input" in st.session_state:
        return
    borough, label, _ = label_for_id(default_id)
    st.session_state.borough_select = borough
    st.session_state.neighborhood_select = label
    st.session_state.id_input = default_id


# --- callbacks: each writes only the OTHER widgets' keys ----------------------
def _on_borough_change() -> None:
    borough = st.session_state.borough_select
    first_label = _labels_for(borough)[0]
    st.session_state.neighborhood_select = first_label
    lid = _id_for(borough, first_label)
    if lid is not None:
        st.session_state.id_input = lid


def _on_neighborhood_change() -> None:
    borough = st.session_state.borough_select
    label = st.session_state.neighborhood_select
    lid = _id_for(borough, label)
    if lid is not None:
        st.session_state.id_input = lid


def _on_id_change() -> None:
    lid = int(st.session_state.id_input)
    hit = label_for_id(lid)
    if hit is None:
        return  # out of range / unknown id: leave dropdowns as-is
    borough, label, _ = hit
    st.session_state.borough_select = borough
    st.session_state.neighborhood_select = label


_seed_state()

# --- zone selection -----------------------------------------------------------
st.subheader("Pickup zone")
zcol1, zcol2 = st.columns(2)
with zcol1:
    st.selectbox(
        "Borough",
        options=BOROUGHS,
        key="borough_select",
        on_change=_on_borough_change,
    )
with zcol2:
    # Options must match the currently selected borough. Because the borough
    # callback already reset neighborhood_select to a valid label, the key always
    # holds a value that exists in these options.
    st.selectbox(
        "Neighborhood",
        options=_labels_for(st.session_state.borough_select),
        key="neighborhood_select",
        on_change=_on_neighborhood_change,
    )

st.number_input(
    "Pickup zone id (1-265)",
    min_value=1,
    max_value=265,
    step=1,
    key="id_input",
    on_change=_on_id_change,
    help="Type an id to populate the dropdowns above, or use the dropdowns.",
)

_resolved = label_for_id(st.session_state.id_input)
if _resolved:
    st.caption(f"Selected: **{_resolved[0]} / {_resolved[2]}** (id {st.session_state.id_input})")
else:
    st.warning(f"Id {st.session_state.id_input} is not a known zone.")

# --- when & weather -----------------------------------------------------------
st.subheader("When & weather")
wcol1, wcol2, wcol3 = st.columns(3)
with wcol1:
    target_date = st.date_input("Date", value=dt.date.today())
with wcol2:
    hour = st.slider("Hour of day", 0, 23, 8)
with wcol3:
    temp = st.slider("Forecasted avg temp (°F)", -10, 110, 60)
rainy = st.checkbox("Forecasted rain?", value=False)

# --- predict ------------------------------------------------------------------
if st.button("Predict demand", type="primary"):
    zone = int(st.session_state.id_input)
    payload = {
        "pickup_location_id": zone,
        "pickup_date": target_date.isoformat(),
        "pickup_hour": int(hour),
        "temp_avg_fahrenheit": float(temp),
        "is_rainy": bool(rainy),
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        st.metric(
            label=f"Predicted trips — zone {zone}, {target_date} @ {hour:02d}:00",
            value=f"{data['predicted_trip_count']:.0f}",
        )
        st.caption(f"Served by model stage: {data.get('model_stage', 'Production')}")
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(f"Is the API running and reachable at {API_URL}?")
