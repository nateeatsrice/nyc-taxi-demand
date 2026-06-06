"""Streamlit demand-prediction UI.

Lets a user pick a zone, date, hour, and forecasted weather, then calls the
FastAPI ``/predict`` endpoint and shows the predicted trip count. The UI holds no
model itself -- it is a thin client over the API, so there is exactly one place
the Production model is loaded.

Run: ``streamlit run src/nyc_taxi_demand/serving/ui/app.py``
Set the API endpoint via env ``NTD_API_URL`` (defaults to localhost:8000).
"""

from __future__ import annotations

import datetime as dt
import os

import requests
import streamlit as st

API_URL = os.environ.get("NTD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="NYC Taxi Demand Forecast", page_icon="🚕")
st.title("🚕 NYC Taxi Demand Forecast")
st.caption(
    "Predict hourly trip demand for a zone using only pre-knowable inputs "
    "(zone, calendar, forecasted weather)."
)

col1, col2 = st.columns(2)
with col1:
    zone = st.number_input("Pickup zone id", min_value=1, max_value=265, value=132)
    target_date = st.date_input("Date", value=dt.date.today())
    hour = st.slider("Hour of day", 0, 23, 8)
with col2:
    temp = st.slider("Forecasted avg temp (°F)", -10, 110, 60)
    rainy = st.checkbox("Forecasted rain?", value=False)

if st.button("Predict demand", type="primary"):
    payload = {
        "pickup_location_id": int(zone),
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
