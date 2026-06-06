"""Request/response schemas for the prediction API.

The request carries ONLY pre-knowable inputs: zone, date, hour, and forecasted
weather. Calendar features (day_of_week, is_weekend, time_of_day) are DERIVED
server-side via the shared feature module -- the client never supplies them,
which keeps the API leakage-safe by construction.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pickup_location_id: int = Field(..., ge=1, description="NYC taxi zone id")
    pickup_date: date = Field(..., description="Target date (calendar features derived from this)")
    pickup_hour: int = Field(..., ge=0, le=23, description="Target hour, 0-23")
    temp_avg_fahrenheit: float = Field(..., description="Forecasted avg temp (F)")
    is_rainy: bool = Field(False, description="Forecasted rain for the date")


class PredictionResponse(BaseModel):
    pickup_location_id: int
    pickup_date: date
    pickup_hour: int
    predicted_trip_count: float
    model_stage: str = "Production"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
