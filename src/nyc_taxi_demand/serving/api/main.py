"""FastAPI prediction service.

Loads the current Production model from the MLflow registry once at startup and
serves predictions. Features are built through the SHARED feature module with
``derive_calendar=True``, so the exact same transform logic used in training is
applied here -- this is the runtime half of the train/serve consistency guarantee.

NO AUTH. This is intentionally out of scope for the platform (see README "future
work"). Do not expose this publicly without adding an auth layer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from nyc_taxi_demand.features.transform import build_features
from nyc_taxi_demand.registry.promote import load_production_model
from nyc_taxi_demand.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

_state: dict[str, object] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup. If no Production model exists this raises and the
    # container fails fast -- preferable to serving a missing model.
    try:
        _state["model"] = load_production_model()
    except Exception as exc:  # noqa: BLE001
        # Keep the app up so /health can report the problem clearly.
        _state["model"] = None
        _state["load_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(title="nyc-taxi-demand prediction API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state.get("model") is not None else "degraded",
        model_loaded=_state.get("model") is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Production model not loaded: {_state.get('load_error', 'unknown')}",
        )

    raw = pd.DataFrame(
        [
            {
                "pickup_location_id": req.pickup_location_id,
                "pickup_date": req.pickup_date,
                "pickup_hour": req.pickup_hour,
                "temp_avg_fahrenheit": req.temp_avg_fahrenheit,
                "is_rainy": req.is_rainy,
            }
        ]
    )
    features = build_features(raw, derive_calendar=True)
    pred = float(model.predict(features)[0])

    return PredictionResponse(
        pickup_location_id=req.pickup_location_id,
        pickup_date=req.pickup_date,
        pickup_hour=req.pickup_hour,
        predicted_trip_count=max(0.0, pred),
    )
