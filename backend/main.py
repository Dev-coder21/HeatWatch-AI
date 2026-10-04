"""HeatWatch AI API: live heat risk for any point in India.

Run: uvicorn backend.main:app --reload   (from the project root)
Contract: docs/API.md
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import (  # noqa: E402
    ErrorResponse,
    ExplainResponse,
    FeaturedResponse,
    HealthResponse,
    ReverseResponse,
    RiskResponse,
    SearchResponse,
)
from src.config import project_path, settings  # noqa: E402
from src.locations.geo import in_india  # noqa: E402
from src.locations.geocode import geocode, reverse  # noqa: E402
from src.locations.profile import cache  # noqa: E402
from src.openmeteo import UpstreamError, get_json  # noqa: E402
from src.service.assess import OutsideIndia, assess_cached, explain  # noqa: E402
from src.service.models import ModelsUnavailable, load_models  # noqa: E402


def _load_env():
    """Minimal .env reader (KEY=VALUE lines) so no extra dependency is needed."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

app = FastAPI(
    title="HeatWatch AI API",
    description="Live heat-risk assessment for any location in India.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)

ERRORS = {
    400: {"model": ErrorResponse, "description": "Invalid input or point outside India"},
    502: {"model": ErrorResponse, "description": "Open-Meteo unreachable and no cached data"},
    503: {"model": ErrorResponse, "description": "Models not available"},
}

Lat = Query(..., ge=-90, le=90, description="Latitude in decimal degrees")
Lon = Query(..., ge=-180, le=180, description="Longitude in decimal degrees")


def _run(func, *args):
    try:
        return func(*args)
    except OutsideIndia as error:
        raise HTTPException(status_code=400, detail=str(error))
    except UpstreamError as error:
        raise HTTPException(status_code=502, detail=f"Weather data unavailable: {error}")
    except ModelsUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error))


@app.get("/", include_in_schema=False)
def root():
    return {"name": "HeatWatch AI API", "version": app.version, "docs": "/docs"}


@app.get("/api/search", response_model=SearchResponse, responses=ERRORS)
def search(q: str = Query(..., min_length=2, max_length=100, description="Place name")):
    try:
        results = geocode(q)
    except UpstreamError as error:
        raise HTTPException(status_code=502, detail=f"Geocoding unavailable: {error}")
    return {"query": q, "results": results}


@app.get("/api/reverse", response_model=ReverseResponse, responses=ERRORS)
def reverse_lookup(lat: float = Lat, lon: float = Lon):
    if not in_india(lat, lon):
        raise HTTPException(status_code=400, detail=f"{lat:.3f}, {lon:.3f} is outside India")
    return {**reverse(lat, lon), "in_india": True}


@app.get("/api/risk", response_model=RiskResponse, responses=ERRORS)
def risk(lat: float = Lat, lon: float = Lon):
    return _run(assess_cached, lat, lon)


@app.get("/api/explain", response_model=ExplainResponse, responses=ERRORS)
def explain_point(lat: float = Lat, lon: float = Lon):
    return _run(explain, lat, lon)


def _featured_places():
    return pd.read_csv(project_path("config/locations.csv")).to_dict("records")


def _featured_one(place):
    base = {
        "slug": place["slug"],
        "name": place["name"],
        "state": place["state"],
        "lat": place["latitude"],
        "lon": place["longitude"],
    }
    try:
        result = assess_cached(place["latitude"], place["longitude"])
    except (UpstreamError, OutsideIndia, ModelsUnavailable) as error:
        return {**base, "error": str(error)}
    p = result["prediction"]
    return {
        **base,
        "terrain_type": result["location"]["terrain_type"],
        "forecast_date": result["forecast_date"],
        "predicted_tmax": p["predicted_tmax"],
        "heatwave_probability": p["heatwave_probability"],
        "severity": p["severity"],
        "risk_score": p["risk_score"],
        "risk_category": p["risk_category"],
        "stale": result["data"]["stale"],
    }


@app.get("/api/featured", response_model=FeaturedResponse, responses=ERRORS)
def featured():
    with ThreadPoolExecutor(max_workers=4) as pool:
        places = list(pool.map(_featured_one, _featured_places()))
    return {"places": places}


@app.get("/api/health", response_model=HealthResponse)
def health():
    try:
        models = load_models()
        models_loaded, model_error, features = True, None, list(models["temperature_features"])
    except ModelsUnavailable as error:
        models_loaded, model_error, features = False, str(error), []
    try:
        get_json(
            settings()["api"]["forecast_url"],
            {"latitude": 28.6, "longitude": 77.2, "daily": "temperature_2m_max", "forecast_days": 1},
            retries=1,
            timeout=5,
        )
        upstream = True
    except UpstreamError:
        upstream = False
    store = cache()
    return {
        "status": "ok" if models_loaded and upstream else "degraded",
        "models_loaded": models_loaded,
        "model_error": model_error,
        "model_features": features,
        "upstream_reachable": upstream,
        "cache": {ns: store.count(ns) for ns in ["profile", "normals", "weather", "risk", "explain"]},
    }
