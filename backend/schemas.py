"""Pydantic response models: the API contract documented in docs/API.md."""

from typing import Literal, Optional

from pydantic import BaseModel

Severity = Literal["No Heatwave", "Heatwave", "Severe Heatwave"]
Terrain = Literal["plains", "coastal", "hilly"]


class ErrorResponse(BaseModel):
    detail: str


class Place(BaseModel):
    name: str
    district: Optional[str] = None
    state: Optional[str] = None
    lat: float
    lon: float
    elevation: Optional[float] = None
    population: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    results: list[Place]


class ReverseResponse(BaseModel):
    name: str
    label: str
    state: Optional[str] = None
    distance_km: float
    lat: float
    lon: float
    in_india: bool


class Location(BaseModel):
    name: str
    label: str
    state: Optional[str] = None
    query_lat: float
    query_lon: float
    grid_lat: float
    grid_lon: float
    grid_key: str
    elevation_m: float
    terrain_type: Terrain
    distance_to_coast_km: float


class RiskComponents(BaseModel):
    temperature: float
    heatwave_probability: float
    anomaly: float
    persistence: float
    humidity: float


class Prediction(BaseModel):
    predicted_tmax: float
    raw_forecast_tmax: Optional[float] = None
    normal_tmax: float
    archive_normal_tmax: float
    normal_bias_correction_c: float
    departure: float
    heatwave_probability: float
    model_severity: Severity
    imd_rule_severity: Severity
    imd_rule: Literal["none", "departure", "absolute"]
    severity: Severity
    consecutive_heatwave_days: int
    humidity: Optional[float] = None
    risk_score: float
    risk_category: Literal["Low", "Moderate", "High", "Very High", "Extreme"]
    risk_components: RiskComponents
    top_factors: list[str]


class Audiences(BaseModel):
    outdoor_workers: str
    elderly: str
    children: str


class Advisory(BaseModel):
    level: Literal["low", "moderate", "high", "very_high", "extreme"]
    general: str
    audiences: Audiences
    context: list[str]


class Today(BaseModel):
    date: str
    tmax: Optional[float] = None
    normal_tmax: Optional[float] = None
    departure: Optional[float] = None
    consecutive_heatwave_days: int


class RecentDay(BaseModel):
    date: str
    tmax: Optional[float] = None
    tmin: Optional[float] = None
    normal_tmax: Optional[float] = None
    departure: Optional[float] = None
    imd_rule_severity: Severity


class OutlookDay(BaseModel):
    date: str
    forecast_tmax: Optional[float] = None
    forecast_tmin: Optional[float] = None
    normal_tmax: Optional[float] = None
    departure: Optional[float] = None
    humidity: Optional[float] = None
    precipitation_mm: Optional[float] = None
    imd_rule_severity: Severity
    confidence: Literal["model", "forecast", "low"]
    alert_label: Optional[
        Literal["Heatwave", "Severe Heatwave", "Possible heatwave", "Possible severe heatwave"]
    ] = None


class DataInfo(BaseModel):
    stale: bool
    weather_age_s: int
    normal_bias_correction_c: float
    normal_bias_overlap_days: int
    source: str
    grid_resolution: str


class RiskResponse(BaseModel):
    location: Location
    issued_date: str
    forecast_date: str
    prediction: Prediction
    advisory: Advisory
    today: Today
    recent_days: list[RecentDay]
    outlook: list[OutlookDay]
    data: DataInfo


class Contribution(BaseModel):
    feature: str
    label: str
    value: Optional[float] = None
    shap: float


class Explanation(BaseModel):
    base_value: float
    prediction: float
    contributions: list[Contribution]


class ExplainLocation(BaseModel):
    name: str
    grid_key: str


class ExplainResponse(BaseModel):
    location: ExplainLocation
    issued_date: str
    forecast_date: str
    stale: bool
    predicted_tmax: Explanation
    heatwave_probability: Explanation


class FeaturedPlace(BaseModel):
    slug: str
    name: str
    state: str
    lat: float
    lon: float
    terrain_type: Optional[Terrain] = None
    forecast_date: Optional[str] = None
    predicted_tmax: Optional[float] = None
    heatwave_probability: Optional[float] = None
    severity: Optional[Severity] = None
    risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    stale: Optional[bool] = None
    error: Optional[str] = None


class FeaturedResponse(BaseModel):
    places: list[FeaturedPlace]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    model_error: Optional[str] = None
    model_features: list[str]
    upstream_reachable: bool
    cache: dict[str, int]
