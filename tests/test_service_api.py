"""Service and API tests with Open-Meteo mocked (no network)."""

import datetime as dt

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.locations.profile as profile_module
from src.advisory import AUDIENCES, generate_advisory
from src.config import project_path, thresholds
from src.locations.profile import Cache
from src.openmeteo import UpstreamError
from src.risk_score import calculate_risk_score, risk_category

MODELS_PRESENT = (project_path("models") / "temperature_model.joblib").exists()
needs_models = pytest.mark.skipif(not MODELS_PRESENT, reason="trained models not present")

TODAY = dt.date(2026, 5, 20)
ELEVATION = 224.0


def weather_records(tmax_today=44.0, hot=True):
    """7 past days + today + 7 forecast days."""
    days = [TODAY + dt.timedelta(days=i) for i in range(-7, 8)]
    base = 43.0 if hot else 33.0
    return [
        {
            "date": d.isoformat(),
            "temperature_max": base + (i % 3) * 0.5 if d != TODAY else tmax_today,
            "temperature_min": base - 13,
            "relative_humidity": 25.0,
            "wind_speed": 12.0,
            "precipitation": 0.0,
        }
        for i, d in enumerate(days)
    ]


@pytest.fixture
def mocked(tmp_path, monkeypatch):
    """Isolated cache + mocked elevation, archive and forecast calls; records what was requested."""
    seen = {"elevation": [], "archive": [], "forecast": []}
    store = Cache(tmp_path / "cache.sqlite")
    monkeypatch.setattr(profile_module, "_cache", store)

    def fake_elevation(lat, lon):
        seen["elevation"].append((lat, lon))
        return ELEVATION

    def fake_archive(lat, lon, elevation, start, end):
        seen["archive"].append(elevation)
        days = pd.date_range(start, end)
        return pd.Series(38.0, index=days)

    state = {"records": weather_records(), "fail": False}

    def fake_fetch_weather(profile):
        from src.service.weather import forecast_params

        if state["fail"]:
            raise UpstreamError("Open-Meteo down")
        seen["forecast"].append(forecast_params(profile))
        return state["records"]

    monkeypatch.setattr(profile_module, "fetch_elevation", fake_elevation)
    monkeypatch.setattr("src.locations.normals.archive_tmax", fake_archive)
    monkeypatch.setattr(
        "src.locations.profile.compute_normals",
        lambda lat, lon, elevation, today: __import__("src.locations.normals", fromlist=["x"]).compute_normals(
            lat, lon, elevation, today=today, fetch=fake_archive
        ),
    )
    monkeypatch.setattr("src.service.weather.fetch_weather", fake_fetch_weather)
    return {"store": store, "seen": seen, "state": state}


# -----------------------------------
# Pure units
# -----------------------------------

def test_risk_score_bounds():
    cfg = thresholds()
    low = calculate_risk_score(10, 0, -10, 0, 0, "plains", cfg)
    high = calculate_risk_score(60, 1, 20, 10, 100, "plains", cfg)
    assert low["risk_score"] == 0
    assert high["risk_score"] == 100
    for value in high.values():
        assert 0 <= value <= 100


@pytest.mark.parametrize(
    "score, expected",
    [(0, "Low"), (19.5, "Low"), (20, "Moderate"), (39.99, "Moderate"), (59.5, "High"), (79.9, "Very High"), (100, "Extreme")],
)
def test_risk_category_has_no_gaps(score, expected):
    assert risk_category(score, thresholds()) == expected


def test_advisory_audiences_and_escalation():
    low = generate_advisory("Low", "No Heatwave", 30.0, 0.0)
    assert low["level"] == "low"
    assert set(low["audiences"]) == set(AUDIENCES)
    # A heatwave never gets a mild advisory, a severe one is always extreme.
    assert generate_advisory("Moderate", "Heatwave", 41.0, 0.6)["level"] == "very_high"
    assert generate_advisory("Low", "Severe Heatwave", 46.0, 0.9)["level"] == "extreme"
    assert any("46.0" in c for c in generate_advisory("Extreme", "Severe Heatwave", 46.0, 0.9)["context"])


def test_no_feature_list_contains_latitude_or_longitude():
    import re

    for script in ["src/train_temperature.py", "src/train_heatwave.py"]:
        text = (project_path(script)).read_text()
        block = re.search(r"FEATURE_COLUMNS = \[(.*?)\]", text, re.S).group(1)
        assert "latitude" not in block and "longitude" not in block
    if MODELS_PRESENT:
        import joblib

        for name in ["temperature_features.joblib", "classification_features.joblib"]:
            features = joblib.load(project_path("models") / name)
            assert "latitude" not in features and "longitude" not in features


# -----------------------------------
# Service (mocked network)
# -----------------------------------

def test_same_elevation_everywhere(mocked):
    """Profile, normals archive calls and the forecast call all use the DEM elevation."""
    from src.service.weather import get_weather
    from src.locations.profile import get_normals, get_profile

    profile = get_profile(28.61, 77.21)
    get_weather(profile, mocked["store"])
    get_normals(profile, today=TODAY)
    assert profile["elevation_m"] == ELEVATION
    assert mocked["seen"]["archive"] and set(mocked["seen"]["archive"]) == {ELEVATION}
    assert mocked["seen"]["forecast"][0]["elevation"] == ELEVATION
    # Weather is requested at the cell centre the profile describes.
    assert (mocked["seen"]["forecast"][0]["latitude"], mocked["seen"]["forecast"][0]["longitude"]) == (28.6, 77.2)


@needs_models
def test_assess_shape_and_imd_labels(mocked):
    from src.service.assess import assess

    result = assess(28.61, 77.21)
    p = result["prediction"]
    assert result["issued_date"] == TODAY.isoformat()
    assert result["forecast_date"] == (TODAY + dt.timedelta(days=1)).isoformat()
    assert result["location"]["terrain_type"] == "plains"
    assert len(result["outlook"]) == 7 and len(result["recent_days"]) == 8
    assert 0 <= p["risk_score"] <= 100 and 0 <= p["heatwave_probability"] <= 1
    # Normals are 38 C, observed days 43-44 C: departure >= 4.5 at >= 40 C -> heatwave by rule.
    assert result["recent_days"][-1]["imd_rule_severity"] != "No Heatwave"
    # Forecast days reach 44 C, normal 38 C (departure 6) -> heatwave at least.
    assert all(day["imd_rule_severity"] != "No Heatwave" for day in result["outlook"])
    # The model regresses towards persistence, so tomorrow may sit just under the rule.
    assert result["advisory"]["level"] in {"high", "very_high", "extreme"}


@needs_models
def test_assess_serves_stale_weather_when_upstream_fails(mocked):
    from src.service.assess import assess

    assess(28.61, 77.21)
    # Expire the weather cache and break upstream.
    mocked["store"].conn.execute("UPDATE kv SET ts = 0 WHERE ns = 'weather'")
    mocked["store"].conn.commit()
    mocked["state"]["fail"] = True
    result = assess(28.61, 77.21)
    assert result["data"]["stale"] is True


@needs_models
def test_explain_returns_shap_for_both_models(mocked):
    from src.service.assess import explain

    result = explain(28.61, 77.21)
    for key in ["predicted_tmax", "heatwave_probability"]:
        block = result[key]
        assert len(block["contributions"]) == 14
        total = block["base_value"] + sum(c["shap"] for c in block["contributions"])
        assert total == pytest.approx(block["prediction"], abs=1e-3)


# -----------------------------------
# API
# -----------------------------------

@pytest.fixture
def client(mocked):
    from backend.main import app

    return TestClient(app)


def test_api_outside_india_is_400(client):
    for path in ["/api/risk", "/api/explain", "/api/reverse"]:
        response = client.get(path, params={"lat": 51.5, "lon": -0.12})
        assert response.status_code == 400
        assert "outside India" in response.json()["detail"]


def test_api_invalid_coordinates_422(client):
    assert client.get("/api/risk", params={"lat": 200, "lon": 77}).status_code == 422
    assert client.get("/api/search", params={"q": "a"}).status_code == 422


def test_api_reverse(client):
    response = client.get("/api/reverse", params={"lat": 26.27, "lon": 73.01})
    assert response.status_code == 200
    assert response.json()["name"] == "Jodhpur"


def test_api_search_upstream_down_is_502(client, monkeypatch):
    def boom(*args, **kwargs):
        raise UpstreamError("down")

    monkeypatch.setattr("backend.main.geocode", boom)
    assert client.get("/api/search", params={"q": "Delhi"}).status_code == 502


def test_api_risk_upstream_down_no_cache_is_502(client, mocked):
    mocked["state"]["fail"] = True
    assert client.get("/api/risk", params={"lat": 28.61, "lon": 77.21}).status_code == 502


@needs_models
def test_api_risk_and_explain(client):
    response = client.get("/api/risk", params={"lat": 28.61, "lon": 77.21})
    assert response.status_code == 200
    body = response.json()
    assert body["location"]["grid_key"] == "28.60,77.20"
    assert body["data"]["stale"] is False
    response = client.get("/api/explain", params={"lat": 28.61, "lon": 77.21})
    assert response.status_code == 200


@needs_models
def test_api_featured(client):
    response = client.get("/api/featured")
    assert response.status_code == 200
    places = response.json()["places"]
    assert len(places) == len(pd.read_csv(project_path("config/locations.csv")))
    assert all(p["risk_score"] is not None for p in places)
