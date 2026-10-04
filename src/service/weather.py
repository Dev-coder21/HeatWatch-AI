"""Live weather for a profile's grid cell: last N days + today + next N days."""

import pandas as pd

from src.config import settings
from src.openmeteo import UpstreamError, get_json

DAILY_VARS = {
    "temperature_2m_max": "temperature_max",
    "temperature_2m_min": "temperature_min",
    "relative_humidity_2m_mean": "relative_humidity",
    "wind_speed_10m_max": "wind_speed",
    "precipitation_sum": "precipitation",
}


def forecast_params(profile):
    cfg = settings()["api"]
    return {
        "latitude": profile["lat"],
        "longitude": profile["lon"],
        # Same elevation as the profile and the normals (Open-Meteo lapse-corrects to it).
        "elevation": profile["elevation_m"],
        "past_days": cfg["past_days"],
        # +1 because the forecast window includes today.
        "forecast_days": cfg["forecast_days"] + 1,
        "daily": ",".join(DAILY_VARS),
        "timezone": cfg["timezone"],
    }


def fetch_weather(profile):
    data = get_json(settings()["api"]["forecast_url"], forecast_params(profile))
    daily = data["daily"]
    frame = pd.DataFrame({"date": daily["time"]})
    for source, name in DAILY_VARS.items():
        frame[name] = pd.to_numeric(pd.Series(daily[source]), errors="coerce")
    return frame.to_dict("records")


def get_weather(profile, store):
    """Return (records, stale, age_seconds). Serves stale cache if Open-Meteo fails."""
    key = profile["grid_key"]
    ttl = settings()["cache"]["weather_ttl_s"]
    cached, age = store.get("weather", key, ttl=ttl)
    if cached is not None:
        return cached, False, age
    try:
        records = fetch_weather(profile)
    except UpstreamError:
        stale, age = store.get_stale("weather", key)
        if stale is None:
            raise
        return stale, True, age
    store.set("weather", key, records)
    return records, False, 0.0
