"""Place search (Open-Meteo Geocoding, India only) and offline reverse lookup."""

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import project_path, settings
from src.locations.geo import haversine_km, in_india
from src.openmeteo import get_json

PLACES_FILE = "data/static/india_places.csv"
# Snap a map click to a named place only if one is this close.
REVERSE_MAX_KM = 15


def geocode(query, count=10):
    """Return India-only candidates {name, district, state, lat, lon, elevation, population}."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    data = get_json(
        settings()["api"]["geocoding_url"],
        {"name": query, "count": count, "countryCode": "IN", "language": "en", "format": "json"},
    )
    results = []
    for item in data.get("results", []) or []:
        if item.get("country_code") != "IN":
            continue
        results.append(
            {
                "name": item["name"],
                "district": item.get("admin2"),
                "state": item.get("admin1"),
                "lat": round(item["latitude"], 4),
                "lon": round(item["longitude"], 4),
                "elevation": item.get("elevation"),
                "population": item.get("population"),
            }
        )
    return results


@lru_cache(maxsize=1)
def _places():
    return pd.read_csv(project_path(PLACES_FILE))


def reverse(lat, lon):
    """Nearest named place (GeoNames, pop >= 5000) or a coordinate label."""
    places = _places()
    distances = haversine_km(lat, lon, places["lat"].values, places["lon"].values)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    coordinate_label = f"{lat:.2f}, {lon:.2f}"
    if distance <= REVERSE_MAX_KM:
        row = places.iloc[index]
        return {
            "name": row["name"],
            "state": row["state"] if isinstance(row["state"], str) else None,
            "label": f"{row['name']}, {row['state']}" if isinstance(row["state"], str) else row["name"],
            "distance_km": round(distance, 1),
            "lat": lat,
            "lon": lon,
        }
    row = places.iloc[index]
    return {
        "name": coordinate_label,
        "state": row["state"] if isinstance(row["state"], str) else None,
        "label": f"{coordinate_label} (near {row['name']})",
        "distance_km": round(distance, 1),
        "lat": lat,
        "lon": lon,
    }


def check_in_india(lat, lon):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude/longitude out of range")
    if not in_india(lat, lon):
        raise ValueError(f"{lat:.3f}, {lon:.3f} is outside India")
