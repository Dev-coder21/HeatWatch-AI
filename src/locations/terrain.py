"""Terrain classification used to pick the IMD heatwave criteria."""

from src.config import settings
from src.locations.geo import distance_to_coast_km

HILLY, COASTAL, PLAINS = "hilly", "coastal", "plains"


def terrain_type(lat, lon, elevation, coast_km=None):
    """hilly if high enough, else coastal if near the sea, else plains."""
    cfg = settings()["terrain"]
    if elevation is not None and elevation >= cfg["hilly_min_elevation_m"]:
        return HILLY
    if coast_km is None:
        coast_km = distance_to_coast_km(lat, lon)
    if coast_km <= cfg["coastal_max_distance_km"]:
        return COASTAL
    return PLAINS
