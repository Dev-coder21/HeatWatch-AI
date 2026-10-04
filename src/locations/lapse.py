"""Lapse-rate correction from a grid cell's model height to a point's real elevation.

Applied identically to normals, training targets/labels and live output so that
departures from normal stay consistent:
  * normals grid (raw ERA5-Land cell values): T_point = T_cell - rate * dz
  * Open-Meteo data (already moved to the point at Open-Meteo's own rate):
        T_point = T_om + (rate_om - rate) * dz
with dz = (point elevation - cell elevation) in km.
"""

from src.config import settings


def _rates():
    cfg = settings()["lapse"]
    return cfg["rate_c_per_km"], cfg["openmeteo_rate_c_per_km"]


def dz_km(point_elevation_m, cell_elevation_m):
    if point_elevation_m is None or cell_elevation_m is None:
        return 0.0
    return (point_elevation_m - cell_elevation_m) / 1000.0


def cell_to_point(value, point_elevation_m, cell_elevation_m):
    """Correct a raw grid-cell temperature (e.g. normals) to the point's elevation."""
    rate, _ = _rates()
    return value - rate * dz_km(point_elevation_m, cell_elevation_m)


def openmeteo_to_point(value, point_elevation_m, cell_elevation_m):
    """Re-express Open-Meteo output with our configured lapse rate (no-op at equal rates)."""
    rate, rate_om = _rates()
    return value + (rate_om - rate) * dz_km(point_elevation_m, cell_elevation_m)


def openmeteo_to_cell(value, point_elevation_m, cell_elevation_m):
    """Undo Open-Meteo's correction: the raw grid-cell value (used for audits)."""
    _, rate_om = _rates()
    return value + rate_om * dz_km(point_elevation_m, cell_elevation_m)
