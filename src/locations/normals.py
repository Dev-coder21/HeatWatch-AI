"""Smoothed daily Tmax normals (1991-2020) from the precomputed India grid.

The grid stores, for every valid ERA5-Land 0.1 degree land cell, the coefficients
of a harmonic fit  T(doy) = a0 + sum_k [a_k cos(k w doy) + b_k sin(k w doy)],
with w = 2 pi / 365.25. Build it with src/locations/build_normals_grid.py.
"""

from functools import lru_cache

import numpy as np

from src.config import project_path, settings
from src.locations.geo import haversine_km

OMEGA = 2 * np.pi / 365.25


class NormalsUnavailable(RuntimeError):
    """No valid normals cell near the requested point."""


def harmonic_design(doy, harmonics):
    doy = np.asarray(doy, dtype=float)
    columns = [np.ones_like(doy)]
    for k in range(1, harmonics + 1):
        columns += [np.cos(k * OMEGA * doy), np.sin(k * OMEGA * doy)]
    return np.stack(columns, axis=-1)


def fit_harmonics(doy, values, harmonics):
    """Least-squares harmonic fit. values may be (n,) or (n, cells)."""
    design = harmonic_design(doy, harmonics)
    coefs, *_ = np.linalg.lstsq(design, values, rcond=None)
    return coefs.T  # (cells, 2H+1) or (2H+1,)


def evaluate(coefs, doy):
    harmonics = (len(coefs) - 1) // 2
    return harmonic_design(doy, harmonics) @ np.asarray(coefs, dtype=float)


@lru_cache(maxsize=1)
def _grid():
    path = project_path(settings()["normals"]["grid_file"])
    if not path.exists():
        raise NormalsUnavailable(
            f"Normals grid {path} not found. Run python -m src.locations.build_normals_grid"
        )
    data = np.load(path)
    return {
        "lat": data["lat"].astype(float),
        "lon": data["lon"].astype(float),
        "coefs": data["coefs"].astype(float),
        "source": str(data["source"]) if "source" in data else "",
    }


def nearest_cell(lat, lon):
    """Index and distance (km) of the nearest valid cell."""
    grid = _grid()
    # Pre-filter to a small box, then exact haversine.
    box = (np.abs(grid["lat"] - lat) < 1.0) & (np.abs(grid["lon"] - lon) < 1.0)
    candidates = np.flatnonzero(box)
    if candidates.size == 0:
        raise NormalsUnavailable(f"No normals cell near {lat}, {lon}")
    distances = haversine_km(lat, lon, grid["lat"][candidates], grid["lon"][candidates])
    best = int(np.argmin(distances))
    distance = float(distances[best])
    if distance > settings()["normals"]["max_fallback_km"]:
        raise NormalsUnavailable(f"Nearest normals cell is {distance:.0f} km away")
    return int(candidates[best]), distance


def normal_coefficients(lat, lon):
    index, distance = nearest_cell(lat, lon)
    grid = _grid()
    return {
        "coefs": grid["coefs"][index].tolist(),
        "cell_lat": float(grid["lat"][index]),
        "cell_lon": float(grid["lon"][index]),
        "cell_distance_km": round(distance, 2),
    }


def normals(lat, lon, doy=None):
    """Daily normal Tmax (deg C) for the given day(s) of year; all 366 days by default."""
    coefs = normal_coefficients(lat, lon)["coefs"]
    doy = np.arange(1, 367) if doy is None else doy
    return evaluate(coefs, doy)
