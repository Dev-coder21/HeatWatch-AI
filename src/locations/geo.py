"""Geometry helpers: India boundary check, coastline distance, grid snapping."""

import json
import math
from functools import lru_cache

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

from src.config import project_path, settings

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _read_geometry(relative):
    with open(project_path(relative)) as file:
        data = json.load(file)
    return unary_union([shape(f["geometry"]) for f in data["features"]])


@lru_cache(maxsize=1)
def _india():
    geom = _read_geometry(settings()["geo"]["india_boundary_file"])
    return geom, prep(geom)


@lru_cache(maxsize=1)
def _coast_points():
    """Coastline vertices densified to ~2 km, as radians, for fast haversine."""
    geom = _read_geometry(settings()["geo"]["coastline_file"])
    lines = getattr(geom, "geoms", [geom])
    points = []
    for line in lines:
        length = line.length
        steps = max(int(length / 0.02), 1)
        for i in range(steps + 1):
            p = line.interpolate(i / steps, normalized=True)
            points.append((p.y, p.x))
    return np.array(points)


def in_india(lat, lon):
    """True if the point is in India or within the configured buffer of it."""
    geom, prepared = _india()
    point = Point(lon, lat)
    if prepared.contains(point):
        return True
    buffer_km = settings()["geo"]["boundary_buffer_km"]
    # Cheap degree pre-check before the exact haversine distance.
    if geom.distance(point) > buffer_km / 80.0:
        return False
    return _nearest_boundary_km(lat, lon) <= buffer_km


def _nearest_boundary_km(lat, lon):
    geom, _ = _india()
    nearest = nearest_points(geom, Point(lon, lat))[0]
    return float(haversine_km(lat, lon, nearest.y, nearest.x))


def distance_to_coast_km(lat, lon):
    coast = _coast_points()
    return float(haversine_km(lat, lon, coast[:, 0], coast[:, 1]).min())


def snap_to_grid(lat, lon, step=None):
    step = step or settings()["geo"]["grid_step_deg"]
    decimals = max(0, -int(math.floor(math.log10(step))))
    return (
        round(round(lat / step) * step, decimals),
        round(round(lon / step) * step, decimals),
    )
