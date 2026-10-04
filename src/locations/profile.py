"""Location profiles (elevation, terrain, coast distance) and normals, cached in SQLite."""

import datetime as dt
import json
import sqlite3
import threading
import time

from src.config import project_path, settings
from src.locations.geo import distance_to_coast_km, snap_to_grid
from src.locations.geocode import check_in_india, reverse
from src.locations.normals import compute_normals
from src.locations.terrain import terrain_type
from src.openmeteo import get_json

_lock = threading.Lock()


class Cache:
    """Tiny key/value store with optional TTL, shared by profiles and weather."""

    def __init__(self, path=None):
        path = path or project_path(settings()["cache"]["dir"]) / "cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (ns TEXT, key TEXT, value TEXT, ts REAL, PRIMARY KEY (ns, key))"
        )
        self.conn.commit()

    def get(self, ns, key, ttl=None):
        """Return (value, age_seconds) or (None, None)."""
        with _lock:
            row = self.conn.execute("SELECT value, ts FROM kv WHERE ns=? AND key=?", (ns, key)).fetchone()
        if row is None:
            return None, None
        age = time.time() - row[1]
        if ttl is not None and age > ttl:
            return None, age
        return json.loads(row[0]), age

    def get_stale(self, ns, key):
        """Return a value regardless of age, for serving stale data when upstream fails."""
        return self.get(ns, key, ttl=None)

    def set(self, ns, key, value):
        with _lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO kv VALUES (?, ?, ?, ?)", (ns, key, json.dumps(value), time.time())
            )
            self.conn.commit()

    def count(self, ns):
        with _lock:
            return self.conn.execute("SELECT COUNT(*) FROM kv WHERE ns=?", (ns,)).fetchone()[0]


_cache = None


def cache():
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def fetch_elevation(lat, lon):
    data = get_json(settings()["api"]["elevation_url"], {"latitude": lat, "longitude": lon})
    return float(data["elevation"][0])


def grid_key(lat, lon):
    cell_lat, cell_lon = snap_to_grid(lat, lon)
    return f"{cell_lat:.2f},{cell_lon:.2f}", cell_lat, cell_lon


def get_profile(lat, lon, store=None):
    """Profile for the grid cell containing (lat, lon). One elevation call the first time."""
    check_in_india(lat, lon)
    store = store or cache()
    key, cell_lat, cell_lon = grid_key(lat, lon)
    cached, _ = store.get("profile", key)
    if cached is not None:
        return cached

    # Elevation is taken at the cell centre so every point in the cell shares a profile.
    # This one value (Open-Meteo's 90 m DEM) is passed to every archive/forecast call,
    # so normals, live weather and terrain all refer to the same height.
    elevation = fetch_elevation(cell_lat, cell_lon)
    coast_km = distance_to_coast_km(cell_lat, cell_lon)
    profile = {
        "grid_key": key,
        "lat": cell_lat,
        "lon": cell_lon,
        "elevation_m": elevation,
        "distance_to_coast_km": round(coast_km, 1),
        "terrain_type": terrain_type(cell_lat, cell_lon, elevation, coast_km=coast_km),
        "place": reverse(cell_lat, cell_lon),
    }
    store.set("profile", key, profile)
    return profile


def get_normals(profile, today=None, store=None, compute=None):
    """Daily normals around today for the profile's cell; cached per cell and date."""
    store = store or cache()
    today = today or dt.date.today()
    key = f"{profile['grid_key']}|{today.isoformat()}"
    ttl = settings()["cache"]["normals_ttl_s"]
    cached, _ = store.get("normals", key, ttl=ttl)
    if cached is not None:
        return cached
    compute = compute or compute_normals
    values = compute(profile["lat"], profile["lon"], profile["elevation_m"], today=today)
    store.set("normals", key, values)
    return values
