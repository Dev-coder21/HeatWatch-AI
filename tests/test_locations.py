import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.locations.geo import distance_to_coast_km, in_india, snap_to_grid
from src.locations.geocode import check_in_india, reverse
from src.locations.normals import compute_normals
from src.locations.profile import Cache, get_normals, get_profile
from src.locations.terrain import terrain_type

# (lat, lon, elevation from the Open-Meteo 90 m DEM)
HILLY = {
    "Shimla": (31.10, 77.17, 1940),
    "Darjeeling": (27.04, 88.26, 1924),
    "Ooty": (11.41, 76.70, 2231),
}
COASTAL = {
    "Mumbai": (19.07, 72.88, 9),
    "Chennai": (13.08, 80.27, 10),
    "Kochi": (9.93, 76.27, 2),
    "Puri": (19.81, 85.83, 12),
}
PLAINS = {
    "Delhi": (28.61, 77.21, 217),
    "Jaipur": (26.91, 75.79, 430),
    "Nagpur": (21.15, 79.09, 310),
    "Lucknow": (26.85, 80.95, 117),
}


@pytest.mark.parametrize("name", HILLY)
def test_hilly(name):
    assert terrain_type(*HILLY[name]) == "hilly"


@pytest.mark.parametrize("name", COASTAL)
def test_coastal(name):
    assert terrain_type(*COASTAL[name]) == "coastal"


@pytest.mark.parametrize("name", PLAINS)
def test_plains(name):
    assert terrain_type(*PLAINS[name]) == "plains"


def test_hilly_beats_coastal():
    # A high point right next to the sea is still hilly.
    lat, lon, _ = COASTAL["Mumbai"]
    assert terrain_type(lat, lon, 1200) == "hilly"


def test_coast_distance_reasonable():
    assert distance_to_coast_km(19.07, 72.88) < 10
    assert distance_to_coast_km(28.61, 77.21) > 700


@pytest.mark.parametrize(
    "lat, lon, expected",
    [
        (28.61, 77.21, True),   # Delhi
        (11.62, 92.73, True),   # Port Blair
        (10.57, 72.64, True),   # Kavaratti
        (34.15, 77.58, True),   # Leh
        (31.55, 74.34, False),  # Lahore
        (23.81, 90.41, False),  # Dhaka
        (6.93, 79.85, False),   # Colombo
        (51.5, -0.12, False),   # London
    ],
)
def test_in_india(lat, lon, expected):
    assert in_india(lat, lon) is expected


def test_check_in_india_raises():
    with pytest.raises(ValueError):
        check_in_india(51.5, -0.12)


def test_snap_to_grid():
    assert snap_to_grid(28.6139, 77.2090) == (28.6, 77.2)
    assert snap_to_grid(18.96, 72.84) == (19.0, 72.8)


def test_reverse_named_place():
    result = reverse(26.27, 73.01)
    assert result["name"] == "Jodhpur"
    assert result["state"] == "Rajasthan"


def test_reverse_remote_point_gets_coordinate_label():
    result = reverse(27.9, 70.6)  # Thar desert, far from towns
    assert result["name"] == "27.90, 70.60"


def fake_archive(calls):
    """Archive stub: Tmax = 30 + day-of-year/100, so normals are predictable."""

    def fetch(lat, lon, elevation, start, end):
        calls.append({"lat": lat, "lon": lon, "elevation": elevation, "start": start, "end": end})
        days = pd.date_range(start, end)
        return pd.Series(30 + days.dayofyear / 100.0, index=days)

    return fetch


def test_normals_use_last_ten_complete_years():
    calls = []
    today = dt.date(2026, 5, 20)
    normals = compute_normals(28.6, 77.2, 224.0, today=today, fetch=fake_archive(calls))
    years = sorted(c["start"].year for c in calls)
    assert years == list(range(2016, 2026))
    # Covers today - 7 .. today + 7.
    assert min(normals) == "2026-05-13" and max(normals) == "2026-05-27"
    # Mean over a +/-7 day window of a linear series is the centre value.
    assert normals["2026-05-20"] == pytest.approx(30 + 140 / 100.0, abs=0.02)


def test_normals_cost_about_twenty_call_units():
    calls = []
    compute_normals(28.6, 77.2, 224.0, today=dt.date(2026, 5, 20), fetch=fake_archive(calls))
    # Open-Meteo counts one call-unit per 14 days of data per request.
    units = sum(max(1.0, ((c["end"] - c["start"]).days + 1) / 14) for c in calls)
    assert len(calls) == 10
    assert units <= 22


def test_normals_handle_leap_day_and_year_boundary():
    calls = []
    normals = compute_normals(28.6, 77.2, 224.0, today=dt.date(2028, 2, 29), fetch=fake_archive(calls))
    assert "2028-02-29" in normals
    normals = compute_normals(28.6, 77.2, 224.0, today=dt.date(2026, 1, 2), fetch=fake_archive(calls))
    assert "2025-12-26" in normals and "2026-01-09" in normals


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("src.locations.profile.fetch_elevation", lambda lat, lon: 217.0)
    return Cache(tmp_path / "c.sqlite")


def test_profile_cached_after_first_call(store, monkeypatch):
    calls = []
    monkeypatch.setattr("src.locations.profile.fetch_elevation", lambda lat, lon: calls.append(1) or 217.0)
    first = get_profile(28.61, 77.21, store=store)
    second = get_profile(28.63, 77.18, store=store)  # same 0.1 degree cell
    assert first == second
    assert len(calls) == 1
    assert first["terrain_type"] == "plains"
    assert first["grid_key"] == "28.60,77.20"


def test_normals_cached_per_cell_and_day(store):
    calls = []
    profile = get_profile(28.61, 77.21, store=store)

    def compute(lat, lon, elevation, today):
        calls.append(1)
        return compute_normals(lat, lon, elevation, today=today, fetch=fake_archive([]))

    today = dt.date(2026, 5, 20)
    get_normals(profile, today=today, store=store, compute=compute)
    get_normals(profile, today=today, store=store, compute=compute)
    assert len(calls) == 1


def test_profile_outside_india_raises(store):
    with pytest.raises(ValueError):
        get_profile(51.5, -0.12, store=store)
