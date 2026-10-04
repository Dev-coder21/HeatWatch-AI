import numpy as np
import pytest

import importlib

normals_module = importlib.import_module("src.locations.normals")
from src.locations.geo import distance_to_coast_km, in_india, snap_to_grid
from src.locations.geocode import check_in_india, reverse
from src.locations.normals import evaluate, fit_harmonics
from src.locations.profile import Cache, get_profile
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


def test_harmonic_fit_recovers_seasonal_curve():
    doy = np.tile(np.arange(1, 366), 5)
    truth = 33 + 6 * np.sin(2 * np.pi * (doy - 80) / 365.25) + 1.5 * np.cos(4 * np.pi * doy / 365.25)
    noisy = truth + np.random.default_rng(0).normal(0, 2, doy.size)
    coefs = fit_harmonics(doy, noisy, harmonics=3)
    fitted = evaluate(coefs, np.arange(1, 366))
    assert np.abs(fitted - truth[:365]).max() < 0.5
    # Smooth: no day-to-day jumps like the old monthly steps.
    assert np.abs(np.diff(fitted)).max() < 0.2


@pytest.fixture
def tiny_grid(tmp_path, monkeypatch):
    """A 3-cell normals grid; cell values encode a known annual mean."""
    path = tmp_path / "grid.npz"
    coefs = np.zeros((3, 7), dtype=np.float32)
    coefs[:, 0] = [30.0, 35.0, 20.0]
    np.savez(path, lat=np.array([28.6, 19.0, 31.1], np.float32), lon=np.array([77.2, 72.9, 77.2], np.float32), coefs=coefs)

    original = normals_module.settings

    def fake_settings():
        cfg = dict(original())
        cfg["normals"] = dict(cfg["normals"], grid_file=str(path))
        return cfg

    monkeypatch.setattr(normals_module, "settings", fake_settings)
    normals_module._grid.cache_clear()
    yield
    normals_module._grid.cache_clear()


def test_normals_lookup_nearest_cell(tiny_grid):
    values = normals_module.normals(28.61, 77.21)
    assert values.shape == (366,)
    assert np.allclose(values, 30.0)


def test_normals_fallback_to_nearest_valid_cell(tiny_grid):
    # Offshore point near Mumbai uses the nearest land cell.
    info = normals_module.normal_coefficients(18.95, 72.80)
    assert info["cell_lat"] == pytest.approx(19.0)
    assert info["cell_distance_km"] < 20


def test_normals_too_far_raises(tiny_grid):
    with pytest.raises(normals_module.NormalsUnavailable):
        normals_module.normal_coefficients(23.0, 80.0)


def test_profile_cached_after_first_call(tiny_grid, tmp_path, monkeypatch):
    calls = []

    def fake_elevation(lat, lon):
        calls.append((lat, lon))
        return 217.0

    monkeypatch.setattr("src.locations.profile.fetch_elevation", fake_elevation)
    store = Cache(tmp_path / "c.sqlite")
    first = get_profile(28.61, 77.21, store=store)
    second = get_profile(28.63, 77.18, store=store)  # same 0.1 degree cell
    assert first == second
    assert len(calls) == 1
    assert first["terrain_type"] == "plains"
    assert first["grid_key"] == "28.60,77.20"
