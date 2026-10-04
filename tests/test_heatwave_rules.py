import numpy as np
import pytest

from src.heatwave_rules import classify, consecutive_days


def one(tmax, normal, terrain):
    row = classify([tmax], [normal], terrain).iloc[0]
    return int(row["severity"]), row["rule"]


@pytest.mark.parametrize(
    "tmax, normal, terrain, expected",
    [
        # Plains: needs 40 C and departure 4.5 / 6.5
        (39.9, 30.0, "plains", (0, "none")),
        (40.0, 35.5, "plains", (1, "departure")),
        (40.0, 35.6, "plains", (0, "none")),
        (42.0, 35.5, "plains", (2, "departure")),
        # Plains absolute rule, regardless of departure
        (44.9, 43.0, "plains", (0, "none")),
        (45.0, 44.0, "plains", (1, "absolute")),
        (46.9, 46.0, "plains", (1, "absolute")),
        (47.0, 46.5, "plains", (2, "absolute")),
        # Absolute gives severe even if departure only gives heatwave
        (47.0, 42.0, "plains", (2, "absolute")),
        # Departure severe beats absolute heatwave
        (45.5, 38.0, "plains", (2, "departure")),
        # Coastal: 37 C minimum, no absolute rule
        (36.9, 30.0, "coastal", (0, "none")),
        (37.0, 32.5, "coastal", (1, "departure")),
        (37.0, 30.5, "coastal", (2, "departure")),
        (46.0, 45.0, "coastal", (0, "none")),
        # Hilly: 30 C minimum, no absolute rule
        (29.9, 20.0, "hilly", (0, "none")),
        (30.0, 25.5, "hilly", (1, "departure")),
        (31.0, 24.5, "hilly", (2, "departure")),
        (46.0, 45.0, "hilly", (0, "none")),
    ],
)
def test_imd_rules(tmax, normal, terrain, expected):
    assert one(tmax, normal, terrain) == expected


def test_nan_never_labels():
    result = classify([np.nan, 48.0], [30.0, np.nan], "plains")
    assert result["severity"].tolist() == [0, 0]


def test_vectorised_mixed_terrain():
    result = classify([41, 38, 31], [36, 33, 26], ["plains", "coastal", "hilly"])
    assert result["heatwave"].tolist() == [1, 1, 1]


def test_consecutive_days():
    assert consecutive_days([0, 1, 1, 0, 1, 1, 1]).tolist() == [0, 1, 2, 0, 1, 2, 3]
