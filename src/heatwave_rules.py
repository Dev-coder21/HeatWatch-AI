"""IMD heatwave criteria, shared by training labels and live inference.

Per terrain (thresholds in config/thresholds.yaml):
  * the station must reach the terrain's minimum Tmax (plains 40, coastal 37, hilly 30 C);
  * then departure from normal >= 4.5 C is a heatwave, >= 6.5 C a severe heatwave.
Plains also have IMD's absolute criterion, regardless of departure:
  * Tmax >= 45 C is a heatwave, >= 47 C a severe heatwave.

Severity codes: 0 = none, 1 = heatwave, 2 = severe heatwave.
Rule codes: "none", "departure", "absolute" (absolute wins when both apply at the
same severity or gives the higher severity).
"""

import numpy as np
import pandas as pd

from src.config import thresholds

SEVERITY_NAMES = {0: "No Heatwave", 1: "Heatwave", 2: "Severe Heatwave"}


def classify(tmax, normal, terrain):
    """Vectorised IMD classification.

    tmax, normal: array-like deg C; terrain: scalar or array of hilly/coastal/plains.
    Returns a DataFrame with departure, severity (0/1/2), heatwave (0/1) and rule.
    """
    tmax = np.asarray(tmax, dtype=float)
    normal = np.asarray(normal, dtype=float)
    terrain = np.broadcast_to(np.asarray(terrain, dtype=object), tmax.shape)
    departure = tmax - normal
    rules = thresholds()["heatwave"]

    severity_dep = np.zeros(tmax.shape, dtype=int)
    severity_abs = np.zeros(tmax.shape, dtype=int)

    for name, cfg in rules.items():
        mask = terrain == name
        hot_enough = mask & (tmax >= cfg["temperature_threshold_c"])
        severity_dep[hot_enough & (departure >= cfg["departure_threshold_c"])] = 1
        severity_dep[hot_enough & (departure >= cfg["severe_departure_threshold_c"])] = 2
        if "absolute_heatwave_c" in cfg:
            severity_abs[mask & (tmax >= cfg["absolute_heatwave_c"])] = 1
            severity_abs[mask & (tmax >= cfg["absolute_severe_c"])] = 2

    # NaNs never label.
    invalid = ~np.isfinite(tmax) | ~np.isfinite(normal)
    severity_dep[invalid] = 0
    severity_abs[invalid] = 0

    severity = np.maximum(severity_dep, severity_abs)
    rule = np.where(
        severity == 0,
        "none",
        np.where(severity_abs >= severity_dep, "absolute", "departure"),
    )
    return pd.DataFrame(
        {
            "departure": departure,
            "severity": severity,
            "heatwave": (severity > 0).astype(int),
            "rule": rule,
        }
    )


def consecutive_days(heatwave):
    """Running count of consecutive heatwave days ending on each day (0 on non-heatwave days)."""
    count = 0
    out = []
    for value in heatwave:
        count = count + 1 if value else 0
        out.append(count)
    return np.array(out, dtype=int)
