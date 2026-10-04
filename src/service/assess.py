"""Live heat-risk assessment for any point in India (no web framework imports).

assess(lat, lon):
  profile (cached) -> weather (cached ~1 h) -> archive normals + forecast-vs-archive
  bias (both cached per cell/day)
  -> today's feature row -> models predict tomorrow -> IMD rules -> risk score
  -> advisory, plus a 7-day outlook of the raw forecast checked against IMD rules.
"""

import datetime as dt

import numpy as np
import pandas as pd

from src.advisory import generate_advisory
from src.config import settings, thresholds
from src.heatwave_rules import SEVERITY_NAMES, classify, consecutive_days
from src.locations.geocode import check_in_india, reverse
from src.locations.profile import cache, get_normal_bias, get_normals, get_profile, grid_key
from src.risk_score import calculate_risk_score, risk_category
from src.service.models import explainer, load_models
from src.service.weather import get_weather

FEATURE_LABELS = {
    "temp_max_lag_1": "Yesterday's max temperature",
    "temp_max_lag_2": "Max temperature 2 days ago",
    "temp_max_lag_3": "Max temperature 3 days ago",
    "temp_max_rolling_3": "3-day average max temperature",
    "temp_max_rolling_7": "7-day average max temperature",
    "temp_min_lag_1": "Yesterday's min temperature",
    "humidity_lag_1": "Yesterday's humidity",
    "wind_lag_1": "Yesterday's max wind",
    "precip_3d": "Rain in the last 3 days",
    "temp_trend_3d": "3-day temperature trend",
    "departure": "Today's departure from normal",
    "normal_max_temp": "Normal max temperature",
    "month": "Month",
    "day_of_year": "Day of year",
}
SEVERITY_RANK = {"No Heatwave": 0, "Heatwave": 1, "Severe Heatwave": 2}
MODEL_SEVERITY_NAMES = {name: rank for name, rank in SEVERITY_RANK.items()}


class OutsideIndia(ValueError):
    pass


def _r(value, digits=1):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def build_features(weather, normals, today_index):
    """Feature row for 'today' in the exact form the models were trained on."""
    w = pd.DataFrame(weather)
    t = today_index
    tmax = w["temperature_max"].values
    today = pd.Timestamp(w.loc[t, "date"])
    normal_today = normals[today.date().isoformat()]
    row = {
        "temp_max_lag_1": tmax[t - 1],
        "temp_max_lag_2": tmax[t - 2],
        "temp_max_lag_3": tmax[t - 3],
        "temp_max_rolling_3": float(np.mean(tmax[t - 3:t])),
        "temp_max_rolling_7": float(np.mean(tmax[t - 7:t])),
        "temp_min_lag_1": w["temperature_min"].values[t - 1],
        "humidity_lag_1": w["relative_humidity"].values[t - 1],
        "wind_lag_1": w["wind_speed"].values[t - 1],
        "precip_3d": float(np.nansum(w["precipitation"].values[t - 3:t])),
        "temp_trend_3d": tmax[t - 1] - tmax[t - 3],
        "departure": tmax[t] - normal_today,
        "normal_max_temp": normal_today,
        "month": today.month,
        "day_of_year": today.dayofyear,
    }
    return pd.DataFrame([row])


def _context(lat, lon, store=None):
    store = store or cache()
    try:
        profile = get_profile(lat, lon, store=store)
    except ValueError as error:
        raise OutsideIndia(str(error)) from error
    weather, stale, age = get_weather(profile, store)
    today_index = settings()["api"]["past_days"]
    today = dt.date.fromisoformat(weather[today_index]["date"])
    archive_normals = get_normals(profile, today=today, store=store)
    forecast_past = {r["date"]: r["temperature_max"] for r in weather[:today_index]}
    bias = get_normal_bias(profile, forecast_past, today=today, store=store)
    # Put the archive normals on the forecast's footing before any departure is computed.
    normals = {d: round(v + bias["bias_c"], 2) for d, v in archive_normals.items()}
    features = build_features(weather, normals, today_index)
    return profile, weather, stale, age, today_index, normals, features, bias, archive_normals


def assess(lat, lon, store=None):
    store = store or cache()
    profile, weather, stale, age, t, normals, features, bias, archive_normals = _context(lat, lon, store)
    models = load_models()
    terrain = profile["terrain_type"]
    w = pd.DataFrame(weather)
    w["normal_max_temp"] = w["date"].map(normals)

    # Observed days through today: IMD labels and the running heatwave streak.
    # The last 7 days + today (earlier past days only serve the bias estimate).
    observed = w.iloc[t - 7: t + 1]
    observed_labels = classify(observed["temperature_max"], observed["normal_max_temp"], terrain)
    streak_today = int(consecutive_days(observed_labels["heatwave"].values)[-1])

    # Tomorrow: model predictions.
    predicted = float(models["temperature"].predict(features[models["temperature_features"]])[0])
    cls_x = features[models["classification_features"]]
    probability = float(models["heatwave"].predict_proba(cls_x)[0][list(models["heatwave"].classes_).index(1)])
    model_severity = str(models["severity"].predict(cls_x)[0])

    tomorrow = w.iloc[t + 1]
    normal_tomorrow = float(tomorrow["normal_max_temp"])
    rule = classify([predicted], [normal_tomorrow], terrain).iloc[0]
    rule_severity = SEVERITY_NAMES[int(rule["severity"])]
    # Conservative: report the higher of the model's and the IMD rule's severity.
    severity = max([model_severity, rule_severity], key=SEVERITY_RANK.get)
    is_heatwave = severity != "No Heatwave"
    streak = streak_today + 1 if is_heatwave else 0
    humidity = float(tomorrow["relative_humidity"]) if pd.notna(tomorrow["relative_humidity"]) else 50.0

    cfg = thresholds()
    risk = calculate_risk_score(
        predicted_temp=predicted,
        heatwave_probability=probability,
        departure=predicted - normal_tomorrow,
        consecutive_heatwave_days=streak,
        humidity=humidity,
        region_type=terrain,
        thresholds=cfg,
    )
    category = risk_category(risk["risk_score"], cfg)
    components = {k.replace("_component", ""): v for k, v in risk.items() if k != "risk_score"}
    top_factors = [
        name for name, _ in sorted(components.items(), key=lambda item: item[1], reverse=True)[:3]
    ]

    # 7-day outlook: raw forecast against IMD rules (the ML models only predict tomorrow).
    future = w.iloc[t + 1:]
    outlook_labels = classify(future["temperature_max"], future["normal_max_temp"], terrain)
    outlook = [
        {
            "date": row["date"],
            "forecast_tmax": _r(row["temperature_max"]),
            "forecast_tmin": _r(row["temperature_min"]),
            "normal_tmax": _r(row["normal_max_temp"]),
            "departure": _r(label["departure"]),
            "humidity": _r(row["relative_humidity"], 0),
            "precipitation_mm": _r(row["precipitation"]),
            "imd_rule_severity": SEVERITY_NAMES[int(label["severity"])],
        }
        for (_, row), (_, label) in zip(future.iterrows(), outlook_labels.iterrows())
    ]
    recent = [
        {
            "date": row["date"],
            "tmax": _r(row["temperature_max"]),
            "tmin": _r(row["temperature_min"]),
            "normal_tmax": _r(row["normal_max_temp"]),
            "departure": _r(label["departure"]),
            "imd_rule_severity": SEVERITY_NAMES[int(label["severity"])],
        }
        for (_, row), (_, label) in zip(observed.iterrows(), observed_labels.iterrows())
    ]

    # Name the place from the clicked/searched point, not the cell centre.
    place = reverse(lat, lon)
    return {
        "location": {
            "name": place["name"],
            "label": place["label"],
            "state": place["state"],
            "query_lat": lat,
            "query_lon": lon,
            "grid_lat": profile["lat"],
            "grid_lon": profile["lon"],
            "grid_key": profile["grid_key"],
            "elevation_m": profile["elevation_m"],
            "terrain_type": terrain,
            "distance_to_coast_km": profile["distance_to_coast_km"],
        },
        "issued_date": w.loc[t, "date"],
        "forecast_date": tomorrow["date"],
        "prediction": {
            "predicted_tmax": _r(predicted),
            "raw_forecast_tmax": _r(tomorrow["temperature_max"]),
            "normal_tmax": _r(normal_tomorrow),
            "archive_normal_tmax": _r(archive_normals[tomorrow["date"]]),
            "normal_bias_correction_c": bias["bias_c"],
            "departure": _r(predicted - normal_tomorrow),
            "heatwave_probability": round(probability, 3),
            "model_severity": model_severity,
            "imd_rule_severity": rule_severity,
            "imd_rule": rule["rule"],
            "severity": severity,
            "consecutive_heatwave_days": streak,
            "humidity": _r(humidity, 0),
            "risk_score": risk["risk_score"],
            "risk_category": category,
            "risk_components": components,
            "top_factors": top_factors,
        },
        "advisory": generate_advisory(category, severity, predicted, probability, humidity),
        "today": {
            "date": w.loc[t, "date"],
            "tmax": _r(w.loc[t, "temperature_max"]),
            "normal_tmax": _r(w.loc[t, "normal_max_temp"]),
            "departure": _r(features["departure"].iloc[0]),
            "consecutive_heatwave_days": streak_today,
        },
        "recent_days": recent,
        "outlook": outlook,
        "data": {
            "stale": stale,
            "weather_age_s": round(age or 0.0),
            "normal_bias_correction_c": bias["bias_c"],
            "normal_bias_overlap_days": bias["overlap_days"],
            "source": "Open-Meteo forecast API (past_days + forecast); normals from the Open-Meteo ERA5 archive (era5_seamless) for the last 10 complete years, shifted by the forecast-vs-archive bias measured on overlapping recent days; all values lapse-corrected by Open-Meteo to elevation_m",
            "grid_resolution": "~11 km cache cell; underlying weather models ~9-25 km",
        },
    }


def explain(lat, lon, store=None):
    """SHAP contributions for tomorrow's Tmax and heatwave probability; cached per cell/day."""
    store = store or cache()
    profile, weather, stale, age, t, normals, features, _, _ = _context(lat, lon, store)
    key = f"{profile['grid_key']}|{weather[t]['date']}"
    cached, _ = store.get("explain", key, ttl=settings()["cache"]["weather_ttl_s"])
    if cached is not None:
        return cached

    models = load_models()
    result = {
        "location": {"name": reverse(lat, lon)["name"], "grid_key": profile["grid_key"]},
        "issued_date": weather[t]["date"],
        "forecast_date": weather[t + 1]["date"],
        "stale": stale,
    }
    for name, feature_key, label in [
        ("temperature", "temperature_features", "predicted_tmax"),
        ("heatwave", "classification_features", "heatwave_probability"),
    ]:
        x = features[models[feature_key]]
        values = explainer(name).shap_values(x)
        base = explainer(name).expected_value
        if name == "heatwave":
            # Classifier: explain the probability of class 1.
            values = np.asarray(values)
            values = values[0, :, 1] if values.ndim == 3 else values[1][0]
            base = np.asarray(base).ravel()[1]
        else:
            values = np.asarray(values)[0]
            base = float(np.asarray(base).ravel()[0])
        contributions = sorted(
            (
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "value": _r(float(x[feature].iloc[0]), 2),
                    "shap": round(float(v), 4),
                }
                for feature, v in zip(models[feature_key], values)
            ),
            key=lambda item: abs(item["shap"]),
            reverse=True,
        )
        result[label] = {
            "base_value": round(float(base), 4),
            "prediction": round(float(base + sum(c["shap"] for c in contributions)), 4),
            "contributions": contributions,
        }
    store.set("explain", key, result)
    return result


def assess_cached(lat, lon, store=None):
    """assess() with a ~1 h response cache per grid cell."""
    store = store or cache()
    try:
        check_in_india(lat, lon)
    except ValueError as error:
        raise OutsideIndia(str(error)) from error
    key, _, _ = grid_key(lat, lon)
    cached, _ = store.get("risk", key, ttl=settings()["cache"]["response_ttl_s"])
    if cached is not None:
        place = reverse(lat, lon)
        cached["location"].update(
            query_lat=lat, query_lon=lon, name=place["name"], label=place["label"], state=place["state"]
        )
        return cached
    result = assess(lat, lon, store)
    if not result["data"]["stale"]:
        store.set("risk", key, result)
    return result
