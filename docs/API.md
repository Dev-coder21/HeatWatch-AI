# HeatWatch AI — API contract (v2)

Base URL (local): `http://localhost:8000`. Interactive schema: `/docs` (OpenAPI).
All endpoints are `GET`, return JSON, and are typed with Pydantic models in
[`backend/schemas.py`](../backend/schemas.py). Real responses captured on 2026-10-04 (refreshed after the normal bias correction)
live in [`docs/api-examples/`](api-examples/).

## Conventions

- **Coordinates** are decimal degrees (WGS84). Any point in India is accepted, including
  the Andaman & Nicobar and Lakshadweep islands (5 km tolerance at the border).
- **Grid cell.** Points are snapped to a 0.1° cell (~11 km). Profiles, normals, weather and
  responses are cached per cell; two clicks in the same cell get the same numbers
  (`location.grid_key`). Underlying weather models are ~9–25 km, so this is town/district
  scale, not street level.
- **Dates** are ISO `YYYY-MM-DD` in IST. `issued_date` is today; `forecast_date` is tomorrow.
- **Temperatures** are °C, lapse-corrected by Open-Meteo to `location.elevation_m`
  (Open-Meteo's 90 m DEM at the cell centre). Normals use the same elevation.
- **Severity** values: `"No Heatwave"`, `"Heatwave"`, `"Severe Heatwave"`.
- **Risk categories**: `Low` (0–19), `Moderate` (20–39), `High` (40–59), `Very High` (60–79),
  `Extreme` (80–100).
- **Caching**: `/api/risk` ~1 h per cell, normals and normal bias 24 h per cell, profiles forever.

### Errors

Every error body is `{"detail": "<message>"}` (422 bodies follow FastAPI's validation format).

| Status | When |
|---|---|
| 400 | Point outside India |
| 422 | Missing/invalid parameter (e.g. `lat=200`, `q` shorter than 2 chars) |
| 502 | Open-Meteo unreachable **and** no cached weather for that cell |
| 503 | Model files missing on the server |

If Open-Meteo fails but older weather for the cell is cached, `/api/risk` returns 200 with
`data.stale: true` and `data.weather_age_s` set.

Example 400 (`GET /api/risk?lat=51.5&lon=-0.12`):

```json
{
  "detail": "51.500, -0.120 is outside India"
}
```

---

## `GET /api/search?q=`

Place autocomplete (Open-Meteo geocoding, India only). Up to 10 candidates, ordered by the
geocoder's relevance. Same-named places in different states are all returned; show
`district`/`state` so users can pick.

| Param | Type | Notes |
|---|---|---|
| `q` | string, 2–100 chars | Place name |

Response fields: `query`, `results[]` with `name`, `district`, `state`, `lat`, `lon`,
`elevation` (m, may be null), `population` (may be null).

`GET /api/search?q=Jodhpur` (truncated to 3 results; real file has 10):

```json
{
  "query": "Jodhpur",
  "results": [
    {
      "name": "Jodhpur",
      "district": "Jodhpur",
      "state": "Rajasthan",
      "lat": 26.2684,
      "lon": 73.0059,
      "elevation": 237.0,
      "population": 1056191
    },
    {
      "name": "Jodhpur",
      "district": "Jamnagar",
      "state": "Gujarat",
      "lat": 21.9017,
      "lon": 70.0327,
      "elevation": 99.0,
      "population": 47329
    },
    {
      "name": "Jodhpur",
      "district": "Panna",
      "state": "Madhya Pradesh",
      "lat": 24.073,
      "lon": 79.8643,
      "elevation": 491.0,
      "population": null
    }
  ]
}
```

## `GET /api/reverse?lat=&lon=`

Name for a map click, offline (bundled GeoNames list of Indian towns with population ≥ 5,000).
Within 8 km the most populous town wins; within 15 km the nearest town; otherwise a
coordinate label `"lat, lon (near <town>)"` and `name` is the coordinates.

Fields: `name`, `label` (display string), `state`, `distance_km` (to the named town),
`lat`, `lon` (echo of the input), `in_india` (always `true`; outside India is a 400).

`GET /api/reverse?lat=23.52&lon=78.11` (rural Madhya Pradesh):

```json
{
  "name": "23.52, 78.11",
  "label": "23.52, 78.11 (near Gairtganj)",
  "state": "Madhya Pradesh",
  "distance_km": 16.6,
  "lat": 23.52,
  "lon": 78.11,
  "in_india": true
}
```

## `GET /api/risk?lat=&lon=` — main endpoint

Full assessment for the point: tomorrow's ML prediction, risk score, advisory, last 8 days
and a 7-day outlook.

**Fields**

- `location`: `name`, `label`, `state` (from the queried point), `query_lat`, `query_lon`,
  `grid_lat`, `grid_lon`, `grid_key`, `elevation_m`, `terrain_type` (`plains` | `coastal` |
  `hilly`), `distance_to_coast_km`.
- `issued_date`, `forecast_date`.
- `prediction` (tomorrow):
  - `predicted_tmax`: ML (HistGradientBoosting) predicted max temperature.
  - `raw_forecast_tmax`: Open-Meteo's own forecast for tomorrow, for comparison.
  - `normal_tmax`: the normal used for every departure, = `archive_normal_tmax` +
    `normal_bias_correction_c`.
  - `archive_normal_tmax`: mean ERA5 (`era5_seamless`) Tmax for this calendar window over the
    last 10 complete years.
  - `normal_bias_correction_c`: mean (forecast − ERA5) Tmax over the recent days where both
    exist (~9 of the last 14), clipped to ±5 °C. It puts the archive normal on the same
    footing as the forecast, so departures aren't inflated by the gap between the two
    sources. 0 if fewer than 3 overlapping days.
  - `departure` (= predicted − normal).
  - `heatwave_probability`: 0–1 from the heatwave classifier (Random Forest, not calibrated; see README).
  - `model_severity`: severity classifier output.
  - `imd_rule_severity` and `imd_rule` (`none` | `departure` | `absolute`): IMD criteria
    applied to `predicted_tmax` (includes the plains 45/47 °C absolute rule).
  - `severity`: the higher of `model_severity` and `imd_rule_severity`.
  - `consecutive_heatwave_days`: observed streak through today, +1 if tomorrow is a heatwave.
  - `humidity`: tomorrow's forecast mean RH (%).
  - `risk_score` (0–100), `risk_category`, `risk_components` (each 0–100: `temperature`,
    `heatwave_probability`, `anomaly`, `persistence`, `humidity`; weights 0.40/0.25/0.20/
    0.10/0.05 from `config/thresholds.yaml`), `top_factors` (3 largest components).
- `advisory`: `level` (`low` … `extreme`), `general` text, `audiences.outdoor_workers`,
  `audiences.elderly`, `audiences.children`, `context[]` (short factual sentences).
- `today`: today's Tmax (forecast-API value; partly forecast), normal, departure, streak.
- `recent_days[]`: 8 entries (7 past days + today; the API fetches 14 past days but only the bias estimate uses the older ones) with `tmax`, `tmin`, `normal_tmax`,
  `departure`, `imd_rule_severity`.
- `outlook[]`: 7 entries (tomorrow … +7 days) of the **raw Open-Meteo forecast** checked
  against IMD rules: `forecast_tmax`, `forecast_tmin`, `normal_tmax`, `departure`,
  `humidity`, `precipitation_mm`, `imd_rule_severity`. The ML models only predict tomorrow.
- `data`: `stale`, `weather_age_s`, `normal_bias_correction_c` (same value as in
  `prediction`), `normal_bias_overlap_days`, `source`, `grid_resolution`.

All `normal_tmax` values in `today`, `recent_days` and `outlook` include the bias correction.

`GET /api/risk?lat=28.6139&lon=77.2090` (Delhi, complete response):

```json
{
  "location": {
    "name": "Delhi",
    "label": "Delhi, Delhi",
    "state": "Delhi",
    "query_lat": 28.6139,
    "query_lon": 77.209,
    "grid_lat": 28.6,
    "grid_lon": 77.2,
    "grid_key": "28.60,77.20",
    "elevation_m": 224.0,
    "terrain_type": "plains",
    "distance_to_coast_km": 824.5
  },
  "issued_date": "2026-10-04",
  "forecast_date": "2026-10-05",
  "prediction": {
    "predicted_tmax": 34.4,
    "raw_forecast_tmax": 34.8,
    "normal_tmax": 33.6,
    "archive_normal_tmax": 32.2,
    "normal_bias_correction_c": 1.37,
    "departure": 0.8,
    "heatwave_probability": 0.0,
    "model_severity": "No Heatwave",
    "imd_rule_severity": "No Heatwave",
    "imd_rule": "none",
    "severity": "No Heatwave",
    "consecutive_heatwave_days": 0,
    "humidity": 54.0,
    "risk_score": 4.15,
    "risk_category": "Low",
    "risk_components": {
      "temperature": 0.0,
      "heatwave_probability": 0.0,
      "anomaly": 12.02,
      "persistence": 0.0,
      "humidity": 35.0
    },
    "top_factors": [
      "humidity",
      "anomaly",
      "temperature"
    ]
  },
  "advisory": {
    "level": "low",
    "general": "Current heat risk is low. Maintain normal hydration and routine heat-safety precautions.",
    "audiences": {
      "outdoor_workers": "Normal work routines are fine; keep drinking water available.",
      "elderly": "No special precautions needed beyond regular fluids.",
      "children": "Normal play is fine; send a water bottle to school."
    },
    "context": []
  },
  "today": {
    "date": "2026-10-04",
    "tmax": 35.2,
    "normal_tmax": 33.6,
    "departure": 1.6,
    "consecutive_heatwave_days": 0
  },
  "recent_days": [
    {
      "date": "2026-09-27",
      "tmax": 28.6,
      "tmin": 22.2,
      "normal_tmax": 33.5,
      "departure": -4.9,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-09-28",
      "tmax": 28.2,
      "tmin": 22.2,
      "normal_tmax": 33.5,
      "departure": -5.3,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-09-29",
      "tmax": 30.8,
      "tmin": 22.0,
      "normal_tmax": 33.4,
      "departure": -2.6,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-09-30",
      "tmax": 32.5,
      "tmin": 23.2,
      "normal_tmax": 33.5,
      "departure": -1.0,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-01",
      "tmax": 33.4,
      "tmin": 23.8,
      "normal_tmax": 33.6,
      "departure": -0.2,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-02",
      "tmax": 34.5,
      "tmin": 23.9,
      "normal_tmax": 33.6,
      "departure": 0.9,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-03",
      "tmax": 35.5,
      "tmin": 24.3,
      "normal_tmax": 33.6,
      "departure": 1.9,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-04",
      "tmax": 35.2,
      "tmin": 25.1,
      "normal_tmax": 33.6,
      "departure": 1.6,
      "imd_rule_severity": "No Heatwave"
    }
  ],
  "outlook": [
    {
      "date": "2026-10-05",
      "forecast_tmax": 34.8,
      "forecast_tmin": 24.3,
      "normal_tmax": 33.6,
      "departure": 1.2,
      "humidity": 54.0,
      "precipitation_mm": 0.0,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-06",
      "forecast_tmax": 34.4,
      "forecast_tmin": 24.4,
      "normal_tmax": 33.5,
      "departure": 0.9,
      "humidity": 66.0,
      "precipitation_mm": 0.0,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-07",
      "forecast_tmax": 34.0,
      "forecast_tmin": 23.8,
      "normal_tmax": 33.5,
      "departure": 0.5,
      "humidity": 69.0,
      "precipitation_mm": 0.0,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-08",
      "forecast_tmax": 33.2,
      "forecast_tmin": 22.8,
      "normal_tmax": 33.5,
      "departure": -0.3,
      "humidity": 71.0,
      "precipitation_mm": 1.2,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-09",
      "forecast_tmax": 31.1,
      "forecast_tmin": 21.1,
      "normal_tmax": 33.4,
      "departure": -2.3,
      "humidity": 73.0,
      "precipitation_mm": 1.5,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-10",
      "forecast_tmax": 29.9,
      "forecast_tmin": 22.4,
      "normal_tmax": 33.3,
      "departure": -3.4,
      "humidity": 73.0,
      "precipitation_mm": 0.0,
      "imd_rule_severity": "No Heatwave"
    },
    {
      "date": "2026-10-11",
      "forecast_tmax": 30.9,
      "forecast_tmin": 22.5,
      "normal_tmax": 33.2,
      "departure": -2.3,
      "humidity": 70.0,
      "precipitation_mm": 0.0,
      "imd_rule_severity": "No Heatwave"
    }
  ],
  "data": {
    "stale": false,
    "weather_age_s": 153,
    "normal_bias_correction_c": 1.37,
    "normal_bias_overlap_days": 9,
    "source": "Open-Meteo forecast API (past_days + forecast); normals from the Open-Meteo ERA5 archive (era5_seamless) for the last 10 complete years, shifted by the forecast-vs-archive bias measured on overlapping recent days; all values lapse-corrected by Open-Meteo to elevation_m",
    "grid_resolution": "~11 km cache cell; underlying weather models ~9-25 km"
  }
}
```

Other complete examples: [Churu](api-examples/risk_churu.json),
[Mumbai](api-examples/risk_mumbai.json), [Shimla](api-examples/risk_shimla.json),
[Guwahati](api-examples/risk_guwahati.json).

## `GET /api/explain?lat=&lon=`

SHAP (TreeExplainer) breakdown of tomorrow's two ML outputs for the point's cell. Cached per
cell and day. For each of `predicted_tmax` (°C) and `heatwave_probability` (0–1):
`base_value` (model average), `prediction` (= base + sum of contributions) and
`contributions[]` sorted by absolute size, each with `feature` (model name), `label`
(human-readable), `value` (input value), `shap` (signed contribution in the output's units).

Features: yesterday's / 2- / 3-days-ago Tmax, 3- and 7-day mean Tmax, yesterday's Tmin,
humidity and wind, 3-day rain, 3-day trend, today's departure from normal, normal Tmax,
month, day of year. No latitude/longitude.

`GET /api/explain?lat=28.6139&lon=77.2090` (Delhi):

```json
{
  "location": {
    "name": "Delhi",
    "grid_key": "28.60,77.20"
  },
  "issued_date": "2026-10-04",
  "forecast_date": "2026-10-05",
  "stale": false,
  "predicted_tmax": {
    "base_value": 27.3089,
    "prediction": 34.3515,
    "contributions": [
      {
        "feature": "normal_max_temp",
        "label": "Normal max temperature",
        "value": 33.58,
        "shap": 2.4872
      },
      {
        "feature": "temp_max_lag_1",
        "label": "Yesterday's max temperature",
        "value": 35.5,
        "shap": 1.9538
      },
      {
        "feature": "temp_max_rolling_7",
        "label": "7-day average max temperature",
        "value": 31.93,
        "shap": 1.478
      },
      {
        "feature": "departure",
        "label": "Today's departure from normal",
        "value": 1.62,
        "shap": 0.8567
      },
      {
        "feature": "temp_max_rolling_3",
        "label": "3-day average max temperature",
        "value": 34.47,
        "shap": 0.526
      },
      {
        "feature": "temp_max_lag_2",
        "label": "Max temperature 2 days ago",
        "value": 34.5,
        "shap": -0.0809
      },
      {
        "feature": "day_of_year",
        "label": "Day of year",
        "value": 277.0,
        "shap": -0.0751
      },
      {
        "feature": "temp_trend_3d",
        "label": "3-day temperature trend",
        "value": 2.1,
        "shap": -0.0587
      },
      {
        "feature": "temp_min_lag_1",
        "label": "Yesterday's min temperature",
        "value": 24.3,
        "shap": -0.04
      },
      {
        "feature": "precip_3d",
        "label": "Rain in the last 3 days",
        "value": 0.0,
        "shap": 0.0133
      },
      {
        "feature": "wind_lag_1",
        "label": "Yesterday's max wind",
        "value": 11.4,
        "shap": -0.0124
      },
      {
        "feature": "humidity_lag_1",
        "label": "Yesterday's humidity",
        "value": 56.0,
        "shap": -0.0075
      },
      {
        "feature": "temp_max_lag_3",
        "label": "Max temperature 3 days ago",
        "value": 33.4,
        "shap": 0.0019
      },
      {
        "feature": "month",
        "label": "Month",
        "value": 10.0,
        "shap": 0.0003
      }
    ]
  },
  "heatwave_probability": {
    "base_value": 0.4997,
    "prediction": 0.0001,
    "contributions": [
      {
        "feature": "departure",
        "label": "Today's departure from normal",
        "value": 1.62,
        "shap": -0.1714
      },
      {
        "feature": "temp_max_lag_1",
        "label": "Yesterday's max temperature",
        "value": 35.5,
        "shap": -0.0786
      },
      {
        "feature": "temp_max_rolling_7",
        "label": "7-day average max temperature",
        "value": 31.93,
        "shap": -0.0523
      },
      {
        "feature": "day_of_year",
        "label": "Day of year",
        "value": 277.0,
        "shap": -0.0442
      },
      {
        "feature": "temp_max_rolling_3",
        "label": "3-day average max temperature",
        "value": 34.47,
        "shap": -0.0419
      },
      {
        "feature": "temp_max_lag_2",
        "label": "Max temperature 2 days ago",
        "value": 34.5,
        "shap": -0.0328
      },
      {
        "feature": "wind_lag_1",
        "label": "Yesterday's max wind",
        "value": 11.4,
        "shap": -0.0256
      },
      {
        "feature": "month",
        "label": "Month",
        "value": 10.0,
        "shap": -0.0252
      },
      {
        "feature": "humidity_lag_1",
        "label": "Yesterday's humidity",
        "value": 56.0,
        "shap": -0.0244
      },
      {
        "feature": "temp_max_lag_3",
        "label": "Max temperature 3 days ago",
        "value": 33.4,
        "shap": -0.016
      },
      {
        "feature": "normal_max_temp",
        "label": "Normal max temperature",
        "value": 33.58,
        "shap": 0.0119
      },
      {
        "feature": "precip_3d",
        "label": "Rain in the last 3 days",
        "value": 0.0,
        "shap": 0.0033
      },
      {
        "feature": "temp_trend_3d",
        "label": "3-day temperature trend",
        "value": 2.1,
        "shap": -0.0031
      },
      {
        "feature": "temp_min_lag_1",
        "label": "Yesterday's min temperature",
        "value": 24.3,
        "shap": 0.0007
      }
    ]
  }
}
```

## `GET /api/featured`

Tomorrow's headline numbers for the featured places in `config/locations.csv`. Each item:
`slug`, `name`, `state`, `lat`, `lon`, `terrain_type`, `forecast_date`, `predicted_tmax`,
`heatwave_probability`, `severity`, `risk_score`, `risk_category`, `stale`, and `error`
(non-null, with the other numeric fields null, if that place failed). Use `lat`/`lon` with
`/api/risk` for details. The first call after a restart can take ~30 s (cold caches).

```json
{
  "places": [
    {
      "slug": "pune",
      "name": "Pune",
      "state": "Maharashtra",
      "lat": 18.5204,
      "lon": 73.8567,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 32.9,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 8.96,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "mumbai",
      "name": "Mumbai",
      "state": "Maharashtra",
      "lat": 19.076,
      "lon": 72.8777,
      "terrain_type": "coastal",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 35.6,
      "heatwave_probability": 0.11,
      "severity": "No Heatwave",
      "risk_score": 32.19,
      "risk_category": "Moderate",
      "stale": false,
      "error": null
    },
    {
      "slug": "delhi",
      "name": "Delhi",
      "state": "Delhi",
      "lat": 28.6139,
      "lon": 77.209,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 34.4,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 4.15,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "bengaluru",
      "name": "Bengaluru",
      "state": "Karnataka",
      "lat": 12.9716,
      "lon": 77.5946,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 29.1,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 11.46,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "shimla",
      "name": "Shimla",
      "state": "Himachal Pradesh",
      "lat": 31.1048,
      "lon": 77.1734,
      "terrain_type": "hilly",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 21.1,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 7.15,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "jaisalmer",
      "name": "Jaisalmer",
      "state": "Rajasthan",
      "lat": 26.9157,
      "lon": 70.9083,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 35.8,
      "heatwave_probability": 0.241,
      "severity": "No Heatwave",
      "risk_score": 16.12,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "ahmedabad",
      "name": "Ahmedabad",
      "state": "Gujarat",
      "lat": 23.0225,
      "lon": 72.5714,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 36.5,
      "heatwave_probability": 0.013,
      "severity": "No Heatwave",
      "risk_score": 14.74,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "nagpur",
      "name": "Nagpur",
      "state": "Maharashtra",
      "lat": 21.1458,
      "lon": 79.0882,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 33.6,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 2.87,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "lucknow",
      "name": "Lucknow",
      "state": "Uttar Pradesh",
      "lat": 26.8467,
      "lon": 80.9462,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 32.1,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 5.0,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "kolkata",
      "name": "Kolkata",
      "state": "West Bengal",
      "lat": 22.5726,
      "lon": 88.3639,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 32.6,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 7.37,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "chennai",
      "name": "Chennai",
      "state": "Tamil Nadu",
      "lat": 13.0827,
      "lon": 80.2707,
      "terrain_type": "coastal",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 33.4,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 14.1,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "hyderabad",
      "name": "Hyderabad",
      "state": "Telangana",
      "lat": 17.385,
      "lon": 78.4867,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 32.9,
      "heatwave_probability": 0.022,
      "severity": "No Heatwave",
      "risk_score": 9.46,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "guwahati",
      "name": "Guwahati",
      "state": "Assam",
      "lat": 26.1445,
      "lon": 91.7362,
      "terrain_type": "plains",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 30.4,
      "heatwave_probability": 0.006,
      "severity": "No Heatwave",
      "risk_score": 6.4,
      "risk_category": "Low",
      "stale": false,
      "error": null
    },
    {
      "slug": "port-blair",
      "name": "Port Blair",
      "state": "Andaman and Nicobar Islands",
      "lat": 11.6234,
      "lon": 92.7265,
      "terrain_type": "coastal",
      "forecast_date": "2026-10-05",
      "predicted_tmax": 31.3,
      "heatwave_probability": 0.0,
      "severity": "No Heatwave",
      "risk_score": 10.14,
      "risk_category": "Low",
      "stale": false,
      "error": null
    }
  ]
}
```

## `GET /api/health`

`status` is `ok` when models are loaded and Open-Meteo answered a probe, else `degraded`.
`cache` counts cached entries per namespace.

```json
{
  "status": "ok",
  "models_loaded": true,
  "model_error": null,
  "model_features": [
    "temp_max_lag_1",
    "temp_max_lag_2",
    "temp_max_lag_3",
    "temp_max_rolling_3",
    "temp_max_rolling_7",
    "temp_min_lag_1",
    "humidity_lag_1",
    "wind_lag_1",
    "precip_3d",
    "temp_trend_3d",
    "departure",
    "normal_max_temp",
    "month",
    "day_of_year"
  ],
  "upstream_reachable": true,
  "cache": {
    "profile": 6,
    "normals": 6,
    "weather": 6,
    "risk": 0,
    "explain": 0
  }
}
```
