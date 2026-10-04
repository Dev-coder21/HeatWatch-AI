"""Is ERA5-Land a fair stand-in for what the live forecast API returns as past days?

Live inference builds lag features from the forecast API's past_days (GFS
analysis/short-range). Training would build them from ERA5-Land. This compares:
  1. forecast-API past_days vs Historical Forecast API (same model, last 92 days):
     are they the same product?
  2. Historical Forecast API vs ERA5-Land/ERA5 observed, 2021-2026: bias, MAE
     overall, by month and by terrain.

    python -m src.training.check_lag_bias
Writes outputs/metrics/lag_bias.json.
"""

import datetime as dt
import json
import time

import pandas as pd

from src.config import project_path, settings
from src.openmeteo import get_json
from src.training.download import (
    DAILY_VARS,
    REANALYSIS_LAG_DAYS,
    _daily_frame,
    fetch_hindcast,
    fetch_observed,
)

SAMPLE = [
    "delhi", "churu", "jaisalmer", "nagpur", "titlagarh", "lucknow", "hyderabad", "bengaluru",
    "mumbai", "chennai", "kochi", "kolkata", "kavaratti", "port-blair",
    "shimla", "leh", "darjeeling", "ooty", "guwahati", "imphal",
]
COMPARE = ["tmax", "tmin", "rh_mean"]


def past_days(lat, lon, model):
    cfg = settings()["api"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 92,
        "forecast_days": 1,
        "daily": ",".join(DAILY_VARS),
        "models": model,
        "timezone": cfg["timezone"],
    }
    return _daily_frame(get_json(cfg["forecast_url"], params))


def stats(diff):
    return {"bias": round(float(diff.mean()), 2), "mae": round(float(diff.abs().mean()), 2), "n": int(diff.count())}


def main():
    model = settings()["api"]["forecast_model"]
    sites = pd.read_csv(project_path("config/training_sites.csv")).set_index("site_id").loc[SAMPLE]
    today = dt.date.today()
    end_obs = (today - dt.timedelta(days=REANALYSIS_LAG_DAYS)).isoformat()
    end_fc = (today - dt.timedelta(days=1)).isoformat()

    equivalence, rows = [], []
    for site_id, site in sites.iterrows():
        lat, lon = site["latitude"], site["longitude"]
        obs = fetch_observed(lat, lon, "2021-04-01", end_obs)
        hind = fetch_hindcast(lat, lon, "2021-04-01", end_fc, model)
        live = past_days(lat, lon, model)
        time.sleep(1)

        both = live.merge(hind, on="date", suffixes=("_live", "_hind")).dropna()
        both = both[both["date"] < today.isoformat()]
        equivalence.append({"site": site_id, **stats(both["tmax_live"] - both["tmax_hind"])})

        merged = hind.merge(obs, on="date", suffixes=("_fc", "_obs"))
        merged["site"] = site_id
        merged["terrain"] = site["terrain_type"]
        merged["month"] = pd.to_datetime(merged["date"]).dt.month
        rows.append(merged)
        print(f"{site_id}: done", flush=True)

    data = pd.concat(rows)
    report = {"model": model, "live_vs_hindcast_tmax": equivalence, "hindcast_minus_observed": {}}
    for var in COMPARE:
        diff = (data[f"{var}_fc"] - data[f"{var}_obs"]).rename("d")
        frame = data.assign(d=diff)
        report["hindcast_minus_observed"][var] = {
            "overall": stats(diff),
            "by_terrain": {k: stats(g["d"]) for k, g in frame.groupby("terrain")},
            "by_month": {int(k): stats(g["d"]) for k, g in frame.groupby("month")},
            "by_site": {k: stats(g["d"]) for k, g in frame.groupby("site")},
        }

    out = project_path("outputs/metrics/lag_bias.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["overall"] for k, v in report["hindcast_minus_observed"].items()}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
