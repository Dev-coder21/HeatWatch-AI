"""Download training data for every site in config/training_sites.csv.

Per site and calendar year, writes CSVs under data/raw/sites/<site_id>/:
  observed_<year>.csv   ERA5-Land Tmax/Tmin/RH (ERA5 fallback where ERA5-Land is
                        null), plus ERA5 wind/precip/pressure (ERA5-Land has none
                        of those in Open-Meteo).
  hindcast_<year>.csv   Historical Forecast API, same model as live inputs:
                        the "observed-like" values the live past_days call returns.
  leads_<model>_<year>.csv  Previous Runs API, hourly temperature at lead days 1-7,
                        reduced to daily max on IST calendar days.

Finished years are skipped (resume). The current year is re-fetched unless
--no-refresh is given. Throttled per config/settings.yaml.

    python -m src.training.download [--sites churu,delhi] [--no-refresh]
"""

import argparse
import datetime as dt
import time

import pandas as pd

from src.config import project_path, settings
from src.openmeteo import UpstreamError, get_json

SITES_FILE = project_path("config/training_sites.csv")
OUT_DIR = project_path("data/raw/sites")

START_YEAR = 2021
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "surface_pressure_mean",
]
LAND_VARS = DAILY_VARS[:3]  # the only ones ERA5-Land provides
LEAD_DAYS = range(1, 8)
# First date each model has lead days 1-7 in the Previous Runs API (Phase 0 probes).
LEADS_START = {"gfs_seamless": "2021-04-01", "ecmwf_ifs025": "2024-02-01"}
# ERA5 is final about 5 days behind real time.
REANALYSIS_LAG_DAYS = 6

RENAME = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "relative_humidity_2m_mean": "rh_mean",
    "wind_speed_10m_max": "wind_max",
    "precipitation_sum": "precip",
    "surface_pressure_mean": "pressure",
}


def _throttle():
    time.sleep(settings()["api"]["throttle_s"])


def _daily_frame(data, suffix=""):
    daily = data["daily"]
    frame = pd.DataFrame({"date": daily["time"]})
    for var in DAILY_VARS:
        key = f"{var}{suffix}"
        if key in daily:
            frame[RENAME[var]] = daily[key]
    return frame


def fetch_observed(lat, lon, start, end):
    cfg = settings()["api"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "models": "era5_land,era5",
        "timezone": cfg["timezone"],
    }
    data = get_json(cfg["archive_url"], params)
    land = _daily_frame(data, "_era5_land")
    era5 = _daily_frame(data, "_era5")
    out = era5.copy()
    out["temp_source"] = "era5"
    has_land = pd.to_numeric(land["tmax"], errors="coerce").notna()
    for col in ["tmax", "tmin", "rh_mean"]:
        out[col] = pd.to_numeric(land[col], errors="coerce").where(has_land, pd.to_numeric(out[col], errors="coerce"))
    out.loc[has_land, "temp_source"] = "era5_land"
    return out


def fetch_hindcast(lat, lon, start, end, model):
    cfg = settings()["api"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "models": model,
        "timezone": cfg["timezone"],
    }
    return _daily_frame(get_json(cfg["historical_forecast_url"], params))


def fetch_leads(lat, lon, start, end, model):
    cfg = settings()["api"]
    hourly = ",".join(f"temperature_2m_previous_day{d}" for d in LEAD_DAYS)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": hourly,
        "models": model,
        "timezone": cfg["timezone"],
    }
    data = get_json(cfg["previous_runs_url"], params)["hourly"]
    frame = pd.DataFrame(data)
    frame["date"] = frame.pop("time").str[:10]
    # Daily max over the IST calendar day; a day needs all 24 hours to count.
    columns = {f"temperature_2m_previous_day{d}": f"fc_tmax_lead{d}" for d in LEAD_DAYS}
    frame = frame.rename(columns=columns)
    grouped = frame.groupby("date")
    daily = grouped.max(numeric_only=True)
    complete = grouped.count() == 24
    daily = daily.where(complete[daily.columns])
    return daily.reset_index()


def _year_range(year, earliest, latest):
    start = max(dt.date(year, 1, 1), earliest)
    end = min(dt.date(year, 12, 31), latest)
    return (start.isoformat(), end.isoformat()) if start <= end else None


def download_site(site, refresh_current=True):
    site_dir = OUT_DIR / site["site_id"]
    site_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    reanalysis_end = today - dt.timedelta(days=REANALYSIS_LAG_DAYS)
    api = settings()["api"]
    models = [api["forecast_model"], *api["baseline_models"]]
    calls = 0

    jobs = []
    for year in range(START_YEAR, today.year + 1):
        jobs.append(("observed", None, year, dt.date(START_YEAR, 1, 1), reanalysis_end))
        jobs.append(("hindcast", api["forecast_model"], year, dt.date(START_YEAR, 1, 1), today - dt.timedelta(days=1)))
        for model in models:
            earliest = dt.date.fromisoformat(LEADS_START[model])
            jobs.append(("leads", model, year, earliest, today - dt.timedelta(days=1)))

    for kind, model, year, earliest, latest in jobs:
        span = _year_range(year, earliest, latest)
        if span is None:
            continue
        name = f"{kind}_{year}.csv" if kind != "leads" else f"leads_{model}_{year}.csv"
        target = site_dir / name
        is_current = year == today.year
        if target.exists() and not (is_current and refresh_current):
            continue
        lat, lon = site["latitude"], site["longitude"]
        if kind == "observed":
            frame = fetch_observed(lat, lon, *span)
        elif kind == "hindcast":
            frame = fetch_hindcast(lat, lon, *span, model)
        else:
            frame = fetch_leads(lat, lon, *span, model)
        frame.to_csv(target, index=False)
        calls += 1
        _throttle()
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", help="comma-separated site_ids (default: all)")
    parser.add_argument("--no-refresh", action="store_true", help="don't re-fetch the current year")
    args = parser.parse_args()

    sites = pd.read_csv(SITES_FILE)
    if args.sites:
        sites = sites[sites["site_id"].isin(args.sites.split(","))]

    started = time.time()
    for i, site in enumerate(sites.to_dict("records"), 1):
        # Daily/hourly quotas can run out mid-run: wait and resume the same site.
        for wait_min in (15, 30, 60, 60, 120, 240, 480):
            try:
                calls = download_site(site, refresh_current=not args.no_refresh)
                break
            except UpstreamError as error:
                print(f"{site['site_id']}: {error}; waiting {wait_min} min", flush=True)
                time.sleep(wait_min * 60)
        else:
            raise SystemExit(f"Giving up on {site['site_id']}; rerun later to resume")
        print(f"[{i}/{len(sites)}] {site['site_id']}: {calls} requests ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
