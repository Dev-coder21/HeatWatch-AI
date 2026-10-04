"""On-demand daily Tmax normals for any point.

For a reference date D, the normal for a day d is the mean archive Tmax over
[d - window, d + window] in each of the last N complete years. One archive call
per year covers every day from D - past_days to D + forecast_days, so a new city
costs about N small calls (~21 Open-Meteo call-units with the defaults).

All archive calls pass the profile's elevation, so Open-Meteo lapse-corrects the
gridded values to the same height used for the live forecast.
"""

import datetime as dt

import numpy as np
import pandas as pd

from src.config import settings
from src.openmeteo import get_json


def archive_tmax(lat, lon, elevation, start, end):
    cfg = settings()["api"]
    data = get_json(
        cfg["archive_url"],
        {
            "latitude": lat,
            "longitude": lon,
            "elevation": elevation,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max",
            "timezone": cfg["timezone"],
        },
    )
    daily = data["daily"]
    return pd.Series(daily["temperature_2m_max"], index=pd.to_datetime(daily["time"]), dtype=float)


def _shift_year(day, years_back):
    try:
        return day.replace(year=day.year - years_back)
    except ValueError:  # 29 Feb in a non-leap year
        return day.replace(year=day.year - years_back, day=28)


def compute_normals(lat, lon, elevation, today=None, fetch=archive_tmax):
    """Return {iso_date: normal_tmax} for today - past_days .. today + forecast_days."""
    cfg = settings()["normals"]
    today = today or dt.date.today()
    window = cfg["window_days"]
    first = today - dt.timedelta(days=cfg["past_days_needed"])
    last = today + dt.timedelta(days=cfg["forecast_days_needed"])
    days = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]

    # Last N complete calendar years.
    years_back = [today.year - year for year in range(today.year - cfg["years"], today.year)]
    samples = {day: [] for day in days}
    for back in years_back:
        start = _shift_year(first, back) - dt.timedelta(days=window)
        end = _shift_year(last, back) + dt.timedelta(days=window)
        series = fetch(lat, lon, elevation, start, end)
        for day in days:
            centre = pd.Timestamp(_shift_year(day, back))
            span = series.loc[centre - pd.Timedelta(days=window): centre + pd.Timedelta(days=window)]
            samples[day].extend(span.dropna().tolist())

    return {day.isoformat(): round(float(np.mean(v)), 2) for day, v in samples.items() if v}
