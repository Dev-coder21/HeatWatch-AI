"""Build the India-wide 1991-2020 daily Tmax normals grid from ERA5-Land.

Downloads ERA5-Land daily maximum 2 m temperature from the Copernicus Climate
Data Store (dataset "derived-era5-land-daily-statistics") one month at a time,
then fits a harmonic day-of-year climatology per 0.1 degree land cell by
streaming least squares (the full 30-year cube never sits in memory).

CDS offers whole-hour time zones only, so days are cut at UTC+05:00 rather than
IST (UTC+05:30). Daily Tmax over India occurs around 14-15 IST, far from
midnight, so the 30-minute shift does not change the daily maximum in practice.

Usage:
    python -m src.locations.build_normals_grid            # download + fit
    python -m src.locations.build_normals_grid --fit-only
Requires ~/.cdsapirc (see README) and the dataset licence accepted on the CDS site.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.config import project_path, settings
from src.locations.normals import harmonic_design

DATASET = "derived-era5-land-daily-statistics"
# North, West, South, East: India plus the Andaman & Nicobar and Lakshadweep islands.
AREA = [38, 66, 5, 98.5]
RAW_DIR = project_path("data/external/era5land_tmax")
OROGRAPHY_FILE = project_path("data/external/era5land_geopotential.nc")


def month_file(year, month):
    return RAW_DIR / f"tmax_{year}_{month:02d}.nc"


def _download_month(client, year, month):
    target = month_file(year, month)
    if target.exists() and target.stat().st_size > 0:
        return
    request = {
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+05:00",
        "frequency": "1_hourly",
        "area": AREA,
    }
    for attempt in range(4):
        try:
            partial = target.with_suffix(".part")
            client.retrieve(DATASET, request, str(partial))
            partial.rename(target)
            print(f"Downloaded {year}-{month:02d}", flush=True)
            return
        except Exception as error:  # CDS raises plain Exceptions
            print(f"  {year}-{month:02d} failed ({error}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Could not download {year}-{month:02d}")


def download_orography():
    """ERA5-Land model surface height, needed for the lapse-rate correction."""
    if OROGRAPHY_FILE.exists():
        return
    import cdsapi

    cdsapi.Client().retrieve(
        "reanalysis-era5-land",
        {
            "variable": ["geopotential"],
            "year": "2024", "month": "01", "day": ["01"], "time": ["00:00"],
            "data_format": "netcdf", "download_format": "unarchived",
            "area": AREA,
        },
        str(OROGRAPHY_FILE),
    )


def download(years, workers=4):
    import cdsapi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_orography()
    client = cdsapi.Client(quiet=True)
    jobs = [(year, month) for year in years for month in range(1, 13)]
    # CDS queues a few concurrent requests per user; more just wait in line.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(_download_month, client, y, m) for y, m in jobs]:
            future.result()


def _cell_elevation(lat, lon):
    import xarray as xr

    z = xr.open_dataset(OROGRAPHY_FILE)["z"].squeeze() / 9.80665
    z = z.sel(latitude=xr.DataArray(lat), longitude=xr.DataArray(lon), method="nearest")
    return z.values


def _open_month(path):
    import xarray as xr

    ds = xr.open_dataset(path)
    var = [v for v in ds.data_vars if ds[v].ndim == 3][0]
    da = ds[var]
    time_dim = [d for d in da.dims if d not in ("latitude", "longitude")][0]
    da = da.rename({time_dim: "time"}).transpose("time", "latitude", "longitude")
    values = da.values.astype(np.float64)
    if np.nanmean(values) > 150:  # Kelvin
        values -= 273.15
    doy = da["time"].dt.dayofyear.values
    return values, doy, da["latitude"].values, da["longitude"].values


def fit(years, harmonics):
    """Streaming least squares: accumulate X'X and X'y per cell."""
    xtx = None
    xty = None
    count = None
    lat = lon = None
    total_days = 0

    for year in years:
        for month in range(1, 13):
            values, doy, lat_m, lon_m = _open_month(month_file(year, month))
            if xtx is None:
                lat, lon = lat_m, lon_m
                cells = len(lat) * len(lon)
                params = 2 * harmonics + 1
                xtx = np.zeros((params, params))
                xty = np.zeros((params, cells))
                count = np.zeros(cells, dtype=np.int64)
            design = harmonic_design(doy, harmonics)
            flat = values.reshape(len(doy), -1)
            valid = np.isfinite(flat)
            xtx += design.T @ design
            xty += design.T @ np.where(valid, flat, 0.0)
            count += valid.sum(axis=0)
            total_days += len(doy)
        print(f"  fitted through {year}", flush=True)

    # ERA5-Land's land mask is fixed, so land cells are valid every day.
    land = count == total_days
    coefs = np.linalg.solve(xtx, xty[:, land]).T

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    cell_lat = lat_grid.ravel()[land]
    cell_lon = lon_grid.ravel()[land]
    out = project_path(settings()["normals"]["grid_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        lat=cell_lat.astype(np.float32),
        lon=cell_lon.astype(np.float32),
        coefs=coefs.astype(np.float32),
        # Model surface height of each cell: normals are valid at this height.
        elevation=_cell_elevation(cell_lat, cell_lon).astype(np.float32),
        source=np.array(
            f"ERA5-Land daily max 2m temperature {years[0]}-{years[-1]}, "
            f"{harmonics}-harmonic fit, day boundary UTC+05:00 (CDS {DATASET})"
        ),
    )
    print(f"Saved {land.sum()} land cells to {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    cfg = settings()["normals"]
    years = list(range(cfg["period"][0], cfg["period"][1] + 1))
    if not args.fit_only:
        download(years)
    fit(years, cfg["harmonics"])


if __name__ == "__main__":
    main()
