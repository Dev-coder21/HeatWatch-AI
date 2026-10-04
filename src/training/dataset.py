"""Clean, label and build features from the downloaded site data.

Outputs (data/processed/):
  observed_labeled.parquet  one row per site-day: lapse-corrected ERA5-Land Tmax,
                            normal, IMD label (+ the same without lapse correction).
  features.parquet          one row per site x issue day x lead (1-7): forecast and
                            lag features, target Tmax and IMD label on the target day.

Lags come from the GFS hindcast (= what the live forecast API returns as past days);
targets and labels come from ERA5-Land; normals from the ERA5-Land 1991-2020 grid.
No raw latitude/longitude features.

    python -m src.training.dataset
"""

import glob

import numpy as np
import pandas as pd

from src.config import project_path, settings
from src.heatwave_rules import classify, consecutive_days
from src.locations.lapse import openmeteo_to_cell, openmeteo_to_point
from src.locations.normals import evaluate, normal_coefficients

RAW_DIR = project_path("data/raw/sites")
OUT_DIR = project_path("data/processed")
SITES_FILE = project_path("config/training_sites.csv")
LEADS = range(1, 8)

TERRAINS = ["plains", "coastal", "hilly"]
FEATURES = [
    "fc_tmax",
    "lead",
    "tmax_lag1",
    "tmax_lag2",
    "tmax_lag3",
    "tmax_mean3",
    "tmax_mean7",
    "tmin_lag1",
    "rh_lag1",
    "wind_lag1",
    "precip_3d",
    "trend_3d",
    "normal_tmax",
    "elevation_m",
    "cell_elevation_offset_m",
    "distance_to_coast_km",
    "terrain_plains",
    "terrain_coastal",
    "terrain_hilly",
    "doy_sin",
    "doy_cos",
]
CLASSIFIER_FEATURES = FEATURES + ["fc_departure"]


def _read(site_id, prefix):
    files = sorted(glob.glob(str(RAW_DIR / site_id / f"{prefix}_*.csv")))
    if not files:
        return None
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def site_normals(site):
    """Lapse-corrected harmonic coefficients for the site, and the cell elevation."""
    info = normal_coefficients(site["latitude"], site["longitude"], point_elevation_m=site["elevation_m"])
    raw = normal_coefficients(site["latitude"], site["longitude"], point_elevation_m=None)
    return info, raw["coefs"]


def label_site(site, observed):
    info, raw_coefs = site_normals(site)
    cell_elev = info["cell_elevation_m"]
    doy = observed.index.dayofyear.values

    df = pd.DataFrame(index=observed.index)
    df["site_id"] = site["site_id"]
    df["temp_source"] = observed["temp_source"]
    # Open-Meteo values are already at the point's DEM height; apply our rate as a delta.
    df["tmax"] = openmeteo_to_point(observed["tmax"], site["elevation_m"], cell_elev)
    df["tmin"] = openmeteo_to_point(observed["tmin"], site["elevation_m"], cell_elev)
    df["normal_tmax"] = evaluate(info["coefs"], doy)
    labels = classify(df["tmax"], df["normal_tmax"], site["terrain_type"])
    df["departure"] = labels["departure"].values
    df["severity"] = labels["severity"].values
    df["heatwave"] = labels["heatwave"].values
    df["rule"] = labels["rule"].values
    df["consecutive_hw_days"] = consecutive_days(df["heatwave"].values)

    # Counterfactual without lapse correction (raw cell values and raw cell normals).
    tmax_cell = openmeteo_to_cell(observed["tmax"], site["elevation_m"], cell_elev)
    raw = classify(tmax_cell, evaluate(raw_coefs, doy), site["terrain_type"])
    df["tmax_cell"] = tmax_cell.values
    df["severity_no_lapse"] = raw["severity"].values
    df["cell_elevation_m"] = cell_elev
    return df


def build_site_features(site, labeled, hindcast, leads):
    """One row per issue day t and lead L; target day d = t + L."""
    h = hindcast.reindex(pd.date_range(hindcast.index.min(), hindcast.index.max()))
    lags = pd.DataFrame(index=h.index)
    lags["tmax_lag1"] = h["tmax"]
    lags["tmax_lag2"] = h["tmax"].shift(1)
    lags["tmax_lag3"] = h["tmax"].shift(2)
    lags["tmax_mean3"] = h["tmax"].rolling(3).mean()
    lags["tmax_mean7"] = h["tmax"].rolling(7).mean()
    lags["tmin_lag1"] = h["tmin"]
    lags["rh_lag1"] = h["rh_mean"]
    lags["wind_lag1"] = h["wind_max"]
    lags["precip_3d"] = h["precip"].rolling(3).sum()
    lags["trend_3d"] = h["tmax"] - h["tmax"].shift(2)

    cell_elev = labeled["cell_elevation_m"].iloc[0]
    terrain = site["terrain_type"]
    rows = []
    for lead in LEADS:
        target = pd.DataFrame(index=lags.index + pd.Timedelta(days=lead))
        target["issue_date"] = lags.index
        target[lags.columns] = lags.values
        target["lead"] = lead
        frame = target.join(leads[[f"fc_tmax_lead{lead}"]].rename(columns={f"fc_tmax_lead{lead}": "fc_tmax"}), how="inner")
        rows.append(frame)
    feats = pd.concat(rows)
    feats.index.name = "date"

    feats = feats.join(
        labeled[["tmax", "normal_tmax", "severity", "heatwave", "rule", "temp_source"]], how="inner"
    ).rename(columns={"tmax": "target_tmax"})
    feats["site_id"] = site["site_id"]
    feats["zone"] = site["zone"]
    feats["terrain_type"] = terrain
    feats["elevation_m"] = site["elevation_m"]
    feats["cell_elevation_offset_m"] = site["elevation_m"] - cell_elev
    feats["distance_to_coast_km"] = site["distance_to_coast_km"]
    for name in TERRAINS:
        feats[f"terrain_{name}"] = int(terrain == name)
    doy = feats.index.dayofyear.values
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats["fc_departure"] = feats["fc_tmax"] - feats["normal_tmax"]
    return feats.reset_index()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(SITES_FILE)
    model = settings()["api"]["forecast_model"]
    baselines = settings()["api"]["baseline_models"]

    labeled_all, feats_all, missing = [], [], []
    for site in sites.to_dict("records"):
        observed = _read(site["site_id"], "observed")
        hindcast = _read(site["site_id"], "hindcast")
        leads = _read(site["site_id"], f"leads_{model}")
        if observed is None or hindcast is None or leads is None:
            missing.append(site["site_id"])
            continue
        labeled = label_site(site, observed)
        labeled_all.append(labeled.reset_index())
        feats = build_site_features(site, labeled, hindcast, leads)

        # Raw baseline-model forecasts for the same target day and lead (NaN where absent).
        for base in baselines:
            other = _read(site["site_id"], f"leads_{base}")
            if other is None:
                continue
            long = other.melt(ignore_index=False, var_name="lead", value_name=f"raw_{base}")
            long["lead"] = long["lead"].str.replace("fc_tmax_lead", "").astype(int)
            long = long.reset_index()
            feats = feats.merge(long, on=["date", "lead"], how="left")
        feats_all.append(feats)

    labeled = pd.concat(labeled_all, ignore_index=True)
    features = pd.concat(feats_all, ignore_index=True)
    features = features.dropna(subset=FEATURES + ["target_tmax"])
    labeled.to_parquet(OUT_DIR / "observed_labeled.parquet", index=False)
    features.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"Sites: {labeled.site_id.nunique()} (missing: {missing or 'none'})")
    print(f"Labeled site-days: {len(labeled)}, heatwave: {labeled.heatwave.sum()}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()
