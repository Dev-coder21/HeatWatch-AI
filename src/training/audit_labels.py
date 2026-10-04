"""Label audit for the Phase 3 checkpoint (no fixes applied).

  a) heatwave / severe counts per site and year, split by IMD rule (absolute vs departure)
  b) known heat events: lapse-corrected ERA5-Land Tmax vs reported station maxima
  c) effect of the lapse-rate correction on labels (hill stations in particular)

    python -m src.training.audit_labels
Writes outputs/metrics/label_audit.json and outputs/metrics/label_counts.csv.
"""

import json

import pandas as pd

from src.config import project_path
from src.training.dataset import label_site
from src.training.download import fetch_observed

LABELED = project_path("data/processed/observed_labeled.parquet")
EVENTS = project_path("config/audit_events.csv")
OUT = project_path("outputs/metrics")


def main():
    df = pd.read_parquet(LABELED)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    sites = pd.read_csv(project_path("config/training_sites.csv")).set_index("site_id")
    df["terrain_type"] = df["site_id"].map(sites["terrain_type"])

    hw = df[df["heatwave"] == 1]
    counts = (
        hw.assign(severe=(hw["severity"] == 2).astype(int), absolute=(hw["rule"] == "absolute").astype(int))
        .groupby(["site_id", "year"])
        .agg(heatwave_days=("heatwave", "sum"), severe_days=("severe", "sum"), absolute_rule_days=("absolute", "sum"))
        .reset_index()
    )
    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "label_counts.csv", index=False)

    summary = {
        "site_days": int(len(df)),
        "heatwave_days": int(df["heatwave"].sum()),
        "severe_days": int((df["severity"] == 2).sum()),
        "by_rule": hw["rule"].value_counts().to_dict(),
        "by_terrain": df.groupby("terrain_type")["heatwave"].agg(["sum", "count"]).astype(int).to_dict("index"),
        "by_year": df.groupby("year")["heatwave"].sum().astype(int).to_dict(),
        "sites_with_zero_heatwave_days": sorted(set(sites.index) - set(hw["site_id"])),
    }

    # Lapse correction effect.
    changed = df[df["severity"] != df["severity_no_lapse"]]
    lapse = {
        "changed_site_days": int(len(changed)),
        "gained": int((changed["severity"] > changed["severity_no_lapse"]).sum()),
        "lost": int((changed["severity"] < changed["severity_no_lapse"]).sum()),
        "by_terrain": changed.groupby("terrain_type").size().astype(int).to_dict(),
        "hill_stations": {
            site: {
                "cell_offset_m": round(float(sites.loc[site, "elevation_m"] - g["cell_elevation_m"].iloc[0]), 0),
                "with_lapse": int((g["severity"] > 0).sum()),
                "without_lapse": int((g["severity_no_lapse"] > 0).sum()),
            }
            for site, g in df[df["terrain_type"] == "hilly"].groupby("site_id")
        },
    }

    events = []
    for event in pd.read_csv(EVENTS).to_dict("records"):
        window = df[(df["site_id"] == event["site_id"]) & df["date"].between(event["start"], event["end"])]
        if window.empty:
            # Outside the 2021+ training download: fetch just this window.
            site = sites.loc[event["site_id"]].to_dict() | {"site_id": event["site_id"]}
            observed = fetch_observed(site["latitude"], site["longitude"], event["start"], event["end"])
            observed["date"] = pd.to_datetime(observed["date"])
            window = label_site(site, observed.set_index("date")).reset_index()
        peak = window.loc[window["tmax"].idxmax()]
        on_day = window[window["date"] == event["reported_date"]]
        events.append(
            {
                "event": event["event"],
                "reported_c": event["reported_station_tmax_c"],
                "era5land_peak_c": round(float(peak["tmax"]), 1),
                "era5land_peak_date": str(peak["date"].date()),
                "era5land_on_reported_day_c": round(float(on_day["tmax"].iloc[0]), 1) if len(on_day) else None,
                "shortfall_c": round(event["reported_station_tmax_c"] - float(peak["tmax"]), 1),
                "heatwave_days_in_window": int(window["heatwave"].sum()),
                "severe_days_in_window": int((window["severity"] == 2).sum()),
                "max_departure_c": round(float(window["departure"].max()), 1),
                "note": event["note"],
            }
        )

    report = {"labels": summary, "lapse_correction": lapse, "events": events}
    (OUT / "label_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
