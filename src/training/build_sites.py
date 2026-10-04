"""Geocode the hand-picked training sites and compute their terrain type.

Input:  config/training_sites_input.txt  (zone|Name,State[|lat,lon] per line;
        the optional coordinates pin a place the geocoder confuses with a namesake)
Output: config/training_sites.csv
"""

import re
import time

import pandas as pd

from src.config import project_path
from src.locations.geo import distance_to_coast_km
from src.locations.geocode import geocode
from src.locations.profile import fetch_elevation
from src.locations.terrain import terrain_type

INPUT = project_path("config/training_sites_input.txt")
OUTPUT = project_path("config/training_sites.csv")

# Names GeoNames knows under an older or local spelling.
ALIASES = {
    "Prayagraj": "Allahabad",
    "Kalaburagi": "Gulbarga",
    "Daltonganj": "Medininagar",
    "Ooty": "Udhagamandalam",
    "Mangaluru": "Mangalore",
    "Panaji": "Panjim",
    "Dharamshala": "Dharamsala",
}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _same_state(found, wanted):
    if not found:
        return False
    found, wanted = found.lower(), wanted.lower()
    return found in wanted or wanted in found


def locate(name, state):
    for query in (name, ALIASES.get(name)):
        if not query:
            continue
        candidates = [c for c in geocode(query, count=20) if _same_state(c["state"], state)]
        if candidates:
            return max(candidates, key=lambda c: c.get("population") or 0)
    raise LookupError(f"{name}, {state} not found")


def main():
    rows = []
    missing = []
    for line in INPUT.read_text().splitlines():
        if not line.strip():
            continue
        zone, place, *pinned = line.split("|")
        name, state = place.split(",", 1)
        if pinned:
            lat, lon = (float(v) for v in pinned[0].split(","))
            hit = {"lat": lat, "lon": lon}
        else:
            try:
                hit = locate(name, state)
            except LookupError as error:
                missing.append(str(error))
                continue
        elevation = fetch_elevation(hit["lat"], hit["lon"])
        coast_km = distance_to_coast_km(hit["lat"], hit["lon"])
        rows.append(
            {
                "site_id": slugify(name),
                "name": name,
                "state": state,
                "zone": zone,
                "latitude": hit["lat"],
                "longitude": hit["lon"],
                "elevation_m": elevation,
                "distance_to_coast_km": round(coast_km, 1),
                "terrain_type": terrain_type(hit["lat"], hit["lon"], elevation, coast_km=coast_km),
            }
        )
        time.sleep(0.2)
    if missing:
        raise SystemExit("Not found: " + "; ".join(missing))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print(df.groupby(["zone", "terrain_type"]).size().unstack(fill_value=0))
    print(df.terrain_type.value_counts())


if __name__ == "__main__":
    main()
