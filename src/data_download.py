import time
import requests
import pandas as pd
from pathlib import Path


# -----------------------------------
# Configuration
# -----------------------------------

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

URL = "https://archive-api.open-meteo.com/v1/archive"

LOCATIONS_FILE = Path("config/locations.csv")
OUTPUT_DIR = Path("data/raw")


# -----------------------------------
# Download one location
# -----------------------------------

def download_location(location):

    location_id = location["location_id"]
    city = location["city"]
    latitude = location["latitude"]
    longitude = location["longitude"]
    region_type = location["region_type"]

    print(f"Downloading {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
            "surface_pressure_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    max_retries = 3

    for attempt in range(max_retries):

        response = requests.get(
            URL,
            params=params,
            timeout=60
        )

        if response.status_code == 429:

            wait_time = 10 * (attempt + 1)

            print(
                f"Rate limit reached for {city}. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)
            continue

        response.raise_for_status()
        break

    else:

        raise RuntimeError(
            f"Could not download {city} after "
            f"{max_retries} attempts."
        )

    data = response.json()

    daily = data["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "location_id": location_id,
        "latitude": latitude,
        "longitude": longitude,
        "region_type": region_type,
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "relative_humidity": daily["relative_humidity_2m_mean"],
        "wind_speed": daily["wind_speed_10m_max"],
        "precipitation": daily["precipitation_sum"],
        "pressure": daily["surface_pressure_mean"],
    })

    return df


# -----------------------------------
# Main downloader
# -----------------------------------

def download_all_locations():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    locations = pd.read_csv(LOCATIONS_FILE)

    all_data = []

    for _, location in locations.iterrows():

        try:

            df = download_location(location)

            all_data.append(df)

            print(
                f"✓ {location['city']} downloaded "
                f"({len(df)} records)"
            )

            time.sleep(5)

        except Exception as error:

            print(
                f"✗ Failed to download "
                f"{location['city']}: {error}"
            )

    if not all_data:

        raise RuntimeError(
            "No weather data was downloaded."
        )

    combined_df = pd.concat(
        all_data,
        ignore_index=True
    )

    output_file = OUTPUT_DIR / "weather_raw.csv"

    combined_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("Download complete.")
    print(f"Total records: {len(combined_df)}")
    print(f"Saved to: {output_file}")


# -----------------------------------
# Run
# -----------------------------------

if __name__ == "__main__":
    download_all_locations()