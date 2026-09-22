import pandas as pd
from pathlib import Path


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = Path("data/raw/weather_raw.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "weather_clean.csv"


# -----------------------------------
# Load data
# -----------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    return df


# -----------------------------------
# Clean and validate data
# -----------------------------------

def clean_data(df):

    print("Starting data cleaning...")

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Remove rows with invalid dates
    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(
            f"Removing {invalid_dates} rows "
            f"with invalid dates."
        )

        df = df.dropna(
            subset=["date"]
        )

    # Check numeric columns
    numeric_columns = [
        "latitude",
        "longitude",
        "temperature_max",
        "temperature_min",
        "relative_humidity",
        "wind_speed",
        "precipitation",
        "pressure",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # -----------------------------------
    # Validate latitude
    # -----------------------------------

    invalid_latitude = (
        (df["latitude"] < -90)
        | (df["latitude"] > 90)
    )

    print(
        f"Invalid latitude values: "
        f"{invalid_latitude.sum()}"
    )

    df.loc[invalid_latitude, "latitude"] = pd.NA

    # -----------------------------------
    # Validate longitude
    # -----------------------------------

    invalid_longitude = (
        (df["longitude"] < -180)
        | (df["longitude"] > 180)
    )

    print(
        f"Invalid longitude values: "
        f"{invalid_longitude.sum()}"
    )

    df.loc[
        invalid_longitude,
        "longitude"
    ] = pd.NA

    # -----------------------------------
    # Validate humidity
    # -----------------------------------

    invalid_humidity = (
        (df["relative_humidity"] < 0)
        | (df["relative_humidity"] > 100)
    )

    print(
        f"Invalid humidity values: "
        f"{invalid_humidity.sum()}"
    )

    df.loc[
        invalid_humidity,
        "relative_humidity"
    ] = pd.NA

    # -----------------------------------
    # Validate precipitation
    # -----------------------------------

    invalid_precipitation = (
        df["precipitation"] < 0
    )

    print(
        f"Invalid precipitation values: "
        f"{invalid_precipitation.sum()}"
    )

    df.loc[
        invalid_precipitation,
        "precipitation"
    ] = pd.NA

    # -----------------------------------
    # Validate temperature
    # -----------------------------------

    invalid_temperature = (
        (df["temperature_max"] < -50)
        | (df["temperature_max"] > 60)
        | (df["temperature_min"] < -60)
        | (df["temperature_min"] > 50)
    )

    print(
        f"Suspicious temperature values: "
        f"{invalid_temperature.sum()}"
    )

    # We flag suspicious temperatures
    # instead of silently deleting them.

    df["temperature_flag"] = invalid_temperature

    # -----------------------------------
    # Duplicate check
    # -----------------------------------

    duplicate_mask = df.duplicated(
        subset=["location_id", "date"],
        keep="first"
    )

    duplicate_count = duplicate_mask.sum()

    print(
        f"Duplicate location-date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:
        df = df[
            ~duplicate_mask
        ]

    # -----------------------------------
    # Missing-value report
    # -----------------------------------

    print("\nMissing values by column:")

    missing_values = df.isna().sum()

    print(missing_values)

    # -----------------------------------
    # Sort data
    # -----------------------------------

    df = df.sort_values(
        by=["location_id", "date"]
    ).reset_index(drop=True)

    return df


# -----------------------------------
# Save cleaned data
# -----------------------------------

def save_data(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Cleaning complete.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


# -----------------------------------
# Main
# -----------------------------------

def main():

    df = load_data()

    print(
        f"Loaded {len(df)} raw records."
    )

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":
    main()