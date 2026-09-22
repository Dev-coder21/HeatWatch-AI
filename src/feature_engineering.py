import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = Path(
    "data/processed/weather_labeled.csv"
)

OUTPUT_FILE = Path(
    "data/processed/features.csv"
)


# -----------------------------------
# Create features
# -----------------------------------

def create_features(df):

    # Make sure data is correctly ordered
    df = df.sort_values(
        ["location_id", "date"]
    ).reset_index(drop=True)

    # -----------------------------------
    # Calendar features
    # -----------------------------------

    df["month"] = df["date"].dt.month

    df["day_of_year"] = (
        df["date"].dt.dayofyear
    )

    # -----------------------------------
    # Temperature lag features
    # -----------------------------------

    grouped = df.groupby(
        "location_id",
        group_keys=False
    )

    df["temp_max_lag_1"] = (
        grouped["temperature_max"]
        .shift(1)
    )

    df["temp_max_lag_2"] = (
        grouped["temperature_max"]
        .shift(2)
    )

    df["temp_max_lag_3"] = (
        grouped["temperature_max"]
        .shift(3)
    )

    # -----------------------------------
    # Rolling temperature features
    # -----------------------------------
    # Shift first so today's value is NOT
    # included in the rolling calculation.

    df["temp_max_rolling_3"] = (
        grouped["temperature_max"]
        .shift(1)
        .groupby(df["location_id"])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["temp_max_rolling_7"] = (
        grouped["temperature_max"]
        .shift(1)
        .groupby(df["location_id"])
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------------
    # Minimum temperature
    # -----------------------------------

    df["temp_min_lag_1"] = (
        grouped["temperature_min"]
        .shift(1)
    )

    # -----------------------------------
    # Humidity
    # -----------------------------------

    df["humidity_lag_1"] = (
        grouped["relative_humidity"]
        .shift(1)
    )

    # -----------------------------------
    # Wind
    # -----------------------------------

    df["wind_lag_1"] = (
        grouped["wind_speed"]
        .shift(1)
    )

    # -----------------------------------
    # Recent precipitation
    # -----------------------------------

    df["precip_3d"] = (
        grouped["precipitation"]
        .shift(1)
        .groupby(df["location_id"])
        .rolling(3)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------------
    # Temperature trend
    # -----------------------------------

    df["temp_trend_3d"] = (
        df["temp_max_lag_1"]
        - df["temp_max_lag_3"]
    )

    # -----------------------------------
    # Temperature anomaly
    # -----------------------------------

    # departure was already calculated
    # during heatwave labeling.

    # -----------------------------------
    # Target: next-day maximum temperature
    # -----------------------------------

    df["target_temperature_max"] = (
        grouped["temperature_max"]
        .shift(-1)
    )

    # -----------------------------------
    # Target: next-day heatwave
    # -----------------------------------

    df["target_heatwave"] = (
        grouped["heatwave"]
        .shift(-1)
    )

    # -----------------------------------
    # Target: next-day severity
    # -----------------------------------

    df["target_severity"] = (
        grouped["severity"]
        .shift(-1)
    )

    return df


# -----------------------------------
# Remove rows that cannot be used
# -----------------------------------

def remove_invalid_rows(df):

    feature_columns = [
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
        "target_temperature_max",
        "target_heatwave",
        "target_severity",
    ]

    before = len(df)

    df = df.dropna(
        subset=feature_columns
    ).reset_index(drop=True)

    after = len(df)

    print(
        f"Removed {before - after} rows "
        f"that could not be used for training."
    )

    return df


# -----------------------------------
# Main
# -----------------------------------

def main():

    print("Loading labeled weather data...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Loaded {len(df)} records."
    )

    print("Creating time-series features...")

    df = create_features(df)

    print("Removing incomplete feature rows...")

    df = remove_invalid_rows(df)

    # -----------------------------------
    # Save
    # -----------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Feature engineering complete.")
    print(
        f"Final records: {len(df)}"
    )
    print(
        f"Final columns: {len(df.columns)}"
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()