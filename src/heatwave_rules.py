import pandas as pd
import yaml
from pathlib import Path


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = Path("data/processed/weather_clean.csv")
OUTPUT_FILE = Path("data/processed/weather_labeled.csv")
THRESHOLDS_FILE = Path("config/thresholds.yaml")

# Training period used to calculate
# climatological normal temperatures.
TRAINING_END_DATE = "2022-12-31"


# -----------------------------------
# Load thresholds
# -----------------------------------

def load_thresholds():

    with open(THRESHOLDS_FILE, "r") as file:
        thresholds = yaml.safe_load(file)

    return thresholds["heatwave"]


# -----------------------------------
# Calculate monthly normal temperature
# -----------------------------------

def calculate_monthly_normals(df):

    training_data = df[
        df["date"] <= TRAINING_END_DATE
    ].copy()

    training_data["month"] = (
        training_data["date"].dt.month
    )

    monthly_normals = (
        training_data
        .groupby(
            ["location_id", "month"]
        )["temperature_max"]
        .mean()
        .reset_index()
    )

    monthly_normals = monthly_normals.rename(
        columns={
            "temperature_max": "normal_max_temp"
        }
    )

    return monthly_normals


# -----------------------------------
# Apply heatwave rules
# -----------------------------------

def apply_heatwave_rules(df, thresholds):

    # Add month
    df["month"] = df["date"].dt.month

    # Calculate monthly normal
    monthly_normals = calculate_monthly_normals(df)

    # Merge normal temperature
    df = df.merge(
        monthly_normals,
        on=["location_id", "month"],
        how="left"
    )

    # Calculate departure from normal
    df["departure"] = (
        df["temperature_max"]
        - df["normal_max_temp"]
    )

    # Default values
    df["heatwave"] = 0
    df["severity"] = "No Heatwave"

    # -----------------------------------
    # Plains
    # -----------------------------------

    plains = df["region_type"] == "plains"

    plains_condition = (
        plains
        & (
            df["temperature_max"]
            >= thresholds["plains"][
                "temperature_threshold_c"
            ]
        )
        & (
            df["departure"]
            >= thresholds["plains"][
                "departure_threshold_c"
            ]
        )
    )

    df.loc[
        plains_condition,
        "heatwave"
    ] = 1

    df.loc[
        plains_condition,
        "severity"
    ] = "Heatwave"

    plains_severe = (
        plains_condition
        & (
            df["departure"]
            >= thresholds["plains"][
                "severe_departure_threshold_c"
            ]
        )
    )

    df.loc[
        plains_severe,
        "severity"
    ] = "Severe Heatwave"

    # -----------------------------------
    # Hilly
    # -----------------------------------

    hilly = df["region_type"] == "hilly"

    hilly_condition = (
        hilly
        & (
            df["temperature_max"]
            >= thresholds["hilly"][
                "temperature_threshold_c"
            ]
        )
        & (
            df["departure"]
            >= thresholds["hilly"][
                "departure_threshold_c"
            ]
        )
    )

    df.loc[
        hilly_condition,
        "heatwave"
    ] = 1

    df.loc[
        hilly_condition,
        "severity"
    ] = "Heatwave"

    hilly_severe = (
        hilly_condition
        & (
            df["departure"]
            >= thresholds["hilly"][
                "severe_departure_threshold_c"
            ]
        )
    )

    df.loc[
        hilly_severe,
        "severity"
    ] = "Severe Heatwave"

    # -----------------------------------
    # Coastal
    # -----------------------------------

    coastal = df["region_type"] == "coastal"

    coastal_condition = (
        coastal
        & (
            df["temperature_max"]
            >= thresholds["coastal"][
                "temperature_threshold_c"
            ]
        )
        & (
            df["departure"]
            >= thresholds["coastal"][
                "departure_threshold_c"
            ]
        )
    )

    df.loc[
        coastal_condition,
        "heatwave"
    ] = 1

    df.loc[
        coastal_condition,
        "severity"
    ] = "Heatwave"

    coastal_severe = (
        coastal_condition
        & (
            df["departure"]
            >= thresholds["coastal"][
                "severe_departure_threshold_c"
            ]
        )
    )

    df.loc[
        coastal_severe,
        "severity"
    ] = "Severe Heatwave"

    return df


# -----------------------------------
# Consecutive heatwave days
# -----------------------------------

def calculate_persistence(df):

    df = df.sort_values(
        ["location_id", "date"]
    ).reset_index(drop=True)

    df["heatwave_group"] = (
        df.groupby("location_id")["heatwave"]
        .transform(
            lambda x: x.ne(1).cumsum()
        )
    )

    df["consecutive_heatwave_days"] = (
        df.groupby(
            ["location_id", "heatwave_group"]
        )["heatwave"]
        .transform("sum")
    )

    df.loc[
        df["heatwave"] == 0,
        "consecutive_heatwave_days"
    ] = 0

    df = df.drop(
        columns=["heatwave_group"]
    )

    return df


# -----------------------------------
# Main
# -----------------------------------

def main():

    print("Loading cleaned weather data...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print(
        f"Loaded {len(df)} records."
    )

    thresholds = load_thresholds()

    df = apply_heatwave_rules(
        df,
        thresholds
    )

    df = calculate_persistence(df)

    # Save labeled dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------
    # Summary
    # -----------------------------------

    heatwave_days = (
        df["heatwave"].sum()
    )

    severe_days = (
        (df["severity"] == "Severe Heatwave")
        .sum()
    )

    print()
    print("Heatwave labeling complete.")
    print(
        f"Total records: {len(df)}"
    )
    print(
        f"Heatwave records: {heatwave_days}"
    )
    print(
        f"Severe heatwave records: {severe_days}"
    )
    print()
    print("Severity distribution:")
    print(
        df["severity"].value_counts()
    )
    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()