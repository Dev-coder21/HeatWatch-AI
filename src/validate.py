from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecast_results.csv"
)


# -----------------------------------
# Required prediction schema
# -----------------------------------

REQUIRED_COLUMNS = [
    "forecast_date",
    "location_id",
    "city",
    "state",
    "latitude",
    "longitude",
    "region_type",
    "date",
    "temperature_max",
    "predicted_tmax",
    "heatwave_probability",
    "severity",
    "departure",
    "consecutive_heatwave_days",
    "humidity_lag_1",
    "temperature_component",
    "heatwave_probability_component",
    "anomaly_component",
    "persistence_component",
    "humidity_component",
    "risk_score",
    "risk_category",
    "top_factor_1",
    "top_factor_2",
    "top_factor_3",
    "advisory",
]


# -----------------------------------
# Valid values
# -----------------------------------

VALID_RISK_CATEGORIES = {
    "Low",
    "Moderate",
    "High",
    "Very High",
    "Extreme",
}

VALID_SEVERITIES = {
    "No Heatwave",
    "Heatwave",
    "Severe Heatwave",
}


# -----------------------------------
# Main validation
# -----------------------------------

def main():

    print(
        "Validating prediction results..."
    )
    print()

    # -----------------------------------
    # Check prediction file
    # -----------------------------------

    if not PREDICTION_PATH.exists():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTION_PATH}"
        )

    df = pd.read_csv(
        PREDICTION_PATH
    )

    # -----------------------------------
    # Check required columns
    # -----------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print(
        "✓ Required columns present"
    )

    # -----------------------------------
    # Check data is not empty
    # -----------------------------------

    if df.empty:

        raise ValueError(
            "Prediction output is empty."
        )

    print(
        f"✓ Prediction rows found: "
        f"{len(df)}"
    )

    # -----------------------------------
    # Check locations
    # -----------------------------------

    location_count = df[
        "location_id"
    ].nunique()

    if location_count == 0:

        raise ValueError(
            "No locations found."
        )

    print(
        f"✓ Locations found: "
        f"{location_count}"
    )

    # -----------------------------------
    # Check duplicate locations
    # -----------------------------------

    duplicate_locations = (
        df["location_id"]
        .duplicated()
        .sum()
    )

    if duplicate_locations > 0:

        raise ValueError(
            "Duplicate location predictions found: "
            f"{duplicate_locations}"
        )

    print(
        "✓ One prediction found per location"
    )

    # -----------------------------------
    # Check dates
    # -----------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    if (
        df["date"].isna().any()
        or df["forecast_date"].isna().any()
    ):

        raise ValueError(
            "Invalid dates found."
        )

    invalid_forecast_dates = (
        df["forecast_date"]
        <= df["date"]
    )

    if invalid_forecast_dates.any():

        raise ValueError(
            "Forecast date must be after "
            "the input date."
        )

    print(
        "✓ Forecast dates are valid"
    )

    # -----------------------------------
    # Check predicted temperature
    # -----------------------------------

    if (
        df["predicted_tmax"]
        .isna()
        .any()
    ):

        raise ValueError(
            "Missing predicted temperatures."
        )

    print(
        "✓ Predicted temperatures are valid"
    )

    # -----------------------------------
    # Check risk score range
    # -----------------------------------

    if (
        df["risk_score"].min() < 0
        or df["risk_score"].max() > 100
    ):

        raise ValueError(
            "Risk score outside 0-100 range."
        )

    print(
        "✓ Risk scores are within 0-100"
    )

    # -----------------------------------
    # Check probability range
    # -----------------------------------

    if (
        df["heatwave_probability"].min() < 0
        or df["heatwave_probability"].max() > 1
    ):

        raise ValueError(
            "Heatwave probability outside 0-1."
        )

    print(
        "✓ Heatwave probabilities are within 0-1"
    )

    # -----------------------------------
    # Check severity values
    # -----------------------------------

    actual_severities = set(
        df["severity"].dropna()
    )

    invalid_severities = (
        actual_severities
        - VALID_SEVERITIES
    )

    if invalid_severities:

        raise ValueError(
            "Invalid severity values: "
            + ", ".join(
                sorted(
                    invalid_severities
                )
            )
        )

    print(
        "✓ Severity values are valid"
    )

    # -----------------------------------
    # Check risk categories
    # -----------------------------------

    actual_categories = set(
        df["risk_category"].dropna()
    )

    invalid_categories = (
        actual_categories
        - VALID_RISK_CATEGORIES
    )

    if invalid_categories:

        raise ValueError(
            "Invalid risk categories: "
            + ", ".join(
                sorted(
                    invalid_categories
                )
            )
        )

    print(
        "✓ Risk categories are valid"
    )

    # -----------------------------------
    # Check missing values
    # -----------------------------------

    missing_values = (
        df[REQUIRED_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:

        raise ValueError(
            f"Found {missing_values} "
            "missing values."
        )

    print(
        "✓ No missing values"
    )

    # -----------------------------------
    # Check duplicate rows
    # -----------------------------------

    duplicate_rows = (
        df.duplicated().sum()
    )

    if duplicate_rows > 0:

        raise ValueError(
            f"Found {duplicate_rows} "
            "duplicate rows."
        )

    print(
        "✓ No duplicate rows"
    )

    # -----------------------------------
    # Check advisory fields
    # -----------------------------------

    for column in [
        "top_factor_1",
        "top_factor_2",
        "top_factor_3",
        "advisory",
    ]:

        if (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):

            raise ValueError(
                f"Empty values found in "
                f"{column}."
            )

    print(
        "✓ Risk factors and advisories present"
    )

    # -----------------------------------
    # Final summary
    # -----------------------------------

    print()
    print(
        "Prediction validation PASSED."
    )

    print()
    print(
        "Risk distribution:"
    )

    print(
        df["risk_category"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Severity distribution:"
    )

    print(
        df["severity"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Highest-risk locations:"
    )

    print(
        df[
            [
                "city",
                "predicted_tmax",
                "heatwave_probability",
                "severity",
                "risk_score",
                "risk_category",
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()