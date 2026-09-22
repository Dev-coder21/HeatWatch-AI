from pathlib import Path

import joblib
import pandas as pd

from risk_score import (
    calculate_risk_score,
    risk_category,
    load_thresholds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

LOCATIONS_PATH = (
    PROJECT_ROOT
    / "config"
    / "locations.csv"
)

TEMPERATURE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_model.joblib"
)

TEMPERATURE_FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_features.joblib"
)

HEATWAVE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "heatwave_model.joblib"
)

SEVERITY_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "severity_model.joblib"
)

CLASSIFICATION_FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification_features.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecast_results.csv"
)


# -----------------------------------
# Load models
# -----------------------------------

def load_models():
    """Load trained models and feature lists."""

    temperature_model = joblib.load(
        TEMPERATURE_MODEL_PATH
    )

    temperature_features = joblib.load(
        TEMPERATURE_FEATURES_PATH
    )

    heatwave_model = joblib.load(
        HEATWAVE_MODEL_PATH
    )

    severity_model = joblib.load(
        SEVERITY_MODEL_PATH
    )

    classification_features = joblib.load(
        CLASSIFICATION_FEATURES_PATH
    )

    return (
        temperature_model,
        temperature_features,
        heatwave_model,
        severity_model,
        classification_features,
    )


# -----------------------------------
# Severity label formatting
# -----------------------------------

def format_severity(value):
    """
    Convert the trained severity model output
    into a readable label without inventing
    additional severity classes.
    """

    if pd.isna(value):
        return "Unknown"

    value_string = str(value).strip()

    if value_string.lower() in {
        "0",
        "no heatwave",
        "no_heatwave",
        "none",
    }:
        return "No Heatwave"

    if value_string.lower() in {
        "1",
        "heatwave",
    }:
        return "Heatwave"

    if value_string.lower() in {
        "2",
        "severe heatwave",
        "severe_heatwave",
    }:
        return "Severe Heatwave"

    return value_string


# -----------------------------------
# Advisory generation
# -----------------------------------

def generate_advisory(
    risk_category_value,
    severity,
    predicted_temperature,
    heatwave_probability,
    humidity,
):
    """
    Generate a rule-based advisory.

    The advisory explains the model/risk output.
    It does not calculate or modify the numerical
    risk score.
    """

    category = str(
        risk_category_value
    ).lower()

    severity_text = str(
        severity
    ).lower()

    probability = float(
        heatwave_probability
    )

    temperature = float(
        predicted_temperature
    )

    humidity_value = (
        float(humidity)
        if pd.notna(humidity)
        else None
    )

    if (
        category == "extreme"
        or "severe heatwave" in severity_text
    ):
        advisory = (
            "Extreme heat conditions are indicated. "
            "Avoid prolonged outdoor exposure, stay "
            "hydrated, use cooling measures, and follow "
            "local official heat-health guidance."
        )

    elif (
        category == "very high"
        or severity_text == "heatwave"
        or probability >= 0.50
    ):
        advisory = (
            "High heat risk is indicated. Reduce "
            "prolonged outdoor activity, stay hydrated, "
            "seek shade or cooling, and monitor local "
            "heat-health guidance."
        )

    elif category == "high":
        advisory = (
            "Elevated heat risk is indicated. Stay "
            "hydrated, limit strenuous outdoor activity, "
            "and take regular breaks in shade or cool areas."
        )

    elif category == "moderate":
        advisory = (
            "Moderate heat risk is indicated. Stay "
            "hydrated and take sensible precautions "
            "during prolonged outdoor activity."
        )

    else:
        advisory = (
            "Current heat risk is low. Maintain normal "
            "hydration and routine heat-safety precautions."
        )

    # Add context where useful without changing the score.
    context_parts = []

    if temperature >= 40:
        context_parts.append(
            f"Forecast maximum temperature is "
            f"{temperature:.1f}°C."
        )

    if probability > 0:
        context_parts.append(
            f"Estimated heatwave probability is "
            f"{probability * 100:.1f}%."
        )

    if (
        humidity_value is not None
        and humidity_value >= 70
    ):
        context_parts.append(
            "High humidity may increase heat discomfort."
        )

    if context_parts:
        advisory += " " + " ".join(
            context_parts
        )

    return advisory


# -----------------------------------
# Top risk factors
# -----------------------------------

def get_top_factors(
    risk_result,
    predicted_temperature,
    heatwave_probability,
    departure,
    consecutive_heatwave_days,
    humidity,
):
    """
    Rank the deterministic risk-engine components.

    These are risk-engine contributors, not SHAP values
    and not causal explanations.
    """

    component_map = {
        "Temperature": float(
            risk_result.get(
                "temperature_component",
                0,
            )
        ),
        "Heatwave Probability": float(
            risk_result.get(
                "heatwave_probability_component",
                0,
            )
        ),
        "Temperature Anomaly": float(
            risk_result.get(
                "anomaly_component",
                0,
            )
        ),
        "Heatwave Persistence": float(
            risk_result.get(
                "persistence_component",
                0,
            )
        ),
        "Humidity": float(
            risk_result.get(
                "humidity_component",
                0,
            )
        ),
    }

    ranked = sorted(
        component_map.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        ranked[0][0],
        ranked[1][0],
        ranked[2][0],
    ]


# -----------------------------------
# Main
# -----------------------------------

def main():

    print("Loading data and models...")

    # -----------------------------------
    # Load feature data
    # -----------------------------------

    df = pd.read_csv(
        FEATURES_PATH
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # -----------------------------------
    # Load location metadata
    # -----------------------------------

    locations = pd.read_csv(
        LOCATIONS_PATH
    )

    location_columns = [
        "location_id",
        "city",
        "state",
        "latitude",
        "longitude",
        "region_type",
    ]

    locations = locations[
        location_columns
    ]

    locations = locations.drop_duplicates(
        subset=["location_id"]
    )

    # -----------------------------------
    # Merge metadata
    # -----------------------------------

    df = df.merge(
        locations,
        on="location_id",
        how="left",
        suffixes=("", "_metadata"),
    )

    # -----------------------------------
    # Load models
    # -----------------------------------

    (
        temperature_model,
        temperature_features,
        heatwave_model,
        severity_model,
        classification_features,
    ) = load_models()

    # -----------------------------------
    # Load risk configuration
    # -----------------------------------

    thresholds = load_thresholds()

    # -----------------------------------
    # Latest available row per location
    # -----------------------------------

    latest_rows = (
        df.sort_values(
            ["location_id", "date"]
        )
        .groupby("location_id")
        .tail(1)
        .copy()
    )

    print(
        f"Generating predictions for "
        f"{len(latest_rows)} locations..."
    )

    # -----------------------------------
    # Temperature prediction
    # -----------------------------------

    X_temperature = latest_rows[
        temperature_features
    ]

    predicted_temperature = (
        temperature_model.predict(
            X_temperature
        )
    )

    latest_rows[
        "predicted_temperature_max"
    ] = predicted_temperature

    # -----------------------------------
    # Heatwave probability
    # -----------------------------------

    X_heatwave = latest_rows[
        classification_features
    ]

    heatwave_probability = (
        heatwave_model.predict_proba(
            X_heatwave
        )[:, 1]
    )

    latest_rows[
        "heatwave_probability"
    ] = heatwave_probability

    # -----------------------------------
    # Severity prediction
    # -----------------------------------

    severity_prediction = (
        severity_model.predict(
            X_heatwave
        )
    )

    latest_rows[
        "severity"
    ] = [
        format_severity(value)
        for value in severity_prediction
    ]

    # -----------------------------------
    # Calculate risk scores
    # -----------------------------------

    risk_results = []

    for _, row in latest_rows.iterrows():

        result = calculate_risk_score(
            predicted_temp=(
                row[
                    "predicted_temperature_max"
                ]
            ),
            heatwave_probability=(
                row[
                    "heatwave_probability"
                ]
            ),
            departure=row[
                "departure"
            ],
            consecutive_heatwave_days=(
                row[
                    "consecutive_heatwave_days"
                ]
            ),
            humidity=row[
                "humidity_lag_1"
            ],
            region_type=row[
                "region_type"
            ],
            thresholds=thresholds,
        )

        result[
            "risk_category"
        ] = risk_category(
            result[
                "risk_score"
            ],
            thresholds,
        )

        risk_results.append(
            result
        )

    risk_df = pd.DataFrame(
        risk_results
    )

    # -----------------------------------
    # Combine risk results
    # -----------------------------------

    latest_rows = pd.concat(
        [
            latest_rows.reset_index(
                drop=True
            ),
            risk_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # -----------------------------------
    # Top factors + advisories
    # -----------------------------------

    top_factor_1 = []
    top_factor_2 = []
    top_factor_3 = []
    advisories = []

    for _, row in latest_rows.iterrows():

        factors = get_top_factors(
            risk_result=row.to_dict(),
            predicted_temperature=(
                row[
                    "predicted_temperature_max"
                ]
            ),
            heatwave_probability=(
                row[
                    "heatwave_probability"
                ]
            ),
            departure=row[
                "departure"
            ],
            consecutive_heatwave_days=(
                row[
                    "consecutive_heatwave_days"
                ]
            ),
            humidity=row[
                "humidity_lag_1"
            ],
        )

        top_factor_1.append(
            factors[0]
        )

        top_factor_2.append(
            factors[1]
        )

        top_factor_3.append(
            factors[2]
        )

        advisories.append(
            generate_advisory(
                risk_category_value=(
                    row[
                        "risk_category"
                    ]
                ),
                severity=row[
                    "severity"
                ],
                predicted_temperature=(
                    row[
                        "predicted_temperature_max"
                    ]
                ),
                heatwave_probability=(
                    row[
                        "heatwave_probability"
                    ]
                ),
                humidity=row[
                    "humidity_lag_1"
                ],
            )
        )

    latest_rows[
        "top_factor_1"
    ] = top_factor_1

    latest_rows[
        "top_factor_2"
    ] = top_factor_2

    latest_rows[
        "top_factor_3"
    ] = top_factor_3

    latest_rows[
        "advisory"
    ] = advisories

    # -----------------------------------
    # Forecast date
    # -----------------------------------

    latest_rows[
        "forecast_date"
    ] = (
        latest_rows[
            "date"
        ]
        + pd.Timedelta(days=1)
    )

    # -----------------------------------
    # Final output
    # -----------------------------------

    output_columns = [
        "forecast_date",
        "location_id",
        "city",
        "state",
        "latitude",
        "longitude",
        "region_type",
        "date",
        "temperature_max",
        "predicted_temperature_max",
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

    output = latest_rows[
        output_columns
    ].copy()

    # -----------------------------------
    # Rename workflow-facing fields
    # -----------------------------------

    output = output.rename(
        columns={
            "predicted_temperature_max":
                "predicted_tmax"
        }
    )

    # -----------------------------------
    # Highest risk first
    # -----------------------------------

    output = output.sort_values(
        "risk_score",
        ascending=False,
    )

    # -----------------------------------
    # Save results
    # -----------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------
    # Display results
    # -----------------------------------

    print()
    print(
        "Prediction complete."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()

    print(
        output[
            [
                "location_id",
                "city",
                "predicted_tmax",
                "heatwave_probability",
                "severity",
                "risk_score",
                "risk_category",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()