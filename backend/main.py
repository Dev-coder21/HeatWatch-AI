from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecast_results.csv"
)

EXPLANATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explanations"
    / "shap_local.json"
)

WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_clean.csv"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="HeatWatch AI API",
    description=(
        "AI-Based Hyperlocal Heat Risk Prediction "
        "and Explainable Early Warning System"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_predictions():
    """
    Load the prediction CSV and normalize its important
    column names so the frontend receives a stable API schema.
    """

    if not PREDICTION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Prediction file not found.",
        )

    try:
        df = pd.read_csv(PREDICTION_PATH)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read prediction file: {error}",
        )

    if df.empty:
        raise HTTPException(
            status_code=500,
            detail="Prediction file is empty.",
        )

    # --------------------------------------------------------
    # NORMALIZE LOCATION ID
    # --------------------------------------------------------

    location_candidates = [
        "location_id",
        "location",
        "id",
    ]

    location_column = next(
        (
            column
            for column in location_candidates
            if column in df.columns
        ),
        None,
    )

    if location_column is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "No location identifier found in "
                "forecast_results.csv."
            ),
        )

    if location_column != "location_id":
        df["location_id"] = df[location_column].astype(str)


    # --------------------------------------------------------
    # NORMALIZE PREDICTED TEMPERATURE
    # --------------------------------------------------------

    temperature_candidates = [
        "predicted_tmax",
        "predicted_temperature",
        "predicted_temperature_max",
        "predicted_tmax_c",
        "forecast_tmax",
        "temperature_prediction",
        "prediction",
    ]

    temperature_column = next(
        (
            column
            for column in temperature_candidates
            if column in df.columns
        ),
        None,
    )

    if temperature_column is not None:
        df["predicted_tmax"] = pd.to_numeric(
            df[temperature_column],
            errors="coerce",
        )

    else:
        raise HTTPException(
            status_code=500,
            detail=(
                "No predicted temperature column found in "
                "forecast_results.csv. Expected one of: "
                + ", ".join(temperature_candidates)
            ),
        )

    # Compatibility field used by the frontend.
    df["predicted_temperature_max"] = df["predicted_tmax"]


    # --------------------------------------------------------
    # NORMALIZE HEATWAVE PROBABILITY
    # --------------------------------------------------------

    probability_candidates = [
        "heatwave_probability",
        "heatwave_prob",
        "probability",
        "heatwave_probability_score",
    ]

    probability_column = next(
        (
            column
            for column in probability_candidates
            if column in df.columns
        ),
        None,
    )

    if probability_column is not None:
        df["heatwave_probability"] = pd.to_numeric(
            df[probability_column],
            errors="coerce",
        )

        # If accidentally stored as percentage values,
        # convert them to the required 0-1 range.
        df.loc[
            df["heatwave_probability"] > 1,
            "heatwave_probability",
        ] = (
            df.loc[
                df["heatwave_probability"] > 1,
                "heatwave_probability",
            ]
            / 100.0
        )

        df["heatwave_probability"] = (
            df["heatwave_probability"]
            .clip(0, 1)
        )

    else:
        df["heatwave_probability"] = 0.0


    # --------------------------------------------------------
    # NORMALIZE RISK SCORE
    # --------------------------------------------------------

    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce",
        )

    else:
        df["risk_score"] = 0.0


    # --------------------------------------------------------
    # NORMALIZE RISK CATEGORY
    # --------------------------------------------------------

    if "risk_category" not in df.columns:

        def get_risk_category(score):
            if pd.isna(score):
                return "Low"

            score = float(score)

            if score >= 80:
                return "Extreme"
            elif score >= 60:
                return "Very High"
            elif score >= 40:
                return "High"
            elif score >= 20:
                return "Moderate"
            else:
                return "Low"

        df["risk_category"] = df["risk_score"].apply(
            get_risk_category
        )


    # --------------------------------------------------------
    # NORMALIZE FORECAST DATE
    # --------------------------------------------------------

    date_candidates = [
        "forecast_date",
        "date",
        "prediction_date",
    ]

    date_column = next(
        (
            column
            for column in date_candidates
            if column in df.columns
        ),
        None,
    )

    if date_column is not None:
        df["forecast_date"] = (
            pd.to_datetime(
                df[date_column],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
        )


    # --------------------------------------------------------
    # NORMALIZE LATITUDE / LONGITUDE
    # --------------------------------------------------------

    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        )

    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        )


    # --------------------------------------------------------
    # NORMALIZE SEVERITY
    # --------------------------------------------------------

    if "severity" not in df.columns:

        if "heatwave_severity" in df.columns:
            df["severity"] = df[
                "heatwave_severity"
            ]

        else:
            df["severity"] = "No Heatwave"


    # --------------------------------------------------------
    # ADD LATEST HUMIDITY FROM CLEAN WEATHER DATA
    # --------------------------------------------------------

    if WEATHER_PATH.exists():

        try:
            weather_df = pd.read_csv(
                WEATHER_PATH
            )

            if (
                "location_id" in weather_df.columns
                and "humidity" in weather_df.columns
            ):

                weather_df["location_id"] = (
                    weather_df["location_id"]
                    .astype(str)
                )

                if "date" in weather_df.columns:
                    weather_df["date"] = pd.to_datetime(
                        weather_df["date"],
                        errors="coerce",
                    )

                    latest_weather = (
                        weather_df
                        .sort_values("date")
                        .groupby(
                            "location_id",
                            as_index=False,
                        )
                        .tail(1)
                    )

                else:
                    latest_weather = (
                        weather_df
                        .groupby(
                            "location_id",
                            as_index=False,
                        )
                        .tail(1)
                    )

                humidity_lookup = (
                    latest_weather[
                        [
                            "location_id",
                            "humidity",
                        ]
                    ]
                    .copy()
                )

                humidity_lookup["humidity"] = (
                    pd.to_numeric(
                        humidity_lookup["humidity"],
                        errors="coerce",
                    )
                )

                df = df.merge(
                    humidity_lookup,
                    on="location_id",
                    how="left",
                    suffixes=("", "_weather"),
                )

        except Exception as error:
            print(
                "Warning: unable to load humidity data:",
                error,
            )


    # --------------------------------------------------------
    # ENSURE HUMIDITY EXISTS
    # --------------------------------------------------------

    if "humidity" not in df.columns:
        df["humidity"] = None


    # --------------------------------------------------------
    # ENSURE LAT/LON FROM LOCATION CONFIG IF NEEDED
    # --------------------------------------------------------

    locations_config_path = (
        PROJECT_ROOT
        / "config"
        / "locations.csv"
    )

    if locations_config_path.exists():

        try:
            locations_df = pd.read_csv(
                locations_config_path
            )

            required_location_columns = {
                "location_id",
                "latitude",
                "longitude",
            }

            if required_location_columns.issubset(
                locations_df.columns
            ):

                config_locations = locations_df[
                    [
                        "location_id",
                        "latitude",
                        "longitude",
                        "city",
                        "state",
                        "region_type",
                    ]
                ].copy()

                config_locations["location_id"] = (
                    config_locations["location_id"]
                    .astype(str)
                )

                df["location_id"] = (
                    df["location_id"]
                    .astype(str)
                )

                df = df.merge(
                    config_locations,
                    on="location_id",
                    how="left",
                    suffixes=("", "_config"),
                )

                # Prefer existing valid coordinates,
                # otherwise use config coordinates.
                if "latitude_config" in df.columns:
                    df["latitude"] = pd.to_numeric(
                        df["latitude"],
                        errors="coerce",
                    ).fillna(
                        pd.to_numeric(
                            df["latitude_config"],
                            errors="coerce",
                        )
                    )

                if "longitude_config" in df.columns:
                    df["longitude"] = pd.to_numeric(
                        df["longitude"],
                        errors="coerce",
                    ).fillna(
                        pd.to_numeric(
                            df["longitude_config"],
                            errors="coerce",
                        )
                    )

        except Exception as error:
            print(
                "Warning: unable to load location config:",
                error,
            )


    # --------------------------------------------------------
    # FINAL JSON-SAFE CLEANUP
    # --------------------------------------------------------

    df = df.where(
        pd.notnull(df),
        None,
    )

    return df


def load_explanations():
    """
    Load local SHAP explanations.
    """

    if not EXPLANATION_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "SHAP explanation file not found. "
                "Run: python src/explain.py"
            ),
        )

    try:
        with open(
            EXPLANATION_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read SHAP explanation file: "
                f"{error}"
            ),
        )


def dataframe_to_records(df):
    """
    Convert DataFrame into JSON-safe records.
    """

    records = df.to_dict(
        orient="records"
    )

    cleaned_records = []

    for record in records:

        cleaned_record = {}

        for key, value in record.items():

            if value is None:
                cleaned_record[key] = None

            elif pd.isna(value):
                cleaned_record[key] = None

            elif hasattr(value, "item"):
                cleaned_record[key] = value.item()

            else:
                cleaned_record[key] = value

        cleaned_records.append(
            cleaned_record
        )

    return cleaned_records


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "HeatWatch AI",
        "description": (
            "AI-Based Hyperlocal Heat Risk Prediction "
            "and Explainable Early Warning System"
        ),
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "prediction_file_exists": (
            PREDICTION_PATH.exists()
        ),
        "explanation_file_exists": (
            EXPLANATION_PATH.exists()
        ),
        "weather_file_exists": (
            WEATHER_PATH.exists()
        ),
    }


# ============================================================
# ALL LOCATIONS
# ============================================================

@app.get("/api/locations")
def get_locations():

    df = load_predictions()

    records = dataframe_to_records(
        df
    )

    return {
        "count": len(records),
        "locations": records,
    }


# ============================================================
# ALL PREDICTIONS
# ============================================================

@app.get("/api/predictions")
def get_predictions():

    df = load_predictions()

    records = dataframe_to_records(
        df
    )

    return {
        "count": len(records),
        "predictions": records,
    }


# ============================================================
# HOTSPOTS
# ============================================================

@app.get("/api/hotspots")
def get_hotspots():

    df = load_predictions()

    if "risk_score" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="risk_score column not found.",
        )

    hotspots = (
        df.sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
    )

    records = dataframe_to_records(
        hotspots
    )

    return {
        "count": len(records),
        "hotspots": records,
    }


# ============================================================
# SINGLE LOCATION DETAILS
# ============================================================

@app.get("/api/locations/{location_id}")
def get_location(
    location_id: str,
):

    df = load_predictions()

    if "location_id" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "location_id column not found."
            ),
        )

    location = df[
        df["location_id"].astype(str)
        == str(location_id)
    ]

    if location.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Location '{location_id}' "
                "was not found."
            ),
        )

    record = dataframe_to_records(
        location
    )[0]

    return record


# ============================================================
# ALL SHAP EXPLANATIONS
# ============================================================

@app.get("/api/explanations")
def get_explanations():

    explanations = load_explanations()

    return {
        "count": len(explanations),
        "explanations": explanations,
    }


# ============================================================
# SHAP EXPLANATION FOR ONE LOCATION
# ============================================================

@app.get("/api/explanations/{location_id}")
def get_location_explanation(
    location_id: str,
):

    explanations = load_explanations()

    location_id = str(
        location_id
    )

    if location_id not in explanations:

        raise HTTPException(
            status_code=404,
            detail=(
                f"SHAP explanation for "
                f"'{location_id}' was not found."
            ),
        )

    return explanations[
        location_id
    ]


# ============================================================
# STARTUP MESSAGE
# ============================================================

@app.on_event("startup")
def startup_event():

    print()
    print("==========================================")
    print("HEATWATCH AI BACKEND")
    print("==========================================")

    print(
        f"Prediction file: "
        f"{PREDICTION_PATH}"
    )

    print(
        f"SHAP file: "
        f"{EXPLANATION_PATH}"
    )

    print(
        f"Weather file: "
        f"{WEATHER_PATH}"
    )

    print(
        f"Prediction file exists: "
        f"{PREDICTION_PATH.exists()}"
    )

    print(
        f"SHAP file exists: "
        f"{EXPLANATION_PATH.exists()}"
    )

    print(
        f"Weather file exists: "
        f"{WEATHER_PATH.exists()}"
    )

    print("==========================================")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )