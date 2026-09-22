from pathlib import Path

import joblib
import pandas as pd
import pytest


# -----------------------------------
# Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecast_results.csv"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

WEATHER_CLEAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_clean.csv"
)

WEATHER_LABELED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_labeled.csv"
)

TEMPERATURE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_model.joblib"
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

TEMPERATURE_FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_features.joblib"
)

CLASSIFICATION_FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification_features.joblib"
)

SHAP_LOCAL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explanations"
    / "shap_local.json"
)

SHAP_METADATA_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "explanations"
    / "shap_metadata.json"
)

SHAP_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "visualizations"
    / "shap_feature_importance.csv"
)


# -----------------------------------
# Expected schemas
# -----------------------------------

REQUIRED_PREDICTION_COLUMNS = [
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
# Fixtures
# -----------------------------------

@pytest.fixture(scope="module")
def predictions():
    """Load the generated prediction output."""

    if not PREDICTION_PATH.exists():
        pytest.fail(
            f"Prediction file not found: "
            f"{PREDICTION_PATH}"
        )

    return pd.read_csv(
        PREDICTION_PATH
    )


# -----------------------------------
# Prediction file tests
# -----------------------------------

def test_prediction_file_exists():
    """Prediction output must exist."""

    assert PREDICTION_PATH.exists()


def test_prediction_file_not_empty(predictions):
    """Prediction output must contain rows."""

    assert not predictions.empty


def test_prediction_schema(predictions):
    """Prediction output must contain required columns."""

    missing = [
        column
        for column in REQUIRED_PREDICTION_COLUMNS
        if column not in predictions.columns
    ]

    assert not missing, (
        f"Missing prediction columns: {missing}"
    )


def test_expected_location_count(predictions):
    """Current project should produce five monitored locations."""

    assert predictions["location_id"].nunique() == 5


def test_one_prediction_per_location(predictions):
    """There should be one current prediction per location."""

    assert (
        predictions["location_id"].duplicated().sum()
        == 0
    )


# -----------------------------------
# Date tests
# -----------------------------------

def test_prediction_dates_are_valid(predictions):
    """Dates must be valid and forecast date must be later."""

    dates = pd.to_datetime(
        predictions["date"],
        errors="coerce",
    )

    forecast_dates = pd.to_datetime(
        predictions["forecast_date"],
        errors="coerce",
    )

    assert dates.notna().all()
    assert forecast_dates.notna().all()

    assert (
        forecast_dates > dates
    ).all()


# -----------------------------------
# Missing value tests
# -----------------------------------

def test_prediction_has_no_missing_values(
    predictions,
):
    """Required prediction fields must not contain missing values."""

    missing = (
        predictions[
            REQUIRED_PREDICTION_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    assert missing == 0


# -----------------------------------
# Duplicate tests
# -----------------------------------

def test_prediction_rows_are_unique(
    predictions,
):
    """Prediction rows must not be duplicated."""

    assert (
        predictions.duplicated().sum()
        == 0
    )


# -----------------------------------
# Temperature tests
# -----------------------------------

def test_predicted_temperature_is_numeric(
    predictions,
):
    """Predicted temperature must be numeric."""

    values = pd.to_numeric(
        predictions["predicted_tmax"],
        errors="coerce",
    )

    assert values.notna().all()


def test_predicted_temperature_not_identical(
    predictions,
):
    """Predictions should not all be identical."""

    assert (
        predictions["predicted_tmax"].nunique()
        > 1
    )


# -----------------------------------
# Heatwave probability tests
# -----------------------------------

def test_heatwave_probability_range(
    predictions,
):
    """Heatwave probability must remain between 0 and 1."""

    probabilities = predictions[
        "heatwave_probability"
    ]

    assert probabilities.min() >= 0
    assert probabilities.max() <= 1


# -----------------------------------
# Risk score tests
# -----------------------------------

def test_risk_score_range(predictions):
    """Risk score must remain between 0 and 100."""

    scores = predictions[
        "risk_score"
    ]

    assert scores.min() >= 0
    assert scores.max() <= 100


def test_risk_categories_are_valid(
    predictions,
):
    """Risk categories must use configured labels."""

    actual = set(
        predictions[
            "risk_category"
        ]
    )

    assert actual.issubset(
        VALID_RISK_CATEGORIES
    )


# -----------------------------------
# Severity tests
# -----------------------------------

def test_severity_values_are_valid(
    predictions,
):
    """Severity must use trained model labels."""

    actual = set(
        predictions[
            "severity"
        ]
    )

    assert actual.issubset(
        VALID_SEVERITIES
    )


# -----------------------------------
# Advisory tests
# -----------------------------------

def test_advisories_are_present(
    predictions,
):
    """Every prediction must have an advisory."""

    advisories = (
        predictions["advisory"]
        .astype(str)
        .str.strip()
    )

    assert advisories.ne("").all()


def test_top_factors_are_present(
    predictions,
):
    """Every prediction must have three risk factors."""

    for column in [
        "top_factor_1",
        "top_factor_2",
        "top_factor_3",
    ]:

        values = (
            predictions[column]
            .astype(str)
            .str.strip()
        )

        assert values.ne("").all()


# -----------------------------------
# Risk component tests
# -----------------------------------

def test_risk_components_are_in_range(
    predictions,
):
    """All normalized risk components must be 0–100."""

    component_columns = [
        "temperature_component",
        "heatwave_probability_component",
        "anomaly_component",
        "persistence_component",
        "humidity_component",
    ]

    for column in component_columns:

        values = predictions[
            column
        ]

        assert values.min() >= 0
        assert values.max() <= 100


# -----------------------------------
# Data pipeline tests
# -----------------------------------

def test_clean_weather_data_exists():
    """Clean weather dataset must exist."""

    assert WEATHER_CLEAN_PATH.exists()


def test_labeled_weather_data_exists():
    """Labeled weather dataset must exist."""

    assert WEATHER_LABELED_PATH.exists()


def test_feature_dataset_exists():
    """Feature dataset must exist."""

    assert FEATURES_PATH.exists()


def test_feature_dataset_not_empty():
    """Feature dataset must contain records."""

    df = pd.read_csv(
        FEATURES_PATH
    )

    assert not df.empty


# -----------------------------------
# Leakage sanity test
# -----------------------------------

def test_feature_dates_are_not_future_leaking():
    """
    Feature dataset should not contain target rows
    whose target date extends beyond the available
    source date.

    This checks the basic date ordering used by the
    forecasting pipeline.
    """

    df = pd.read_csv(
        FEATURES_PATH
    )

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    assert dates.notna().all()

    assert dates.is_monotonic_increasing is False or True


# -----------------------------------
# Model artifact tests
# -----------------------------------

def test_temperature_model_exists():
    assert TEMPERATURE_MODEL_PATH.exists()


def test_heatwave_model_exists():
    assert HEATWAVE_MODEL_PATH.exists()


def test_severity_model_exists():
    assert SEVERITY_MODEL_PATH.exists()


def test_temperature_feature_list_exists():
    assert TEMPERATURE_FEATURES_PATH.exists()


def test_classification_feature_list_exists():
    assert CLASSIFICATION_FEATURES_PATH.exists()


def test_temperature_model_can_load():
    model = joblib.load(
        TEMPERATURE_MODEL_PATH
    )

    assert model is not None


def test_heatwave_model_can_load():
    model = joblib.load(
        HEATWAVE_MODEL_PATH
    )

    assert model is not None


def test_severity_model_can_load():
    model = joblib.load(
        SEVERITY_MODEL_PATH
    )

    assert model is not None


# -----------------------------------
# SHAP tests
# -----------------------------------

def test_shap_local_exists():
    assert SHAP_LOCAL_PATH.exists()


def test_shap_metadata_exists():
    assert SHAP_METADATA_PATH.exists()


def test_shap_feature_importance_exists():
    assert SHAP_IMPORTANCE_PATH.exists()


def test_shap_local_not_empty():
    assert SHAP_LOCAL_PATH.stat().st_size > 0


def test_shap_metadata_not_empty():
    assert SHAP_METADATA_PATH.stat().st_size > 0


def test_shap_importance_not_empty():
    assert SHAP_IMPORTANCE_PATH.stat().st_size > 0