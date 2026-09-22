from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"


def load_thresholds():
    """Load risk-score configuration from thresholds.yaml."""
    with open(THRESHOLDS_PATH, "r") as file:
        return yaml.safe_load(file)


def temperature_component(predicted_temp, region_type, thresholds):
    """
    Convert predicted temperature into a 0-100 risk component.

    The component reaches 100 when the predicted temperature is
    5°C above the regional heatwave temperature threshold.
    """

    threshold = thresholds["heatwave"][region_type]["temperature_threshold_c"]

    score = ((predicted_temp - (threshold - 5.0)) / 10.0) * 100.0

    return max(0.0, min(100.0, score))


def probability_component(heatwave_probability):
    """Convert heatwave probability (0-1) into 0-100."""
    return max(0.0, min(100.0, heatwave_probability * 100.0))


def anomaly_component(departure):
    """
    Convert temperature departure into a 0-100 component.

    0°C departure -> 0
    6.5°C departure -> 100
    """

    score = (departure / 6.5) * 100.0

    return max(0.0, min(100.0, score))


def persistence_component(consecutive_heatwave_days):
    """
    Convert consecutive heatwave days into a 0-100 component.

    0 days -> 0
    3 or more days -> 100
    """

    score = (consecutive_heatwave_days / 3.0) * 100.0

    return max(0.0, min(100.0, score))


def humidity_component(humidity):
    """
    Convert humidity into a 0-100 heat-stress component.

    Below 40% -> 0
    80% or above -> 100
    """

    score = ((humidity - 40.0) / 40.0) * 100.0

    return max(0.0, min(100.0, score))


def calculate_risk_score(
    predicted_temp,
    heatwave_probability,
    departure,
    consecutive_heatwave_days,
    humidity,
    region_type,
    thresholds,
):
    """
    Calculate the final 0-100 Hyperlocal Heat Risk Score.
    """

    temp_score = temperature_component(
        predicted_temp,
        region_type,
        thresholds,
    )

    probability_score = probability_component(
        heatwave_probability
    )

    anomaly_score = anomaly_component(departure)

    persistence_score = persistence_component(
        consecutive_heatwave_days
    )

    humidity_score = humidity_component(humidity)

    weights = thresholds["risk_score"]

    final_score = (
        weights["temperature_weight"] * temp_score
        + weights["heatwave_probability_weight"] * probability_score
        + weights["anomaly_weight"] * anomaly_score
        + weights["persistence_weight"] * persistence_score
        + weights["humidity_weight"] * humidity_score
    )

    final_score = max(0.0, min(100.0, final_score))

    return {
        "temperature_component": round(temp_score, 2),
        "heatwave_probability_component": round(probability_score, 2),
        "anomaly_component": round(anomaly_score, 2),
        "persistence_component": round(persistence_score, 2),
        "humidity_component": round(humidity_score, 2),
        "risk_score": round(final_score, 2),
    }


def risk_category(score, thresholds):
    """Convert a 0-100 score into the configured risk category."""

    categories = thresholds["risk_categories"]

    for category, limits in categories.items():
        if limits["min"] <= score <= limits["max"]:
            return category.replace("_", " ").title()

    return "Unknown"


if __name__ == "__main__":

    thresholds = load_thresholds()

    # Example calculation for testing the risk engine.
    result = calculate_risk_score(
        predicted_temp=42.0,
        heatwave_probability=0.80,
        departure=5.5,
        consecutive_heatwave_days=2,
        humidity=65.0,
        region_type="plains",
        thresholds=thresholds,
    )

    category = risk_category(
        result["risk_score"],
        thresholds,
    )

    print("Risk Score Test")
    print("----------------")
    print(f"Temperature Component : {result['temperature_component']}")
    print(
        f"Heatwave Probability : "
        f"{result['heatwave_probability_component']}"
    )
    print(f"Anomaly Component     : {result['anomaly_component']}")
    print(f"Persistence Component : {result['persistence_component']}")
    print(f"Humidity Component    : {result['humidity_component']}")
    print(f"Final Risk Score      : {result['risk_score']}")
    print(f"Risk Category         : {category}")