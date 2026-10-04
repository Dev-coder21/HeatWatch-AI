import json
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = Path(
    "data/processed/features.csv"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/metrics")

HEATWAVE_MODEL_FILE = (
    MODEL_DIR / "heatwave_model.joblib"
)

SEVERITY_MODEL_FILE = (
    MODEL_DIR / "severity_model.joblib"
)

FEATURE_FILE = (
    MODEL_DIR / "classification_features.joblib"
)

METRICS_FILE = (
    OUTPUT_DIR / "classification_metrics.json"
)

TRAIN_END = "2022-12-31"
VALIDATION_END = "2023-12-31"


# -----------------------------------
# Features
# -----------------------------------

FEATURE_COLUMNS = [
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
    "month",
    "day_of_year",
]
# latitude/longitude were removed: with only 5 training cities they let the
# models memorise city identity instead of learning heat behaviour.


# -----------------------------------
# Load data
# -----------------------------------

def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


# -----------------------------------
# Time-based split
# -----------------------------------

def split_data(df):

    train = df[
        df["date"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["date"] > TRAIN_END)
        & (df["date"] <= VALIDATION_END)
    ].copy()

    test = df[
        df["date"] > VALIDATION_END
    ].copy()

    return train, validation, test


# -----------------------------------
# Train heatwave classifier
# -----------------------------------

def train_heatwave_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# -----------------------------------
# Train severity classifier
# -----------------------------------

def train_severity_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# -----------------------------------
# Evaluate heatwave classifier
# -----------------------------------

def evaluate_heatwave(
    model,
    X,
    y
):

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

    except ValueError:

        roc_auc = None

    matrix = confusion_matrix(
        y,
        predictions
    ).tolist()

    # Rare-class metrics: PR-AUC and Brier score are more honest than accuracy.
    pr_auc = (
        average_precision_score(y, probabilities)
        if y.sum() > 0 else None
    )
    brier = brier_score_loss(y, probabilities)

    return {
        "positives": int(y.sum()),
        "pr_auc": pr_auc,
        "brier": brier,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": matrix
    }


# -----------------------------------
# Evaluate severity classifier
# -----------------------------------

def evaluate_severity(
    model,
    X,
    y
):

    predictions = model.predict(X)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    matrix = confusion_matrix(
        y,
        predictions
    ).tolist()

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": matrix
    }


# -----------------------------------
# Main
# -----------------------------------

def main():

    print("Loading feature dataset...")

    df = load_data()

    print(
        f"Total records: {len(df)}"
    )

    train, validation, test = split_data(df)

    print()
    print("Time-based split:")
    print(
        f"Training:   {len(train)}"
    )
    print(
        f"Validation: {len(validation)}"
    )
    print(
        f"Test:       {len(test)}"
    )

    # -----------------------------------
    # Prepare features
    # -----------------------------------

    X_train = train[FEATURE_COLUMNS]
    X_validation = validation[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    # -----------------------------------
    # Heatwave target
    # -----------------------------------

    y_train_heatwave = train[
        "target_heatwave"
    ].astype(int)

    y_validation_heatwave = validation[
        "target_heatwave"
    ].astype(int)

    y_test_heatwave = test[
        "target_heatwave"
    ].astype(int)

    # -----------------------------------
    # Train heatwave model
    # -----------------------------------

    print()
    print("Training heatwave classifier...")

    heatwave_model = train_heatwave_model(
        X_train,
        y_train_heatwave
    )

    # -----------------------------------
    # Evaluate heatwave model
    # -----------------------------------

    validation_heatwave_metrics = (
        evaluate_heatwave(
            heatwave_model,
            X_validation,
            y_validation_heatwave
        )
    )

    test_heatwave_metrics = (
        evaluate_heatwave(
            heatwave_model,
            X_test,
            y_test_heatwave
        )
    )

    # -----------------------------------
    # Severity target
    # -----------------------------------

    y_train_severity = train[
        "target_severity"
    ]

    y_validation_severity = validation[
        "target_severity"
    ]

    y_test_severity = test[
        "target_severity"
    ]

    # -----------------------------------
    # Train severity model
    # -----------------------------------

    print()
    print("Training severity classifier...")

    severity_model = train_severity_model(
        X_train,
        y_train_severity
    )

    # -----------------------------------
    # Evaluate severity model
    # -----------------------------------

    validation_severity_metrics = (
        evaluate_severity(
            severity_model,
            X_validation,
            y_validation_severity
        )
    )

    test_severity_metrics = (
        evaluate_severity(
            severity_model,
            X_test,
            y_test_severity
        )
    )

    # -----------------------------------
    # Save models
    # -----------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        heatwave_model,
        HEATWAVE_MODEL_FILE
    )

    joblib.dump(
        severity_model,
        SEVERITY_MODEL_FILE
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_FILE
    )

    # -----------------------------------
    # Save metrics
    # -----------------------------------

    metrics = {

        "heatwave_classifier": {

            "model": "RandomForestClassifier",

            "random_state": 42,

            "validation":
                validation_heatwave_metrics,

            "test":
                test_heatwave_metrics
        },

        "severity_classifier": {

            "model": "RandomForestClassifier",

            "random_state": 42,

            "validation":
                validation_severity_metrics,

            "test":
                test_severity_metrics
        }
    }

    with open(
        METRICS_FILE,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print()
    print(
        "Heatwave classification training complete."
    )

    print(
        f"Heatwave model: "
        f"{HEATWAVE_MODEL_FILE}"
    )

    print(
        f"Severity model: "
        f"{SEVERITY_MODEL_FILE}"
    )

    print(
        f"Metrics: "
        f"{METRICS_FILE}"
    )


if __name__ == "__main__":
    main()