import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# Configuration
# -----------------------------------

INPUT_FILE = Path(
    "data/processed/features.csv"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/metrics")

MODEL_FILE = MODEL_DIR / "temperature_model.joblib"
FEATURE_FILE = MODEL_DIR / "temperature_features.joblib"
METRICS_FILE = OUTPUT_DIR / "regression_metrics.json"


# Time-based split
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


TARGET_COLUMN = "target_temperature_max"


# -----------------------------------
# Load dataset
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
# Train model
# -----------------------------------

def train_model(X_train, y_train):

    # Chosen over a 300-tree RandomForest (366 MB) and smaller RFs on 2023
    # validation MAE; ~0.2 MB on disk.
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# -----------------------------------
# Evaluate model
# -----------------------------------

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
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

    # -----------------------------------
    # Time split
    # -----------------------------------

    train, validation, test = split_data(df)

    print()
    print("Time-based split:")
    print(
        f"Training:   {len(train)} records"
    )
    print(
        f"Validation: {len(validation)} records"
    )
    print(
        f"Test:       {len(test)} records"
    )

    # -----------------------------------
    # Prepare X and y
    # -----------------------------------

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # -----------------------------------
    # Train
    # -----------------------------------

    print()
    print("Training Random Forest...")

    model = train_model(
        X_train,
        y_train
    )

    # -----------------------------------
    # Evaluate
    # -----------------------------------

    print()
    print("Evaluating model...")

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print()
    print("Validation Metrics:")
    print(validation_metrics)

    print()
    print("Test Metrics:")
    print(test_metrics)

    # -----------------------------------
    # Save model
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
        model,
        MODEL_FILE,
        compress=3
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_FILE
    )

    # -----------------------------------
    # Save metrics
    # -----------------------------------

    metrics = {
        "model": "HistGradientBoostingRegressor",
        "random_state": 42,
        "training_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "features": FEATURE_COLUMNS,
        "validation": validation_metrics,
        "test": test_metrics
    }

    import json

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
    print("Temperature model training complete.")

    print(
        f"Model saved to: {MODEL_FILE}"
    )

    print(
        f"Feature list saved to: {FEATURE_FILE}"
    )

    print(
        f"Metrics saved to: {METRICS_FILE}"
    )


if __name__ == "__main__":
    main()