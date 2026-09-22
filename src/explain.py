from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_model.joblib"
)

FEATURES_LIST_PATH = (
    PROJECT_ROOT
    / "models"
    / "temperature_features.joblib"
)

LOCATIONS_PATH = (
    PROJECT_ROOT
    / "config"
    / "locations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualizations"
)

EXPLANATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "explanations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPLANATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def convert_to_float(value):
    """
    Convert numpy/pandas values into normal Python floats.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_shap_array(shap_values):
    """
    Normalize SHAP output into a 2D array.

    Different SHAP versions can return slightly different
    structures. The temperature model is a regression model,
    so the expected final shape is:

        rows x features
    """

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    if shap_values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )

    return shap_values


def main():

    print("==========================================")
    print("HEATWATCH AI — EXPLAINABLE AI / SHAP")
    print("==========================================")
    print()

    # ---------------------------------------
    # Load model
    # ---------------------------------------

    print("Loading temperature model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"✓ Model loaded: {MODEL_PATH}"
    )

    # ---------------------------------------
    # Load feature list
    # ---------------------------------------

    print(
        "Loading temperature feature list..."
    )

    feature_columns = joblib.load(
        FEATURES_LIST_PATH
    )

    print(
        f"✓ Features loaded: "
        f"{len(feature_columns)}"
    )

    # ---------------------------------------
    # Load feature data
    # ---------------------------------------

    print(
        "Loading processed feature data..."
    )

    df = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"✓ Feature rows loaded: {len(df)}"
    )

    # ---------------------------------------
    # Verify columns
    # ---------------------------------------

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The following model features "
            "are missing from features.csv:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ---------------------------------------
    # Prepare model input
    # ---------------------------------------

    X = df[
        feature_columns
    ].copy()

    # ---------------------------------------
    # Global SHAP analysis
    # ---------------------------------------

    sample_size = min(
        1000,
        len(X)
    )

    X_sample = X.sample(
        n=sample_size,
        random_state=42,
    )

    print()
    print(
        f"Calculating SHAP values for "
        f"{sample_size} samples..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = get_shap_array(
        shap_values
    )

    print(
        f"✓ SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # ---------------------------------------
    # Global feature importance
    # ---------------------------------------

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": (
                np.abs(shap_values)
                .mean(axis=0)
            ),
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )

    importance_path = (
        OUTPUT_DIR
        / "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print()
    print(
        "Top features influencing "
        "temperature prediction:"
    )

    print(
        importance
        .head(10)
        .to_string(index=False)
    )

    # ---------------------------------------
    # SHAP summary plot
    # ---------------------------------------

    print()
    print(
        "Generating SHAP summary plot..."
    )

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.tight_layout()

    summary_path = (
        OUTPUT_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"✓ SHAP summary saved to: "
        f"{summary_path}"
    )

    # ---------------------------------------
    # Latest row for each location
    # ---------------------------------------

    if "location_id" not in df.columns:
        raise ValueError(
            "features.csv does not contain "
            "'location_id'."
        )

    if "date" not in df.columns:
        raise ValueError(
            "features.csv does not contain "
            "'date'."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    latest_rows = (
        df.sort_values(
            [
                "location_id",
                "date",
            ]
        )
        .groupby(
            "location_id",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest_rows = latest_rows.sort_values(
        "location_id"
    )

    X_latest = latest_rows[
        feature_columns
    ].copy()

    print()
    print(
        f"Calculating local SHAP explanations "
        f"for {len(X_latest)} locations..."
    )

    # ---------------------------------------
    # Local SHAP values
    # ---------------------------------------

    local_shap_values = explainer.shap_values(
        X_latest
    )

    local_shap_values = get_shap_array(
        local_shap_values
    )

    # ---------------------------------------
    # Expected/base values
    # ---------------------------------------

    expected_value = explainer.expected_value

    if isinstance(
        expected_value,
        (list, np.ndarray)
    ):

        expected_value = np.asarray(
            expected_value
        ).reshape(-1)[0]

    expected_value = convert_to_float(
        expected_value
    )

    # ---------------------------------------
    # Load location metadata
    # ---------------------------------------

    locations = None

    if LOCATIONS_PATH.exists():

        locations = pd.read_csv(
            LOCATIONS_PATH
        )

        locations = locations[
            [
                "location_id",
                "city",
                "state",
                "latitude",
                "longitude",
                "region_type",
            ]
        ]

        latest_rows = latest_rows.merge(
            locations,
            on="location_id",
            how="left",
            suffixes=("", "_metadata"),
        )

    # ---------------------------------------
    # Build local explanation JSON
    # ---------------------------------------

    local_explanations = {}

    for row_index, (_, row) in enumerate(
        latest_rows.iterrows()
    ):

        location_id = str(
            row["location_id"]
        )

        current_features = X_latest.iloc[
            row_index
        ]

        current_shap = local_shap_values[
            row_index
        ]

        prediction = model.predict(
            current_features
            .to_frame()
            .T
        )[0]

        prediction = convert_to_float(
            prediction
        )

        feature_explanations = []

        for feature_index, feature in enumerate(
            feature_columns
        ):

            feature_value = (
                current_features.iloc[
                    feature_index
                ]
            )

            shap_value = (
                current_shap[
                    feature_index
                ]
            )

            feature_explanations.append(
                {
                    "feature": str(
                        feature
                    ),
                    "value": convert_to_float(
                        feature_value
                    ),
                    "shap_value": convert_to_float(
                        shap_value
                    ),
                    "direction": (
                        "increases_prediction"
                        if shap_value > 0
                        else "decreases_prediction"
                        if shap_value < 0
                        else "neutral"
                    ),
                }
            )

        feature_explanations.sort(
            key=lambda item: abs(
                item["shap_value"] or 0
            ),
            reverse=True,
        )

        # Keep the strongest contributors
        top_features = (
            feature_explanations[:10]
        )

        location_data = {
            "location_id": location_id,
            "date": row["date"].strftime(
                "%Y-%m-%d"
            ),
            "base_value": expected_value,
            "predicted_temperature": prediction,
            "features": top_features,
        }

        if locations is not None:

            city = row.get(
                "city",
                None
            )

            state = row.get(
                "state",
                None
            )

            latitude = row.get(
                "latitude",
                None
            )

            longitude = row.get(
                "longitude",
                None
            )

            region_type = row.get(
                "region_type",
                None
            )

            location_data.update(
                {
                    "city": (
                        str(city)
                        if pd.notna(city)
                        else None
                    ),
                    "state": (
                        str(state)
                        if pd.notna(state)
                        else None
                    ),
                    "latitude": convert_to_float(
                        latitude
                    ),
                    "longitude": convert_to_float(
                        longitude
                    ),
                    "region_type": (
                        str(region_type)
                        if pd.notna(region_type)
                        else None
                    ),
                }
            )

        local_explanations[
            location_id
        ] = location_data

    # ---------------------------------------
    # Save local explanations
    # ---------------------------------------

    local_json_path = (
        EXPLANATION_DIR
        / "shap_local.json"
    )

    with open(
        local_json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            local_explanations,
            file,
            indent=2,
        )

    print(
        f"✓ Local SHAP explanations saved to: "
        f"{local_json_path}"
    )

    # ---------------------------------------
    # Save local explanation table
    # ---------------------------------------

    local_rows = []

    for location_id, explanation in (
        local_explanations.items()
    ):

        for feature in explanation[
            "features"
        ]:

            local_rows.append(
                {
                    "location_id": location_id,
                    "date": explanation[
                        "date"
                    ],
                    "feature": feature[
                        "feature"
                    ],
                    "feature_value": feature[
                        "value"
                    ],
                    "shap_value": feature[
                        "shap_value"
                    ],
                    "direction": feature[
                        "direction"
                    ],
                }
            )

    local_dataframe = pd.DataFrame(
        local_rows
    )

    local_csv_path = (
        EXPLANATION_DIR
        / "shap_local.csv"
    )

    local_dataframe.to_csv(
        local_csv_path,
        index=False,
    )

    print(
        f"✓ Local SHAP table saved to: "
        f"{local_csv_path}"
    )

    # ---------------------------------------
    # Save metadata
    # ---------------------------------------

    metadata = {
        "model": "Random Forest Regressor",
        "target": "next-day Tmax",
        "method": "SHAP TreeExplainer",
        "random_state": 42,
        "sample_size_global": sample_size,
        "locations_explained": len(
            local_explanations
        ),
        "top_features_per_location": 10,
        "interpretation_note": (
            "SHAP values explain the contribution "
            "of model features to the temperature "
            "prediction. They describe model behavior "
            "and should not be interpreted as causal effects."
        ),
    }

    metadata_path = (
        EXPLANATION_DIR
        / "shap_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"✓ SHAP metadata saved to: "
        f"{metadata_path}"
    )

    # ---------------------------------------
    # Final output
    # ---------------------------------------

    print()
    print("==========================================")
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print("==========================================")
    print()
    print(
        "Generated files:"
    )
    print(
        f"  ✓ {importance_path}"
    )
    print(
        f"  ✓ {summary_path}"
    )
    print(
        f"  ✓ {local_json_path}"
    )
    print(
        f"  ✓ {local_csv_path}"
    )
    print(
        f"  ✓ {metadata_path}"
    )
    print()


if __name__ == "__main__":
    main()