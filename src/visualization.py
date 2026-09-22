from pathlib import Path

import folium
import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "forecast_results.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_predictions():
    """Load the latest prediction results."""

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_PATH}"
        )

    return pd.read_csv(
        PREDICTION_PATH
    )


def create_risk_ranking(df):
    """Create a ranked table of locations by risk."""

    ranking = df[
        [
            "location_id",
            "city",
            "state",
            "predicted_temperature_max",
            "heatwave_probability",
            "risk_score",
            "risk_category",
        ]
    ].copy()

    ranking = ranking.sort_values(
        "risk_score",
        ascending=False,
    )

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1),
    )

    output_path = (
        OUTPUT_DIR
        / "risk_ranking.csv"
    )

    ranking.to_csv(
        output_path,
        index=False,
    )

    return ranking


def create_risk_chart(df):
    """Create risk-score comparison chart."""

    chart_df = df.sort_values(
        "risk_score",
        ascending=True,
    )

    fig = px.bar(
        chart_df,
        x="risk_score",
        y="city",
        orientation="h",
        title="Hyperlocal Heat Risk Score by Location",
        labels={
            "risk_score": "Risk Score (0-100)",
            "city": "Location",
        },
        range_x=[0, 100],
        text="risk_score",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig.write_html(
        OUTPUT_DIR
        / "risk_score_chart.html"
    )


def create_temperature_chart(df):
    """Compare observed and predicted temperature."""

    chart_df = df[
        [
            "city",
            "temperature_max",
            "predicted_temperature_max",
        ]
    ].copy()

    chart_df = chart_df.melt(
        id_vars="city",
        value_vars=[
            "temperature_max",
            "predicted_temperature_max",
        ],
        var_name="temperature_type",
        value_name="temperature",
    )

    chart_df[
        "temperature_type"
    ] = chart_df[
        "temperature_type"
    ].replace(
        {
            "temperature_max": "Observed",
            "predicted_temperature_max": "Predicted",
        }
    )

    fig = px.bar(
        chart_df,
        x="city",
        y="temperature",
        color="temperature_type",
        barmode="group",
        title="Observed vs Predicted Maximum Temperature",
        labels={
            "city": "Location",
            "temperature": "Temperature (°C)",
            "temperature_type": "Temperature",
        },
    )

    fig.write_html(
        OUTPUT_DIR
        / "temperature_comparison.html"
    )


def create_probability_chart(df):
    """Create heatwave probability chart."""

    chart_df = df.sort_values(
        "heatwave_probability",
        ascending=True,
    ).copy()

    chart_df[
        "heatwave_probability_percent"
    ] = (
        chart_df["heatwave_probability"]
        * 100
    )

    fig = px.bar(
        chart_df,
        x="heatwave_probability_percent",
        y="city",
        orientation="h",
        title="Heatwave Probability by Location",
        labels={
            "heatwave_probability_percent":
                "Heatwave Probability (%)",
            "city": "Location",
        },
        range_x=[0, 100],
        text="heatwave_probability_percent",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig.write_html(
        OUTPUT_DIR
        / "heatwave_probability_chart.html"
    )


def create_hotspot_map(df):
    """Create an interactive geographic hotspot map."""

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    map_object = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=5,
        tiles="OpenStreetMap",
    )

    for _, row in df.iterrows():

        popup_text = (
            f"<b>{row['city']}</b><br>"
            f"Risk Score: {row['risk_score']:.2f}<br>"
            f"Risk Category: {row['risk_category']}<br>"
            f"Predicted Tmax: "
            f"{row['predicted_temperature_max']:.2f} °C<br>"
            f"Heatwave Probability: "
            f"{row['heatwave_probability'] * 100:.2f}%"
        )

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"],
            ],
            radius=8,
            popup=folium.Popup(
                popup_text,
                max_width=300,
            ),
            tooltip=(
                f"{row['city']} — "
                f"Risk {row['risk_score']:.2f}"
            ),
            fill=True,
        ).add_to(map_object)

    map_path = (
        OUTPUT_DIR
        / "heat_risk_hotspot_map.html"
    )

    map_object.save(
        map_path
    )


def main():

    print(
        "Loading prediction results..."
    )

    df = load_predictions()

    print(
        f"Creating visualizations for "
        f"{len(df)} locations..."
    )

    # Risk ranking
    ranking = create_risk_ranking(
        df
    )

    # Charts
    create_risk_chart(df)
    create_temperature_chart(df)
    create_probability_chart(df)

    # Map
    create_hotspot_map(df)

    print()
    print(
        "Visualization generation complete."
    )

    print()
    print("Risk ranking:")
    print(
        ranking.to_string(
            index=False
        )
    )

    print()
    print(
        f"Visualizations saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()