# HeatWatch AI: Hyperlocal Heatwave Prediction & Explainable Early Warning System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF.svg)](https://vitejs.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-red.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, machine learning-powered early warning system designed for **hyperlocal heatwave prediction, terrain-aware risk assessment, and explainable climate intelligence**. HeatWatch AI combines meteorological data processing, time-series feature engineering, dual-stage machine learning (temperature regression + heatwave classification), Explainable AI (SHAP), a high-performance FastAPI backend, and an interactive React dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Workflow](#architecture--workflow)
- [Climatological Thresholds & Terrain Rules](#climatological-thresholds--terrain-rules)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Regression: Maximum Temperature Forecast](#regression-maximum-temperature-forecast)
  - [Classification: Heatwave Occurrence & Severity](#classification-heatwave-occurrence--severity)
  - [Explainable AI (SHAP)](#explainable-ai-shap)
- [Composite Heat Risk Scoring](#composite-heat-risk-scoring)
- [Project Structure](#project-structure)
- [Monitored Locations](#monitored-locations)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository & Setup Environment](#1-clone-repository--setup-environment)
  - [2. Run the Machine Learning Pipeline](#2-run-the-machine-learning-pipeline)
  - [3. Start the FastAPI Backend](#3-start-the-fastapi-backend)
  - [4. Start the React Frontend](#4-start-the-react-frontend)
- [API Reference](#api-reference)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [License](#license)

---

## Overview

Heatwaves are among the deadliest extreme weather events driven by climate change, yet conventional meteorological advisories are frequently too broad to trigger timely, local interventions. 

**HeatWatch AI** bridges this gap by:
1. Forecasting maximum daily temperatures with high accuracy ($R^2 > 0.96$, $MAE \approx 1.0^\circ\text{C}$).
2. Applying localized, India Meteorological Department (IMD) compliant thresholds across distinct terrain categories (**Plains**, **Hilly**, and **Coastal** regions).
3. Predicting the probability and severity of heatwaves using supervised ensemble learning.
4. Computing a **Composite Heat Risk Index (0–100)** incorporating thermal stress, departures from normal, persistence, and relative humidity.
5. Providing **SHAP-based local and global explanations** so meteorologists and public officials understand *why* an alert was issued.
6. Dispatching targeted, tiered public health advisories for vulnerable populations (outdoor laborers, elderly, and children).

---

## Key Features

- **Hyperlocal Weather Modeling**: Tailored to distinct geographic and climatic topographies across India (Pune, Mumbai, Delhi, Bengaluru, Shimla).
- **Dual-Model ML Architecture**:
  - **Random Forest Regressor**: Predicts continuous next-day maximum temperatures (`target_temperature_max`).
  - **Random Forest Classifiers**: Predicts binary heatwave occurrence and multi-class severity (`Normal`, `Heatwave`, `Severe Heatwave`).
- **Terrain-Aware Rule Engine**: Dynamically calculates climatological normal temperatures and departure deviations adhering to IMD criteria for coastal, plains, and hill stations.
- **Explainable AI (XAI)**:
  - Local per-city waterfall attribution explaining key drivers (e.g., lag temperatures, humidity drops, wind speed, rolling averages).
  - Global feature importance summaries for transparent model governance.
- **Composite Heat Risk Score (CHRS)**: Weighted multi-criteria index categorizing risk into 5 actionable levels: `Low`, `Moderate`, `High`, `Very High`, and `Extreme`.
- **FastAPI REST Service**: Clean, typed endpoints with CORS support, sub-millisecond response times, and automated OpenAPI documentation.
- **Modern React + Vite Dashboard**: Real-time hotspot maps, severity gauges, historical comparisons, interactive charts, and SHAP feature impact visualizers.

---

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Weather Data] --> B[data_cleaning.py]
        B --> C[weather_clean.csv]
        C --> D[heatwave_rules.py]
        D --> E[weather_labeled.csv]
        E --> F[feature_engineering.py]
        F --> G[features.csv]
    end

    subgraph Modeling & Explainability
        G --> H[train_temperature.py<br/>RandomForestRegressor]
        G --> I[train_heatwave.py<br/>RandomForestClassifier]
        H & I --> J[predict.py<br/>Forecast Generator]
        J --> K[risk_score.py<br/>Composite Risk Engine]
        H & J --> L[explain.py<br/>SHAP TreeExplainer]
        K & L --> M[forecast_results.csv & shap_local.json]
    end

    subgraph Serving & UI
        M --> N[FastAPI Backend<br/>backend/main.py]
        N -->|REST API / JSON| O[React Dashboard<br/>frontend/src/App.jsx]
    end
```

---

## Climatological Thresholds & Terrain Rules

Heatwave definitions depend heavily on local geography. HeatWatch AI implements standardized criteria based on climatological normal baselines:

| Region Type | Absolute Temp Threshold | Departure from Normal | Severe Departure Threshold |
| :--- | :---: | :---: | :---: |
| **Plains** (Delhi, Pune, Bengaluru) | $\ge 40.0^\circ\text{C}$ | $+4.5^\circ\text{C}$ | $+6.5^\circ\text{C}$ |
| **Coastal** (Mumbai) | $\ge 37.0^\circ\text{C}$ | $+4.5^\circ\text{C}$ | $+6.5^\circ\text{C}$ |
| **Hilly** (Shimla) | $\ge 30.0^\circ\text{C}$ | $+4.5^\circ\text{C}$ | $+6.5^\circ\text{C}$ |

> **IMD Heatwave Condition**: A heatwave is declared if the normal maximum temperature of a station is $> 40^\circ\text{C}$ and departure is $\ge 4.5^\circ\text{C}$, or when actual maximum temperature reaches $\ge 45^\circ\text{C}$ (plains).

---

## Machine Learning Pipeline

### Regression: Maximum Temperature Forecast
- **Model**: `RandomForestRegressor(n_estimators=100, random_state=42)`
- **Target**: Next-day maximum temperature (`target_temperature_max`)
- **Key Features**: 
  - Autoregressive lags: `temp_max_lag_1`, `temp_max_lag_2`, `temp_max_lag_3`
  - Rolling aggregates: `temp_max_rolling_3`, `temp_max_rolling_7`
  - Microclimate factors: `humidity_lag_1`, `wind_lag_1`, `precip_3d`, `temp_trend_3d`
  - Climatology: `departure`, `normal_max_temp`, `month`, `day_of_year`, `latitude`, `longitude`

**Benchmark Performance:**
| Dataset Split | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Validation Set** | $0.96^\circ\text{C}$ | $1.33^\circ\text{C}$ | **0.9636** |
| **Test Set** | $1.05^\circ\text{C}$ | $1.42^\circ\text{C}$ | **0.9628** |

### Classification: Heatwave Occurrence & Severity
- **Occurrence Model**: Binary `RandomForestClassifier` predicting whether heatwave criteria will be breached.
  - **Test ROC-AUC**: **0.927**
  - **Test Precision**: **0.909**
- **Severity Model**: Multi-class `RandomForestClassifier` assessing degree of event:
  - `0`: Normal
  - `1`: Heatwave
  - `2`: Severe Heatwave
  - **Weighted F1-Score**: **0.980**

### Explainable AI (SHAP)
Using `shap.TreeExplainer`, the system computes local attribution values for every forecast:
- Quantifies positive/negative contribution ($^\circ\text{C}$) of each meteorological feature.
- Generates `visualizations/shap_summary.png` and local JSON explanations for client-side waterfall plots.
- Enables operational transparency and prevents "black-box" alerting failures.

---

## Composite Heat Risk Scoring

The **Composite Heat Risk Score (CHRS)** scales from **0 to 100**, synthesizing five critical dimensions:

$$\text{CHRS} = 0.40 \cdot S_{\text{temp}} + 0.25 \cdot S_{\text{prob}} + 0.20 \cdot S_{\text{anomaly}} + 0.10 \cdot S_{\text{persist}} + 0.05 \cdot S_{\text{humidity}}$$

| Score Range | Risk Category | Advisory Level | Action Recommended |
| :---: | :---: | :---: | :--- |
| **0 – 19** | **Low** | Green | Normal outdoor activities permitted. Standard hydration. |
| **20 – 39** | **Moderate** | Yellow | Monitor conditions; limit intense midday sun exposure. |
| **40 – 59** | **High** | Orange | High alert for vulnerable populations; mandate shaded breaks for laborers. |
| **60 – 79** | **Very High** | Red | Avoid outdoor activity from 11 AM - 4 PM; active hydration centers required. |
| **80 – 100** | **Extreme** | Dark Red / Emergency | Health emergency; severe heat stroke risk; suspend heavy outdoor labor. |

---

## Project Structure

```plaintext
heatwave-intelligence/
├── backend/
│   └── main.py                     # FastAPI REST server & endpoint controllers
├── config/
│   ├── locations.csv               # Monitored locations (coordinates & terrain type)
│   └── thresholds.yaml             # Regional thresholds and risk weight definitions
├── data/
│   ├── raw/                        # Raw historical weather observations
│   └── processed/                  # Cleaned, labeled, and feature-engineered datasets
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Interactive React dashboard
│   │   ├── App.css                 # Custom glassmorphic & responsive styling
│   │   └── main.jsx                # Application root entrypoint
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── temperature_model.joblib    # Trained Random Forest Regressor
│   ├── heatwave_model.joblib       # Trained Heatwave Occurrence Classifier
│   ├── severity_model.joblib       # Trained Severity Classifier
│   ├── temperature_features.joblib # Regression feature list
│   └── classification_features.joblib
├── outputs/
│   ├── advisories/                 # Generated public health advisories
│   ├── explanations/               # Local SHAP JSON outputs
│   ├── metrics/                    # Regression & classification validation metrics
│   └── predictions/                # Latest forecast results (forecast_results.csv)
├── src/
│   ├── advisory.py                 # Public health advisory logic
│   ├── data_cleaning.py            # Missing value handling & quality checks
│   ├── data_download.py            # Weather data ingestion utilities
│   ├── explain.py                  # SHAP explanation pipeline
│   ├── feature_engineering.py      # Rolling windows, lags, and meteorological indicators
│   ├── heatwave_rules.py           # IMD threshold labeling & normal baseline calculation
│   ├── predict.py                  # End-to-end inference engine
│   ├── risk_score.py               # Composite heat risk computation
│   ├── train_heatwave.py           # Heatwave classification training
│   ├── train_temperature.py        # Temperature regression training
│   ├── validate.py                 # Model validation & cross-validation checks
│   └── visualization.py            # Static charts and HTML maps generator
├── tests/
│   └── test_pipeline.py            # End-to-end unit and integration test suite
├── visualizations/                 # Generated figures, HTML maps, and SHAP plots
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Monitored Locations

| Location ID | City | State | Latitude | Longitude | Region Type |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `PUNE_01` | Pune | Maharashtra | 18.5204 | 73.8567 | Plains |
| `MUM_01` | Mumbai | Maharashtra | 19.0760 | 72.8777 | Coastal |
| `DEL_01` | Delhi | Delhi | 28.6139 | 77.2090 | Plains |
| `BENG_01` | Bengaluru | Karnataka | 12.9716 | 77.5946 | Plains |
| `SHIM_01` | Shimla | Himachal Pradesh | 31.1048 | 77.1734 | Hilly |

---

## Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & **npm**

### 1. Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/heatwave-intelligence.git
cd heatwave-intelligence

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline

Execute the pipeline to regenerate processed features, retrain models, and produce latest forecasts:

```bash
# 1. Clean raw weather observations
python -m src.data_cleaning

# 2. Label heatwave occurrences using IMD terrain rules
python -m src.heatwave_rules

# 3. Compute time-series lags and rolling features
python -m src.feature_engineering

# 4. Train regression and classification models
python -m src.train_temperature
python -m src.train_heatwave

# 5. Generate forecasts, risk scores, and SHAP explanations
python -m src.predict
python -m src.explain
```

### 3. Start the FastAPI Backend

```bash
# Launch FastAPI with auto-reload
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 4. Start the React Frontend

Open a new terminal session:

```bash
cd frontend

# Install UI dependencies
npm install

# Start development server
npm run dev
```
Open your browser at **`http://localhost:5173`** to access the HeatWatch AI Dashboard.

---

## API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API metadata, version, and route inventory |
| `GET` | `/api/health` | Health check verifying model & data artifact availability |
| `GET` | `/api/locations` | List of all monitored geographic locations |
| `GET` | `/api/predictions` | Latest forecasts, heatwave classifications, and risk scores |
| `GET` | `/api/hotspots` | Prioritized risk hotspots ranked by composite risk index |
| `GET` | `/api/locations/{location_id}` | Detailed forecast, weather metrics, and advisory for a specific station |
| `GET` | `/api/explanations` | Global feature importance rankings from SHAP |
| `GET` | `/api/explanations/{location_id}` | Local SHAP attribution breakdown for a specific station |

### Sample Response: `GET /api/hotspots`
```json
[
  {
    "location_id": "DEL_01",
    "city": "Delhi",
    "region_type": "plains",
    "predicted_temperature": 42.4,
    "departure": 5.1,
    "heatwave_predicted": true,
    "severity": "Heatwave",
    "risk_score": 78.4,
    "risk_level": "Very High",
    "alert_color": "#DC2626"
  }
]
```

---

## Testing & Quality Assurance

HeatWatch AI includes an automated test suite verifying dataset integrity, model persistence, forecast generation, and API schema compliance:

```bash
# Run pytest test suite
pytest tests/test_pipeline.py -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
