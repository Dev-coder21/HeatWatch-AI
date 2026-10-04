"""Load the trained models once and expose SHAP explainers lazily."""

from functools import lru_cache

import joblib

from src.config import project_path

MODEL_DIR = project_path("models")


class ModelsUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_models():
    try:
        return {
            "temperature": joblib.load(MODEL_DIR / "temperature_model.joblib"),
            "temperature_features": joblib.load(MODEL_DIR / "temperature_features.joblib"),
            "heatwave": joblib.load(MODEL_DIR / "heatwave_model.joblib"),
            "severity": joblib.load(MODEL_DIR / "severity_model.joblib"),
            "classification_features": joblib.load(MODEL_DIR / "classification_features.joblib"),
        }
    except FileNotFoundError as error:
        raise ModelsUnavailable(
            f"{error.filename} missing. Run: python src/train_temperature.py && python src/train_heatwave.py"
        ) from error


@lru_cache(maxsize=2)
def explainer(name):
    import shap

    return shap.TreeExplainer(load_models()[name])
