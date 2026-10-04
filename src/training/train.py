"""Phase 3: train and evaluate the Tmax corrector and heatwave classifiers.

Splits (by target date):
  train 2021-04 .. 2023-12, validate 2024, test 2025-01 .. 2026-09
plus held-out sites (~20%, stratified by terrain, never trained on) and a
leave-one-region-out (LORO) evaluation where a whole climate zone is held out.

Regression (Model Output Statistics): predict observed Tmax as
  raw GFS forecast + learned residual. Baselines: persistence, raw GFS, raw ECMWF.
Classification: observed IMD heatwave on the target day; calibrated on validation,
decision threshold chosen on validation (max F1). Baselines: IMD rules applied to
the raw GFS forecast and to our corrected forecast.

    python -m src.training.train [--quick]
Writes models/*.joblib, models/metadata.json, outputs/metrics/phase3_*.json/csv.
"""

import argparse
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)

from src.config import project_path, settings
from src.heatwave_rules import classify
from src.training.dataset import CLASSIFIER_FEATURES, FEATURES

FEATURES_FILE = project_path("data/processed/features.parquet")
MODEL_DIR = project_path("models")
METRICS_DIR = project_path("outputs/metrics")

TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
TEST_END = "2026-09-30"
HOLDOUT_FRACTION = 0.2
SEED = 42
MODEL_VERSION = "2.0.0"


# -----------------------------------
# Data and splits
# -----------------------------------

def load():
    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= TEST_END]
    df["period"] = np.select(
        [df["date"] <= TRAIN_END, df["date"] <= VAL_END], ["train", "val"], default="test"
    )
    return df


def holdout_sites(df):
    """~20% of sites per terrain, fixed seed, never cherry-picked."""
    rng = np.random.default_rng(SEED)
    sites = df[["site_id", "terrain_type"]].drop_duplicates()
    held = []
    for _, group in sites.groupby("terrain_type"):
        ids = sorted(group["site_id"])
        n = max(1, round(len(ids) * HOLDOUT_FRACTION))
        held += list(rng.choice(ids, size=n, replace=False))
    return sorted(held)


# -----------------------------------
# Models
# -----------------------------------

def regressors(quick):
    return {
        "hgb": HistGradientBoostingRegressor(
            max_iter=300 if quick else 800, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0, random_state=SEED,
        ),
        "rf": RandomForestRegressor(
            n_estimators=60 if quick else 200, max_depth=18, min_samples_leaf=20,
            max_features=0.5, n_jobs=-1, random_state=SEED,
        ),
    }


def classifiers(quick):
    return {
        "hgb": HistGradientBoostingClassifier(
            max_iter=200 if quick else 500, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, class_weight="balanced", random_state=SEED,
        ),
        "rf": RandomForestClassifier(
            n_estimators=60 if quick else 200, max_depth=16, min_samples_leaf=20,
            max_features=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
    }


def subsample(frame, n):
    return frame if len(frame) <= n else frame.sample(n, random_state=SEED)


# -----------------------------------
# Metrics
# -----------------------------------

def reg_metrics(y, pred):
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0}
    return {
        "mae": round(float(mean_absolute_error(y[mask], pred[mask])), 3),
        "rmse": round(float(root_mean_squared_error(y[mask], pred[mask])), 3),
        "n": int(mask.sum()),
    }


def cls_metrics(y, prob=None, pred=None):
    y = np.asarray(y)
    out = {"n": int(len(y)), "positives": int(y.sum())}
    if pred is not None:
        out.update(
            recall=round(float(recall_score(y, pred, zero_division=0)), 3),
            precision=round(float(precision_score(y, pred, zero_division=0)), 3),
            f1=round(float(f1_score(y, pred, zero_division=0)), 3),
            tp=int(((pred == 1) & (y == 1)).sum()),
            fp=int(((pred == 1) & (y == 0)).sum()),
            fn=int(((pred == 0) & (y == 1)).sum()),
        )
    if prob is not None and y.sum() > 0:
        out.update(
            pr_auc=round(float(average_precision_score(y, prob)), 3),
            brier=round(float(brier_score_loss(y, prob)), 4),
            base_rate=round(float(y.mean()), 4),
        )
    return out


def reliability(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    table = []
    for b in range(bins):
        mask = idx == b
        if mask.sum():
            table.append({
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": round(float(prob[mask].mean()), 3),
                "observed_rate": round(float(y[mask].mean()), 3),
            })
    return table


def best_threshold(y, prob):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, prob >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# -----------------------------------
# Training steps
# -----------------------------------

def fit_corrector(train, val, quick):
    """Pick the residual model by validation MAE."""
    results, fitted = {}, {}
    for name, model in regressors(quick).items():
        sample = subsample(train, 300_000) if name == "rf" else train
        model.fit(sample[FEATURES], (sample["target_tmax"] - sample["fc_tmax"]))
        pred = val["fc_tmax"].values + model.predict(val[FEATURES])
        results[name] = reg_metrics(val["target_tmax"].values, pred)
        fitted[name] = model
        print(f"  corrector {name}: val {results[name]}", flush=True)
    best = min(results, key=lambda k: results[k]["mae"])
    return best, fitted[best], results


def corrected(model, frame):
    return frame["fc_tmax"].values + model.predict(frame[FEATURES])


def fit_heatwave(train, val, quick):
    results, fitted = {}, {}
    for name, model in classifiers(quick).items():
        sample = subsample(train, 400_000) if name == "rf" else train
        model.fit(sample[CLASSIFIER_FEATURES], sample["heatwave"])
        prob = model.predict_proba(val[CLASSIFIER_FEATURES])[:, 1]
        results[name] = cls_metrics(val["heatwave"], prob=prob)
        fitted[name] = model
        print(f"  heatwave {name}: val {results[name]}", flush=True)
    best = max(results, key=lambda k: results[k].get("pr_auc", 0))
    # Calibrate the chosen model on validation (isotonic), then pick the threshold there.
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted[best]), method="isotonic")
    calibrated.fit(val[CLASSIFIER_FEATURES], val["heatwave"])
    prob = calibrated.predict_proba(val[CLASSIFIER_FEATURES])[:, 1]
    threshold = best_threshold(val["heatwave"].values, prob)
    return best, calibrated, threshold, results


def fit_severity(train, val, quick):
    results, fitted = {}, {}
    for name, model in classifiers(quick).items():
        sample = subsample(train, 400_000) if name == "rf" else train
        model.fit(sample[CLASSIFIER_FEATURES], sample["severity"])
        pred = model.predict(val[CLASSIFIER_FEATURES])
        results[name] = {"macro_f1": round(float(f1_score(val["severity"], pred, average="macro")), 3)}
        fitted[name] = model
        print(f"  severity {name}: val {results[name]}", flush=True)
    best = max(results, key=lambda k: results[k]["macro_f1"])
    return best, fitted[best], results


# -----------------------------------
# Evaluation
# -----------------------------------

def regression_table(frame, pred, baselines):
    """MAE/RMSE for our model and baselines, overall, per lead, per terrain."""
    y = frame["target_tmax"].values
    columns = {"corrected_gfs": pred, **baselines}
    rows = []
    for group_name, keys in [("overall", None), ("lead", "lead"), ("terrain", "terrain_type")]:
        groups = [("all", np.ones(len(frame), bool))] if keys is None else [
            (str(k), (frame[keys] == k).values) for k in sorted(frame[keys].unique())
        ]
        for label, mask in groups:
            for model_name, values in columns.items():
                m = reg_metrics(y[mask], np.asarray(values)[mask])
                rows.append({"group": group_name, "value": label, "model": model_name, **m})
    return rows


def regression_baselines(frame, ecmwf_col):
    out = {"persistence": frame["tmax_lag1"].values, "raw_gfs": frame["fc_tmax"].values}
    if ecmwf_col in frame:
        out["raw_ecmwf"] = frame[ecmwf_col].values
    return out


def classification_table(frame, prob, threshold, corrected_tmax):
    y = frame["heatwave"].values
    rule_raw = classify(frame["fc_tmax"], frame["normal_tmax"], frame["terrain_type"].values)["heatwave"].values
    rule_corr = classify(corrected_tmax, frame["normal_tmax"], frame["terrain_type"].values)["heatwave"].values
    rows = []
    for group_name, keys in [("overall", None), ("terrain", "terrain_type"), ("lead", "lead")]:
        groups = [("all", np.ones(len(frame), bool))] if keys is None else [
            (str(k), (frame[keys] == k).values) for k in sorted(frame[keys].unique())
        ]
        for label, mask in groups:
            rows.append({"group": group_name, "value": label, "model": "classifier",
                         **cls_metrics(y[mask], prob=prob[mask], pred=(prob[mask] >= threshold).astype(int))})
            rows.append({"group": group_name, "value": label, "model": "imd_rule_on_raw_gfs",
                         **cls_metrics(y[mask], pred=rule_raw[mask])})
            rows.append({"group": group_name, "value": label, "model": "imd_rule_on_corrected",
                         **cls_metrics(y[mask], pred=rule_corr[mask])})
    return rows


def evaluate_split(name, frame, corrector, heatwave, threshold, severity, ecmwf_col):
    pred = corrected(corrector, frame)
    prob = heatwave.predict_proba(frame[CLASSIFIER_FEATURES])[:, 1]
    sev = severity.predict(frame[CLASSIFIER_FEATURES])
    return {
        "split": name,
        "regression": regression_table(frame, pred, regression_baselines(frame, ecmwf_col)),
        "classification": classification_table(frame, prob, threshold, pred),
        "severity_macro_f1": round(float(f1_score(frame["severity"], sev, average="macro")), 3),
        "reliability": reliability(frame["heatwave"].values, prob),
    }


def loro(df, best_reg, best_cls, quick):
    """Hold out one climate zone at a time (train period -> test period)."""
    rows = []
    for zone in sorted(df["zone"].unique()):
        train = df[(df["period"] == "train") & (df["zone"] != zone)]
        test = df[(df["period"] == "test") & (df["zone"] == zone)]
        if test.empty:
            continue
        reg = regressors(quick)[best_reg]
        reg_train = subsample(train, 300_000) if best_reg == "rf" else train
        reg.fit(reg_train[FEATURES], reg_train["target_tmax"] - reg_train["fc_tmax"])
        cls = classifiers(quick)[best_cls]
        cls_train = subsample(train, 400_000) if best_cls == "rf" else train
        cls.fit(cls_train[CLASSIFIER_FEATURES], cls_train["heatwave"])
        pred = test["fc_tmax"].values + reg.predict(test[FEATURES])
        prob = cls.predict_proba(test[CLASSIFIER_FEATURES])[:, 1]
        y = test["target_tmax"].values
        row = {
            "zone": zone,
            "sites": int(test["site_id"].nunique()),
            "mae_corrected": reg_metrics(y, pred)["mae"],
            "mae_raw_gfs": reg_metrics(y, test["fc_tmax"].values)["mae"],
            "mae_persistence": reg_metrics(y, test["tmax_lag1"].values)["mae"],
            **{f"hw_{k}": v for k, v in cls_metrics(test["heatwave"], prob=prob, pred=(prob >= 0.5).astype(int)).items()},
        }
        rows.append(row)
        print(f"  LORO {zone}: {row}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller models, for development")
    parser.add_argument("--skip-loro", action="store_true")
    args = parser.parse_args()

    df = load()
    held = holdout_sites(df)
    df["site_split"] = np.where(df["site_id"].isin(held), "unseen", "seen")
    ecmwf_col = f"raw_{settings()['api']['baseline_models'][0]}"
    print(f"Rows: {len(df)}; held-out sites ({len(held)}): {held}", flush=True)

    seen = df[df["site_split"] == "seen"]
    train, val = seen[seen["period"] == "train"], seen[seen["period"] == "val"]
    print(f"Train {len(train)} rows, {int(train.heatwave.sum())} heatwave; val {len(val)} rows, {int(val.heatwave.sum())} heatwave", flush=True)

    best_reg, corrector, reg_val = fit_corrector(train, val, args.quick)
    best_cls, heatwave, threshold, cls_val = fit_heatwave(train, val, args.quick)
    best_sev, severity, sev_val = fit_severity(train, val, args.quick)

    test = df[df["period"] == "test"]
    splits = [
        evaluate_split("test_seen_sites", test[test["site_split"] == "seen"], corrector, heatwave, threshold, severity, ecmwf_col),
        evaluate_split("test_unseen_sites", test[test["site_split"] == "unseen"], corrector, heatwave, threshold, severity, ecmwf_col),
        evaluate_split("val_seen_sites", val, corrector, heatwave, threshold, severity, ecmwf_col),
    ]
    loro_rows = [] if args.skip_loro else loro(df, best_reg, best_cls, args.quick)

    # Is the correction worth using over raw GFS on unseen sites?
    unseen_overall = {r["model"]: r["mae"] for r in splits[1]["regression"] if r["group"] == "overall"}
    use_correction = unseen_overall["corrected_gfs"] < unseen_overall["raw_gfs"]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "held_out_sites": held,
        "model_selection": {"corrector": reg_val, "heatwave": cls_val, "severity": sev_val,
                            "chosen": {"corrector": best_reg, "heatwave": best_cls, "severity": best_sev}},
        "threshold": threshold,
        "use_correction": bool(use_correction),
        "splits": splits,
        "loro": loro_rows,
    }
    (METRICS_DIR / "phase3_metrics.json").write_text(json.dumps(report, indent=2))

    joblib.dump(corrector, MODEL_DIR / "tmax_corrector.joblib", compress=3)
    joblib.dump(heatwave, MODEL_DIR / "heatwave_classifier.joblib", compress=3)
    joblib.dump(severity, MODEL_DIR / "severity_classifier.joblib", compress=3)
    metadata = {
        "version": MODEL_VERSION,
        "trained": report["generated"],
        "features": FEATURES,
        "classifier_features": CLASSIFIER_FEATURES,
        "forecast_model": settings()["api"]["forecast_model"],
        "train_period": [str(train["date"].min().date()), TRAIN_END],
        "validation_period": ["2024-01-01", VAL_END],
        "test_period": ["2025-01-01", TEST_END],
        "held_out_sites": held,
        "chosen": report["model_selection"]["chosen"],
        "heatwave_threshold": threshold,
        "use_correction": bool(use_correction),
        "summary": {s["split"]: {
            "regression_overall": [r for r in s["regression"] if r["group"] == "overall"],
            "classification_overall": [r for r in s["classification"] if r["group"] == "overall"],
            "severity_macro_f1": s["severity_macro_f1"],
        } for s in splits},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
