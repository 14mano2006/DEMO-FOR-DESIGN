import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score

FEATURE_COLS = [
    "cycle_number",
    "odometer_km",
    "avg_temperature_c",
    "max_temperature_c",
    "fast_charge_ratio",
    "depth_of_discharge",
    "c_rate",
    "internal_resistance_mohm",
    "cell_voltage_variance",
    "cumulative_kwh_throughput"
]

def prepare_features(df):
    """Ensures feature columns exist and handles missing values."""
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLS].copy()

def train_soh_model(df):
    """Trains a Random Forest / Gradient Boosting Regressor for State of Health (SoH) prediction."""
    X = prepare_features(df)
    y = df["state_of_health"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=120, max_depth=6, random_state=42, learning_rate=0.08)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4)
    }

    # Feature importance
    importances = dict(zip(FEATURE_COLS, [round(float(x), 4) for x in model.feature_importances_]))

    return model, metrics, importances

def train_rul_model(df):
    """Trains a Random Forest Regressor for Remaining Useful Life (RUL) cycles estimation."""
    X = prepare_features(df)
    y = df["remaining_useful_life_cycles"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4)
    }

    importances = dict(zip(FEATURE_COLS, [round(float(x), 4) for x in model.feature_importances_]))

    return model, metrics, importances

def train_anomaly_model(df):
    """Trains a Random Forest Classifier / Isolation Forest for detecting thermal/voltage anomalies."""
    X = prepare_features(df)
    y = df["is_anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4)
    }

    importances = dict(zip(FEATURE_COLS, [round(float(x), 4) for x in model.feature_importances_]))

    return model, metrics, importances

def train_and_save_all(df, models_dir="models"):
    """Trains all AI models and serializes artifacts to disk."""
    os.makedirs(models_dir, exist_ok=True)

    print("Training SoH Prediction Model...")
    soh_model, soh_metrics, soh_importance = train_soh_model(df)
    joblib.dump(soh_model, os.path.join(models_dir, "soh_model.pkl"))

    print("Training RUL Estimation Model...")
    rul_model, rul_metrics, rul_importance = train_rul_model(df)
    joblib.dump(rul_model, os.path.join(models_dir, "rul_model.pkl"))

    print("Training Anomaly Detection Model...")
    anomaly_model, anomaly_metrics, anomaly_importance = train_anomaly_model(df)
    joblib.dump(anomaly_model, os.path.join(models_dir, "anomaly_model.pkl"))

    all_metrics = {
        "soh_model": {"metrics": soh_metrics, "feature_importance": soh_importance},
        "rul_model": {"metrics": rul_metrics, "feature_importance": rul_importance},
        "anomaly_model": {"metrics": anomaly_metrics, "feature_importance": anomaly_importance}
    }

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Successfully trained and saved all models to '{models_dir}'.")
    return all_metrics

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_generator import generate_fleet_telemetry

    df = generate_fleet_telemetry(30, 400)
    train_and_save_all(df)
