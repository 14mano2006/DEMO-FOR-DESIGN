import os
import json
import joblib
import numpy as np
import pandas as pd
from src.model_trainer import prepare_features, train_and_save_all, FEATURE_COLS
from src.data_generator import generate_fleet_telemetry, temp_penalty_factor

class BatteryPredictor:
    def __init__(self, models_dir="models", data_dir="data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.soh_model = None
        self.rul_model = None
        self.anomaly_model = None
        self.metrics = {}
        self._load_or_train()

    def _load_or_train(self):
        """Loads trained models or triggers automated training pipeline if binaries are missing."""
        soh_path = os.path.join(self.models_dir, "soh_model.pkl")
        rul_path = os.path.join(self.models_dir, "rul_model.pkl")
        anomaly_path = os.path.join(self.models_dir, "anomaly_model.pkl")
        metrics_path = os.path.join(self.models_dir, "model_metrics.json")

        if not (os.path.exists(soh_path) and os.path.exists(rul_path) and os.path.exists(anomaly_path)):
            print("Models not found. Generating default telemetry and training new AI models...")
            df = generate_fleet_telemetry(num_vehicles=50, max_cycles=500, seed=42)
            os.makedirs(self.data_dir, exist_ok=True)
            df.to_csv(os.path.join(self.data_dir, "fleet_telemetry.csv"), index=False)
            train_and_save_all(df, self.models_dir)

        self.soh_model = joblib.load(soh_path)
        self.rul_model = joblib.load(rul_path)
        self.anomaly_model = joblib.load(anomaly_path)

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

    def predict_soh(self, df):
        """Predicts State of Health (SoH %) for input telemetry dataframe."""
        X = prepare_features(df)
        preds = self.soh_model.predict(X)
        return np.clip(preds, 50.0, 100.0)

    def predict_rul(self, df):
        """Predicts Remaining Useful Life (cycles) for input telemetry dataframe."""
        X = prepare_features(df)
        preds = self.rul_model.predict(X)
        return np.maximum(0, np.round(preds).astype(int))

    def detect_anomalies(self, df):
        """Predicts anomaly status (0: Normal, 1: Anomaly) and probability."""
        X = prepare_features(df)
        preds = self.anomaly_model.predict(X)
        probs = self.anomaly_model.predict_proba(X)[:, 1] if hasattr(self.anomaly_model, "predict_proba") else preds
        return preds, probs

    def predict_future_degradation(self, start_cycle, current_soh, chemistry="NMC 811", capacity_kwh=78.1,
                                   future_cycles=250, custom_params=None):
        """
        Simulates future battery degradation trajectory under custom or baseline operating parameters.
        Returns DataFrame with projected cycle numbers, predicted SoH, and RUL.
        """
        params = {
            "fast_charge_ratio": 0.35,
            "avg_temperature_c": 28.0,
            "max_temperature_c": 35.0,
            "depth_of_discharge": 0.75,
            "c_rate": 1.0,
            "driving_aggressiveness": 1.0
        }
        if custom_params:
            params.update(custom_params)

        future_records = []
        sim_soh = current_soh

        for step in range(1, future_cycles + 1):
            cycle = start_cycle + step
            
            # Electro-thermal penalty
            t_pen = temp_penalty_factor(params["avg_temperature_c"])
            fc_stress = 1.0 + 1.3 * (params["fast_charge_ratio"] ** 1.8)
            dod_stress = (params["depth_of_discharge"] / 0.8) ** 1.4
            chem_factor = 0.65 if "LFP" in chemistry else 1.0

            loss = (0.025 + 0.015 / np.sqrt(cycle)) * chem_factor * t_pen * fc_stress * dod_stress
            sim_soh -= loss
            sim_soh = max(50.0, sim_soh)

            ir = 1.4 + (100.0 - sim_soh) * 0.035
            cell_var = 0.004 + (100.0 - sim_soh) * 0.00045 + (params["fast_charge_ratio"] * 0.012)
            kwh_tp = cycle * capacity_kwh * params["depth_of_discharge"] * 1.85

            row = {
                "cycle_number": cycle,
                "odometer_km": int(cycle * 41.0),
                "avg_temperature_c": params["avg_temperature_c"],
                "max_temperature_c": params["max_temperature_c"],
                "fast_charge_ratio": params["fast_charge_ratio"],
                "depth_of_discharge": params["depth_of_discharge"],
                "c_rate": params["c_rate"],
                "internal_resistance_mohm": round(ir, 3),
                "cell_voltage_variance": round(cell_var, 4),
                "cumulative_kwh_throughput": round(kwh_tp, 1)
            }
            future_records.append(row)

        df_fut = pd.DataFrame(future_records)
        df_fut["predicted_soh"] = self.predict_soh(df_fut)
        df_fut["predicted_rul"] = self.predict_rul(df_fut)
        return df_fut
