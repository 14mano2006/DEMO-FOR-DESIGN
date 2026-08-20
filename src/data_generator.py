import numpy as np
import pandas as pd
import os
import random

VEHICLE_MODELS = [
    {"model": "Tesla Model Y Long Range", "chemistry": "NMC 811", "capacity_kwh": 78.1},
    {"model": "Ford F-150 Lightning", "chemistry": "NMC 622", "capacity_kwh": 131.0},
    {"model": "Rivian Commercial Van (EDV)", "chemistry": "LFP", "capacity_kwh": 100.0},
    {"model": "Hyundai Ioniq 5 AWD", "chemistry": "NMC 811", "capacity_kwh": 77.4},
    {"model": "Nissan Leaf e+", "chemistry": "NMC 622", "capacity_kwh": 62.0},
    {"model": "Volvo EX90 Twin Motor", "chemistry": "NMC 811", "capacity_kwh": 107.0}
]

def temp_penalty_factor(temp):
    """Calculates non-linear battery stress factor based on operating temperature."""
    if temp < 15.0:
        return 1.3 # Lithium plating risk at low temps
    elif temp <= 32.0:
        return 1.0 # Optimal operating window
    else:
        return 1.0 + 0.05 * (temp - 32.0) ** 1.3 # Accelerated Arrhenius SEI layer growth

def generate_fleet_telemetry(num_vehicles=50, max_cycles=500, seed=42):
    """
    Generates synthetic EV fleet battery telemetry dataset based on electro-thermal degradation physics.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []

    for v_idx in range(1, num_vehicles + 1):
        v_id = f"EV-{100 + v_idx}"
        v_spec = VEHICLE_MODELS[(v_idx - 1) % len(VEHICLE_MODELS)]
        model_name = v_spec["model"]
        chemistry = v_spec["chemistry"]
        capacity_kwh = v_spec["capacity_kwh"]

        # Base driver behavior profile per vehicle
        base_fast_charge_ratio = np.random.uniform(0.1, 0.75)
        base_dod = np.random.uniform(0.4, 0.90)
        base_temp_offset = np.random.uniform(-5.0, 10.0) # Climate region effect
        driving_aggressiveness = np.random.uniform(0.7, 1.4)

        # Vehicle specific operational age
        min_cycles = min(100, max_cycles)
        vehicle_max_cycle = np.random.randint(min_cycles, max_cycles + 1)

        # Base internal resistance (mOhm)
        ir_baseline = 1.2 if chemistry == "LFP" else 1.4

        # Initial SoH
        current_soh = 100.0

        for cycle in range(1, vehicle_max_cycle + 1):
            # Operational parameters with slight cycle-to-cycle variation
            fast_charge_ratio = np.clip(base_fast_charge_ratio + np.random.normal(0, 0.05), 0.05, 0.95)
            dod = np.clip(base_dod + np.random.normal(0, 0.03), 0.2, 0.98)
            
            # Temperature profile (°C)
            ambient_temp = 22.0 + base_temp_offset + np.random.normal(0, 3.0)
            heat_buildup = (fast_charge_ratio * 12.0) + (driving_aggressiveness * 5.0) + (cycle * 0.005)
            avg_temp = float(np.clip(ambient_temp + heat_buildup, 15.0, 52.0))
            max_temp = float(avg_temp + np.random.uniform(4.0, 14.0))

            # C-Rate (Charge/Discharge intensity)
            c_rate = float(np.clip(0.6 * driving_aggressiveness + (fast_charge_ratio * 1.5), 0.3, 2.8))

            # Degradation factors calculation
            # 1. Temperature factor
            t_penalty = temp_penalty_factor(avg_temp)

            # 2. Fast charging stress
            fast_charge_stress = 1.0 + 1.3 * (fast_charge_ratio ** 1.8)

            # 3. Depth of Discharge stress
            dod_stress = (dod / 0.8) ** 1.4

            # 4. Chemistry resistance factor
            chem_factor = 0.65 if chemistry == "LFP" else 1.0

            # Incremental SoH degradation for this cycle
            base_loss_per_cycle = (0.025 + 0.015 / np.sqrt(cycle)) * chem_factor
            cycle_soh_loss = base_loss_per_cycle * t_penalty * fast_charge_stress * dod_stress

            # Add occasional severe thermal anomaly
            is_anomaly = False
            if np.random.rand() < 0.015: # 1.5% chance of anomaly event
                max_temp += float(np.random.uniform(8.0, 18.0))
                cycle_soh_loss *= 2.5
                is_anomaly = True

            current_soh -= cycle_soh_loss
            current_soh = max(55.0, min(100.0, current_soh))

            # Internal resistance increase (mOhm)
            ir_growth = ir_baseline + (100.0 - current_soh) * 0.035 + (1 if max_temp > 45 else 0) * 0.4
            
            # Cell voltage variance (cell string imbalance)
            cell_variance = 0.004 + (100.0 - current_soh) * 0.00045 + (fast_charge_ratio * 0.012)
            if is_anomaly:
                cell_variance += 0.025

            # Odometer calculation (~40 km per cycle)
            odometer_km = int(cycle * np.random.uniform(38.0, 44.0))

            # Cumulative kWh throughput
            kwh_throughput = cycle * capacity_kwh * dod * 1.85

            # Remaining Useful Life (cycles until SoH = 70%)
            if current_soh > 70.0:
                avg_degradation_rate = (100.0 - current_soh) / cycle
                rul_cycles = int((current_soh - 70.0) / max(0.005, avg_degradation_rate))
            else:
                rul_cycles = 0

            # Telemetry record
            records.append({
                "vehicle_id": v_id,
                "model": model_name,
                "chemistry": chemistry,
                "capacity_kwh": capacity_kwh,
                "cycle_number": cycle,
                "odometer_km": odometer_km,
                "avg_temperature_c": round(avg_temp, 2),
                "max_temperature_c": round(max_temp, 2),
                "fast_charge_ratio": round(fast_charge_ratio, 3),
                "depth_of_discharge": round(dod, 3),
                "c_rate": round(c_rate, 2),
                "internal_resistance_mohm": round(ir_growth, 3),
                "cell_voltage_variance": round(cell_variance, 4),
                "cumulative_kwh_throughput": round(kwh_throughput, 1),
                "state_of_health": round(current_soh, 2),
                "remaining_useful_life_cycles": rul_cycles,
                "is_anomaly": 1 if is_anomaly or max_temp > 49.0 or cell_variance > 0.045 else 0
            })

    df = pd.DataFrame(records)
    return df

def get_latest_fleet_status(df):
    """Filters dataset to return only the latest telemetry snapshot for each vehicle."""
    latest_df = df.sort_values("cycle_number").groupby("vehicle_id").last().reset_index()
    return latest_df

def save_default_datasets(data_dir="data"):
    """Generates and saves synthetic fleet telemetry data to CSV."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "fleet_telemetry.csv")
    if not os.path.exists(csv_path):
        df = generate_fleet_telemetry(num_vehicles=50, max_cycles=500, seed=42)
        df.to_csv(csv_path, index=False)
        print(f"Generated default fleet telemetry dataset with {len(df)} records at {csv_path}")
    return csv_path

if __name__ == "__main__":
    df = generate_fleet_telemetry()
    print("Generated dataset sample:")
    print(df.head())
