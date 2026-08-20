import random
from datetime import datetime, timedelta

class MaintenanceEngine:
    def __init__(self):
        pass

    def evaluate_vehicle_health(self, telemetry_row):
        """
        Evaluates a single vehicle's telemetry row and returns diagnostic analysis,
        risk category, recommendations, and failure urgency.
        """
        soh = float(telemetry_row.get("state_of_health", 90.0))
        rul = int(telemetry_row.get("remaining_useful_life_cycles", 300))
        max_temp = float(telemetry_row.get("max_temperature_c", 30.0))
        fc_ratio = float(telemetry_row.get("fast_charge_ratio", 0.3))
        ir = float(telemetry_row.get("internal_resistance_mohm", 1.5))
        cell_var = float(telemetry_row.get("cell_voltage_variance", 0.01))
        is_anomaly = bool(telemetry_row.get("is_anomaly", False))
        v_id = str(telemetry_row.get("vehicle_id", "EV-100"))
        model = str(telemetry_row.get("model", "EV Vehicle"))

        findings = []
        recommendations = []
        risk_score = 0 # 0 (Healthy) to 100 (Critical)

        # 1. State of Health (SoH) evaluation
        if soh < 75.0:
            risk_score += 45
            findings.append(f"Severe battery capacity degradation (SoH = {soh:.1f}% < 75% End-of-Life threshold).")
            recommendations.append("Schedule full traction battery pack diagnostic or module replacement.")
        elif soh < 80.0:
            risk_score += 30
            findings.append(f"Moderate degradation (SoH = {soh:.1f}% approaching warranty/service threshold).")
            recommendations.append("Flag vehicle for depot maintenance inspection within 14 days.")
        elif soh < 85.0:
            risk_score += 15
            findings.append(f"Mild degradation (SoH = {soh:.1f}%).")

        # 2. Thermal stress evaluation
        if max_temp > 48.0 or is_anomaly:
            risk_score += 35
            findings.append(f"Thermal runaway stress risk (Peak recorded temp = {max_temp:.1f}°C).")
            recommendations.append("Inspect thermal management coolant lines, pump flow rate, and radiator heat sink.")
        elif max_temp > 42.0:
            risk_score += 15
            findings.append(f"Elevated operating temperature (Peak temp = {max_temp:.1f}°C).")
            recommendations.append("Verify cooling fan actuation and limit high C-rate discharge operations.")

        # 3. Cell String Voltage Variance
        if cell_var > 0.040:
            risk_score += 25
            findings.append(f"High cell string voltage imbalance ({cell_var*1000:.1f} mV delta).")
            recommendations.append("Perform extended overnight passive/active cell re-balancing cycle.")
        elif cell_var > 0.025:
            risk_score += 10
            findings.append(f"Slight cell string imbalance ({cell_var*1000:.1f} mV delta).")

        # 4. Internal Resistance
        if ir > 3.2:
            risk_score += 20
            findings.append(f"Internal resistance growth ({ir:.2f} mΩ vs 1.4 mΩ baseline).")
            recommendations.append("Recalibrate BMS State of Charge (SoC) estimation algorithms.")

        # 5. Fast Charging Usage
        if fc_ratio > 0.65:
            risk_score += 15
            findings.append(f"Excessive DC Fast-Charging ratio ({fc_ratio*100:.0f}% of total charges).")
            recommendations.append("Enforce fleet policy: Cap DC fast charging to maximum 50% of monthly charge sessions.")

        risk_score = min(100, risk_score)

        # Categorize Severity Level
        if risk_score >= 60 or soh < 75.0 or is_anomaly:
            risk_level = "CRITICAL"
            color = "#FF4B4B"
        elif risk_score >= 35 or soh < 80.0:
            risk_level = "HIGH"
            color = "#FFA500"
        elif risk_score >= 15 or soh < 85.0:
            risk_level = "MEDIUM"
            color = "#F0D000"
        else:
            risk_level = "LOW"
            color = "#00CC96"

        if not findings:
            findings.append("Battery operates within nominal electro-thermal limits.")
        if not recommendations:
            recommendations.append("Continue standard operating and charging routine.")

        return {
            "vehicle_id": v_id,
            "model": model,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "badge_color": color,
            "soh": soh,
            "rul_cycles": rul,
            "findings": findings,
            "recommendations": recommendations
        }

    def generate_work_order(self, vehicle_eval):
        """Generates a structured maintenance work order for a vehicle requiring service."""
        v_id = vehicle_eval["vehicle_id"]
        level = vehicle_eval["risk_level"]
        
        # Priority mapping
        days_due = {"CRITICAL": 1, "HIGH": 5, "MEDIUM": 14, "LOW": 30}[level]
        due_date = (datetime.now() + timedelta(days=days_due)).strftime("%Y-%m-%d")
        
        downtime_hrs = {"CRITICAL": 12, "HIGH": 6, "MEDIUM": 3, "LOW": 1}[level]
        est_cost = {"CRITICAL": 3500, "HIGH": 1200, "MEDIUM": 450, "LOW": 150}[level]

        actions_str = " | ".join(vehicle_eval["recommendations"])

        return {
            "work_order_id": f"WO-{random.randint(10000, 99999)}",
            "vehicle_id": v_id,
            "model": vehicle_eval["model"],
            "severity": level,
            "risk_score": vehicle_eval["risk_score"],
            "current_soh": vehicle_eval["soh"],
            "due_date": due_date,
            "estimated_downtime_hrs": downtime_hrs,
            "estimated_cost_usd": est_cost,
            "action_plan": actions_str,
            "status": "PENDING_DISPATCH"
        }

    def process_fleet_maintenance(self, fleet_df):
        """Processes full fleet telemetry dataframe and returns prioritized maintenance list and summary."""
        latest_fleet = fleet_df.sort_values("cycle_number").groupby("vehicle_id").last().reset_index()

        evaluations = []
        work_orders = []

        for idx, row in latest_fleet.iterrows():
            eval_res = self.evaluate_vehicle_health(row)
            evaluations.append(eval_res)
            if eval_res["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM"]:
                wo = self.generate_work_order(eval_res)
                work_orders.append(wo)

        # Sort evaluations by risk score descending
        evaluations.sort(key=lambda x: x["risk_score"], reverse=True)
        work_orders.sort(key=lambda x: x["risk_score"], reverse=True)

        return evaluations, work_orders
