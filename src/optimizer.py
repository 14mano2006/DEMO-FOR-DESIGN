import numpy as np
import pandas as pd

class SmartChargingOptimizer:
    def __init__(self):
        pass

    def optimize_charging_strategy(self, current_fc_ratio, current_avg_temp, current_max_dod, chemistry="NMC 811"):
        """
        Calculates optimized charging parameters and forecasts battery lifespan extension benefits.
        """
        # Optimized targets
        opt_fc_ratio = min(current_fc_ratio, 0.25) # Cap DC fast charge ratio at 25%
        opt_avg_temp = min(current_avg_temp, 26.0)  # Thermal management / night charging target (26°C)
        opt_dod = min(current_max_dod, 0.75)       # Limit DoD stress (charge to 80%, discharge to 20%)

        # Degradation rates comparison (relative scale)
        def calc_deg_rate(fc, temp, dod):
            t_pen = 1.0 + 0.05 * max(0, temp - 32.0) ** 1.3 if temp > 32 else (1.3 if temp < 15 else 1.0)
            fc_pen = 1.0 + 1.3 * (fc ** 1.8)
            dod_pen = (dod / 0.8) ** 1.4
            return t_pen * fc_pen * dod_pen

        unopt_rate = calc_deg_rate(current_fc_ratio, current_avg_temp, current_max_dod)
        opt_rate = calc_deg_rate(opt_fc_ratio, opt_avg_temp, opt_dod)

        degradation_reduction_pct = round(max(0.0, (1.0 - (opt_rate / unopt_rate))) * 100, 1)

        # Life extension (years and cycles)
        baseline_life_years = 6.0
        extended_life_years = round(baseline_life_years * (unopt_rate / opt_rate), 1)
        extended_life_years = min(12.0, max(baseline_life_years + 0.5, extended_life_years))

        extra_years = round(extended_life_years - baseline_life_years, 1)

        # Financial savings estimate ($15,000 average battery pack replacement)
        savings_per_vehicle = round(extra_years * 1850, 2)

        return {
            "current_fc_ratio": current_fc_ratio,
            "optimized_fc_ratio": opt_fc_ratio,
            "current_avg_temp": current_avg_temp,
            "optimized_avg_temp": opt_avg_temp,
            "current_dod": current_max_dod,
            "optimized_dod": opt_dod,
            "degradation_reduction_pct": degradation_reduction_pct,
            "baseline_life_years": baseline_life_years,
            "extended_life_years": extended_life_years,
            "extra_life_years": extra_years,
            "estimated_savings_usd": savings_per_vehicle,
            "recommendations": [
                f"Cap DC fast-charging to max 25% of total sessions (currently {current_fc_ratio*100:.0f}%).",
                f"Schedule heavy charging during off-peak thermal window (11 PM - 6 AM at ~25°C).",
                f"Enforce daily 80% SoC charging cap for routine urban routes."
            ]
        }
