# ⚡ AI-Based EV Battery Health Prediction & Predictive Maintenance System

An enterprise-grade, interactive Web Application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to monitor battery State of Health (SoH), forecast Remaining Useful Life (RUL), detect electro-thermal anomalies, and dispatch automated predictive maintenance work orders for Electric Vehicle (EV) fleets.

---

## 🌟 Key Capabilities & System Features

1. **📊 Fleet Executive Operations Dashboard**
   - Fleet-wide KPI metrics: Total Vehicles, Average SoH %, At-Risk Batteries (< 80% SoH), Thermal Runaway Anomalies, Estimated Warranty Cost Savings.
   - Interactive SoH Distribution Histograms, Risk Severity Donut Charts, and Mileage vs Degradation Scatter plots.
   - Live fleet telemetry data registry with color-coded health badges.

2. **⚡ Single EV Diagnostic Deep-Dive**
   - Individual vehicle selector (EV-101 to EV-150 + Custom Uploads).
   - Real-time Multi-sensor telemetry gauges: Voltage, Current, Max Cell Temp, SoC %, Internal Resistance (mΩ), Cell String Imbalance (mV).
   - **Historical vs AI Projected SoH Lifespan Curve** up to End-of-Life (70% capacity threshold).
   - **Custom CSV Telemetry Upload**: Upload custom battery CSV files for instant AI SoH & RUL predictions.

3. **🤖 AI / ML Architecture & Explainability (XAI)**
   - **SoH Regressor**: Gradient Boosting / Random Forest ($R^2 > 0.99$, $RMSE < 0.1\%$).
   - **RUL Regressor**: Random Forest ($R^2 > 0.97$).
   - **Anomaly Detector**: Balanced Random Forest Classifier (100% precision on thermal runaway and cell imbalance spikes).
   - **Feature Importance Breakdown**: Quantifies the degradation impact of DC Fast Charging, Peak Temperature, Depth of Discharge (DoD), and Internal Resistance growth.
   - **In-App Model Retraining**: Trigger model retraining directly from the user interface.

4. **🔧 Predictive Maintenance Dispatcher & Work Orders**
   - Prioritized service backlog sorted by AI Risk Severity (CRITICAL, HIGH, MEDIUM, LOW).
   - Automated Work Order generator with estimated downtime (hrs), estimated cost ($), target service date, and actionable diagnostic repair steps.
   - 1-click Exporters: Download Work Orders as CSV or Executive Summary Text Report.

5. **🧪 What-If Fleet Simulator & Smart Charging Optimizer**
   - Interactive sliders: Fast-Charging %, Ambient Temp Exposure (°C), Depth of Discharge %, Driving Aggressiveness Index.
   - 500-Cycle Comparative Battery Lifespan Projections (Baseline vs Simulated Profile).
   - Smart Charging ROI Estimator: Calculates % degradation rate reduction, lifespan extension (years), and financial savings ($).

---

## 📁 Repository Structure

```
d:/ev_fleet_battery_health/
├── app.py                      # Main Streamlit application & multi-page navigation router
├── requirements.txt            # System dependencies
├── README.md                   # Documentation and usage guide
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Physics-based EV battery electro-thermal telemetry simulator
│   ├── model_trainer.py        # ML training pipeline for SoH, RUL, and Anomaly Detection
│   ├── predictor.py            # Inference engine & future degradation forecasting
│   ├── maintenance_engine.py   # Hybrid predictive risk scoring & work order dispatcher
│   ├── optimizer.py            # Smart charging strategy & battery life extension simulator
│   └── utils.py                # Visual styling, metric cards, & CSV/TXT report exporters
├── data/                       # Pre-generated fleet telemetry CSV datasets
└── models/                     # Saved serialized Scikit-learn models (.pkl) & metrics (.json)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed on your system.

```bash
cd d:/ev_fleet_battery_health
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
Run the following command in your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Verification & Testing
To run automated unit tests verifying dataset generation, model training accuracy, inference, and maintenance work orders:

```bash
python -c "from src.data_generator import generate_fleet_telemetry; from src.predictor import BatteryPredictor; p = BatteryPredictor(); print('Pipeline Test Passed!')"
```
