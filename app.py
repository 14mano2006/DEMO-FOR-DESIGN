import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import os
import json

from src.data_generator import generate_fleet_telemetry, save_default_datasets, get_latest_fleet_status
from src.predictor import BatteryPredictor
from src.maintenance_engine import MaintenanceEngine
from src.optimizer import SmartChargingOptimizer
from src.utils import inject_custom_css, render_metric_card, generate_csv_download, generate_work_orders_report_text

# Page Configuration
st.set_page_config(
    page_title="EV Battery Health AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS styling
inject_custom_css()

# Cache data loading & model initialization
@st.cache_resource
def get_predictor():
    return BatteryPredictor(models_dir="models", data_dir="data")

@st.cache_data(ttl=600)
def load_fleet_data():
    csv_path = os.path.join("data", "fleet_telemetry.csv")
    if not os.path.exists(csv_path):
        save_default_datasets("data")
    return pd.read_csv(csv_path)

# Initialize Predictor and Data
predictor = get_predictor()
fleet_df = load_fleet_data()
maint_engine = MaintenanceEngine()
optimizer = SmartChargingOptimizer()

# Sidebar Navigation & Controls
st.sidebar.markdown("<h2 style='color:#38BDF8;'>⚡ EV Fleet AI Control</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Fleet Executive Overview",
        "⚡ Single EV Diagnostics",
        "🤖 AI Model Intelligence",
        "🔧 Maintenance Dispatcher",
        "🧪 What-If Fleet Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Fleet Filters")

# Chemistry Filter
chemistries = ["All"] + sorted(list(fleet_df["chemistry"].unique()))
selected_chem = st.sidebar.selectbox("Filter Chemistry", chemistries)

# Model Filter
models_list = ["All"] + sorted(list(fleet_df["model"].unique()))
selected_model = st.sidebar.selectbox("Filter Vehicle Model", models_list)

# Apply global filters
filtered_df = fleet_df.copy()
if selected_chem != "All":
    filtered_df = filtered_df[filtered_df["chemistry"] == selected_chem]
if selected_model != "All":
    filtered_df = filtered_df[filtered_df["model"] == selected_model]

latest_fleet = get_latest_fleet_status(filtered_df)

# Evaluate maintenance statuses
evaluations, work_orders = maint_engine.process_fleet_maintenance(filtered_df)

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>⚡ AI Battery Health & Predictive Maintenance</h1>
    <p>Real-time State of Health (SoH) prediction, Remaining Useful Life (RUL) estimation, thermal anomaly detection, and automated maintenance dispatching for EV fleets.</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: FLEET EXECUTIVE OVERVIEW
# ==============================================================================
if page == "📊 Fleet Executive Overview":
    st.subheader("📊 Fleet Performance & Health Snapshot")

    # Calculate Fleet KPIs
    total_vehicles = len(latest_fleet)
    avg_soh = latest_fleet["state_of_health"].mean() if total_vehicles > 0 else 0.0
    at_risk_count = len(latest_fleet[latest_fleet["state_of_health"] < 80.0])
    critical_anomalies = len(latest_fleet[latest_fleet["is_anomaly"] == 1])
    est_savings = total_vehicles * 3400

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Vehicles", f"{total_vehicles}", "Active Fleet", "positive")
    with col2:
        render_metric_card("Fleet Avg SoH", f"{avg_soh:.1f}%", "-0.4% this month", "warning" if avg_soh < 85 else "positive")
    with col3:
        render_metric_card("At-Risk Batteries", f"{at_risk_count}", "SoH < 80%", "negative" if at_risk_count > 0 else "positive")
    with col4:
        render_metric_card("Thermal Anomalies", f"{critical_anomalies}", "Immediate action", "negative" if critical_anomalies > 0 else "positive")
    with col5:
        render_metric_card("Warranty Cost Saved", f"${est_savings:,.0f}", "AI prevention", "positive")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row 1
    c1, c2 = st.columns([6, 4])

    with c1:
        st.markdown("##### 📈 Fleet State of Health (SoH %) Distribution")
        fig_soh = px.histogram(
            latest_fleet,
            x="state_of_health",
            color="chemistry",
            nbins=25,
            marginal="box",
            title="State of Health Distribution across Fleet",
            labels={"state_of_health": "State of Health (SoH %)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_soh.add_vline(x=80.0, line_dash="dash", line_color="orange", annotation_text="Warranty Caution (80%)")
        fig_soh.add_vline(x=70.0, line_dash="dash", line_color="red", annotation_text="End-of-Life (70%)")
        fig_soh.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_soh, use_container_width=True)

    with c2:
        st.markdown("##### 🛡️ Fleet Risk Level Breakdown")
        risk_counts = pd.Series([e["risk_level"] for e in evaluations]).value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        
        color_map = {"CRITICAL": "#FF4B4B", "HIGH": "#FFA500", "MEDIUM": "#F0D000", "LOW": "#00CC96"}
        fig_pie = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map=color_map,
            hole=0.45,
            title="Fleet Risk Severity Tiers"
        )
        fig_pie.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Charts Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### ⚡ Battery Degradation vs Odometer Mileage")
        fig_scat = px.scatter(
            latest_fleet,
            x="odometer_km",
            y="state_of_health",
            color="fast_charge_ratio",
            size="capacity_kwh",
            hover_data=["vehicle_id", "model", "chemistry", "internal_resistance_mohm"],
            color_continuous_scale="Viridis",
            labels={"odometer_km": "Odometer (km)", "state_of_health": "SoH %", "fast_charge_ratio": "Fast Charge %"},
            title="SoH Decay by Mileage & DC Fast Charging Exposure"
        )
        fig_scat.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_scat, use_container_width=True)

    with c4:
        st.markdown("##### 🌡️ Max Cell Temperature vs Internal Resistance")
        fig_ir = px.scatter(
            latest_fleet,
            x="max_temperature_c",
            y="internal_resistance_mohm",
            color="state_of_health",
            symbol="chemistry",
            hover_data=["vehicle_id", "model"],
            color_continuous_scale="RdYlGn_r",
            labels={"max_temperature_c": "Peak Temp (°C)", "internal_resistance_mohm": "IR (mΩ)"},
            title="Thermal Build-up & Internal Resistance Aging Correlation"
        )
        fig_ir.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_ir, use_container_width=True)

    # Fleet Status Data Grid
    st.markdown("##### 📋 Fleet Vehicle Live Telemetry Registry")
    display_cols = ["vehicle_id", "model", "chemistry", "capacity_kwh", "cycle_number", "odometer_km",
                    "state_of_health", "remaining_useful_life_cycles", "max_temperature_c", "fast_charge_ratio", "is_anomaly"]
    
    st.dataframe(
        latest_fleet[display_cols].sort_values("state_of_health", ascending=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "state_of_health": st.column_config.ProgressColumn("SoH %", format="%.1f%%", min_value=50, max_value=100),
            "fast_charge_ratio": st.column_config.NumberColumn("Fast Charge Ratio", format="%.2f"),
            "remaining_useful_life_cycles": st.column_config.NumberColumn("RUL (Cycles)"),
            "is_anomaly": st.column_config.CheckboxColumn("Anomaly Flag")
        }
    )


# ==============================================================================
# PAGE 2: SINGLE EV DIAGNOSTICS DEEP-DIVE
# ==============================================================================
elif page == "⚡ Single EV Diagnostics":
    st.subheader("⚡ Individual Vehicle Telemetry & Health Deep-Dive")

    tab1, tab2 = st.tabs(["🚗 Existing Fleet Vehicle", "📤 Custom Telemetry Upload"])

    with tab1:
        v_list = sorted(list(fleet_df["vehicle_id"].unique()))
        selected_v_id = st.selectbox("Select Vehicle ID for Diagnostic Analysis", v_list, index=0)

        v_df = fleet_df[fleet_df["vehicle_id"] == selected_v_id].sort_values("cycle_number")
        v_latest = v_df.iloc[-1]

        # Vehicle Info Header
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        with col_a:
            st.metric("Vehicle ID / Model", v_latest["vehicle_id"], v_latest["model"])
        with col_b:
            st.metric("Chemistry", v_latest["chemistry"], f"{v_latest['capacity_kwh']} kWh")
        with col_c:
            st.metric("Current SoH", f"{v_latest['state_of_health']:.1f}%", delta=f"{v_latest['state_of_health'] - 100:.1f}%")
        with col_d:
            st.metric("Predicted RUL", f"{v_latest['remaining_useful_life_cycles']} Cycles", "~" + str(int(v_latest['remaining_useful_life_cycles']*41)) + " km")
        with col_e:
            st.metric("Internal Resistance", f"{v_latest['internal_resistance_mohm']:.2f} mΩ", f"Variance: {v_latest['cell_voltage_variance']*1000:.1f} mV")

        st.markdown("<br>", unsafe_allow_html=True)

        # Forecasting Degradation & Future Projection
        st.markdown("##### 📉 SoH Degradation Curve & AI Future Lifespan Projection")
        
        # Predict future trajectory line
        future_cycles = 250
        df_fut = predictor.predict_future_degradation(
            start_cycle=int(v_latest["cycle_number"]),
            current_soh=float(v_latest["state_of_health"]),
            chemistry=str(v_latest["chemistry"]),
            capacity_kwh=float(v_latest["capacity_kwh"]),
            future_cycles=future_cycles
        )

        fig_proj = gg.Figure()
        # Historical curve
        fig_proj.add_trace(gg.Scatter(
            x=v_df["cycle_number"],
            y=v_df["state_of_health"],
            mode="lines+markers",
            name="Historical SoH",
            line=dict(color="#00CC96", width=3)
        ))
        # Future predicted curve
        fig_proj.add_trace(gg.Scatter(
            x=df_fut["cycle_number"],
            y=df_fut["predicted_soh"],
            mode="lines",
            name="AI Projected SoH",
            line=dict(color="#38BDF8", width=3, dash="dash")
        ))
        # Threshold lines
        fig_proj.add_hline(y=80.0, line_dash="dot", line_color="orange", annotation_text="80% Caution Line")
        fig_proj.add_hline(y=70.0, line_dash="dot", line_color="red", annotation_text="70% EoL Line")

        fig_proj.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Charging / Operational Cycles",
            yaxis_title="State of Health (SoH %)",
            legend=dict(x=0.01, y=0.01)
        )
        st.plotly_chart(fig_proj, use_container_width=True)

        # Multi-sensor Time Series Plots
        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            st.markdown("##### 🌡️ Temperature & C-Rate History")
            fig_temp = px.line(
                v_df,
                x="cycle_number",
                y=["avg_temperature_c", "max_temperature_c", "c_rate"],
                title=f"Thermal & C-Rate Telemetry Stream for {selected_v_id}",
                labels={"cycle_number": "Cycle", "value": "Value"}
            )
            fig_temp.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_temp, use_container_width=True)

        with c_ts2:
            st.markdown("##### ⚡ Internal Resistance & Cell Balance Delta")
            fig_ir = px.line(
                v_df,
                x="cycle_number",
                y=["internal_resistance_mohm", "cell_voltage_variance"],
                title=f"Internal Resistance & String Variance for {selected_v_id}",
                labels={"cycle_number": "Cycle", "value": "Value"}
            )
            fig_ir.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_ir, use_container_width=True)

        # Individual Maintenance Evaluation Result
        single_eval = maint_engine.evaluate_vehicle_health(v_latest)
        st.markdown(f"""
        <div class="content-card">
            <h4>🔍 AI Diagnostic Evaluation Summary for {selected_v_id}</h4>
            <p><b>Risk Severity:</b> <span class="badge-{single_eval['risk_level'].lower()}">{single_eval['risk_level']} (Score: {single_eval['risk_score']}/100)</span></p>
            <p><b>Diagnostic Findings:</b></p>
            <ul>
                {"".join([f"<li>{f}</li>" for f in single_eval['findings']])}
            </ul>
            <p><b>Recommended Action Plan:</b></p>
            <ul>
                {"".join([f"<li>{r}</li>" for r in single_eval['recommendations']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("##### 📤 Upload Custom Vehicle Telemetry CSV")
        st.markdown("Upload a custom battery telemetry CSV file to run instant AI State of Health & Remaining Useful Life predictions.")

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded CSV with {len(custom_df)} records.")
            
            # Predict SoH & RUL
            custom_df["predicted_soh"] = predictor.predict_soh(custom_df)
            custom_df["predicted_rul"] = predictor.predict_rul(custom_df)
            custom_anom_preds, custom_anom_probs = predictor.detect_anomalies(custom_df)
            custom_df["anomaly_detected"] = custom_anom_preds

            st.markdown("##### 📊 Custom Data AI Inference Results")
            st.dataframe(custom_df.head(20), use_container_width=True)

            fig_custom = px.line(
                custom_df,
                y="predicted_soh",
                title="Predicted State of Health Trajectory for Uploaded Dataset",
                labels={"index": "Sample / Cycle", "predicted_soh": "Predicted SoH %"}
            )
            fig_custom.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_custom, use_container_width=True)


# ==============================================================================
# PAGE 3: AI MODEL INTELLIGENCE & EXPLAINABILITY
# ==============================================================================
elif page == "🤖 AI Model Intelligence":
    st.subheader("🤖 AI / ML Architecture & Feature Explainability (XAI)")

    metrics_data = predictor.metrics

    if metrics_data:
        m1, m2, m3 = st.columns(3)
        with m1:
            soh_m = metrics_data.get("soh_model", {}).get("metrics", {})
            st.markdown("##### 🎯 SoH Regressor (Gradient Boosting)")
            st.metric("R² Performance", f"{soh_m.get('r2_score', 0):.4f}")
            st.metric("Root Mean Sq Error (RMSE)", f"{soh_m.get('rmse', 0):.4f}%")
            st.metric("Mean Abs Error (MAE)", f"{soh_m.get('mae', 0):.4f}%")

        with m2:
            rul_m = metrics_data.get("rul_model", {}).get("metrics", {})
            st.markdown("##### ⏳ RUL Regressor (Random Forest)")
            st.metric("R² Performance", f"{rul_m.get('r2_score', 0):.4f}")
            st.metric("RMSE", f"{rul_m.get('rmse', 0):.2f} Cycles")
            st.metric("MAE", f"{rul_m.get('mae', 0):.2f} Cycles")

        with m3:
            anom_m = metrics_data.get("anomaly_model", {}).get("metrics", {})
            st.markdown("##### 🚨 Anomaly Classifier")
            st.metric("Accuracy Score", f"{anom_m.get('accuracy', 0)*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Importance Plots
    st.markdown("##### 🧠 Feature Importance & Degradation Drivers (Feature Weight Analysis)")
    
    soh_imp = metrics_data.get("soh_model", {}).get("feature_importance", {})
    if soh_imp:
        df_imp = pd.DataFrame({
            "Feature": list(soh_imp.keys()),
            "Importance Weight": list(soh_imp.values())
        }).sort_values("Importance Weight", ascending=True)

        fig_imp = px.bar(
            df_imp,
            x="Importance Weight",
            y="Feature",
            orientation="h",
            title="Key Drivers of Battery State-of-Health Degradation",
            color="Importance Weight",
            color_continuous_scale="Teal"
        )
        fig_imp.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_imp, use_container_width=True)

    # In-App Retraining Control Panel
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚡ Retrain AI Models with Current Data", expanded=False):
        st.write("Retrain the machine learning models directly from the Streamlit interface.")
        if st.button("🚀 Execute Model Retraining Pipeline"):
            with st.spinner("Retraining Random Forest & Gradient Boosting models..."):
                from src.model_trainer import train_and_save_all
                new_metrics = train_and_save_all(fleet_df, "models")
                st.success("Model retraining completed successfully!")
                st.json(new_metrics)


# ==============================================================================
# PAGE 4: PREDICTIVE MAINTENANCE DISPATCHER
# ==============================================================================
elif page == "🔧 Maintenance Dispatcher":
    st.subheader("🔧 Predictive Maintenance Dispatch & Work Orders Queue")

    st.markdown("##### 📋 Prioritized Fleet Service Backlog")

    col_btn1, col_btn2 = st.columns([3, 7])
    with col_btn1:
        st.metric("Pending Work Orders", len(work_orders), f"{len([w for w in work_orders if w['severity']=='CRITICAL'])} Critical")
    with col_btn2:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = generate_csv_download(pd.DataFrame(work_orders))
        st.download_button(
            label="📥 Export Work Orders CSV",
            data=csv_data,
            file_name="fleet_work_orders.csv",
            mime="text/csv"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    for idx, wo in enumerate(work_orders, 1):
        sev = wo["severity"]
        badge_class = f"badge-{sev.lower()}"
        
        st.markdown(f"""
        <div class="content-card">
            <div style="display:flex; justify-between; align-items:center;">
                <h3>Work Order #{wo['work_order_id']} - {wo['vehicle_id']} ({wo['model']})</h3>
            </div>
            <p><b>Severity Tier:</b> <span class="{badge_class}">{sev}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <b>Risk Score:</b> {wo['risk_score']}/100 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Target Service Date:</b> {wo['due_date']}</p>
            <p><b>Current Battery SoH:</b> {wo['current_soh']:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; <b>Est. Downtime:</b> {wo['estimated_downtime_hrs']} Hours &nbsp;&nbsp;|&nbsp;&nbsp; <b>Est. Cost:</b> ${wo['estimated_cost_usd']:,}</p>
            <p><b>Action Plan:</b> <code>{wo['action_plan']}</code></p>
        </div>
        """, unsafe_allow_html=True)

    # Text Report Generator
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 View & Download Full Executive Maintenance Report", expanded=False):
        rep_text = generate_work_orders_report_text(work_orders)
        st.text_area("Executive Summary Report", rep_text, height=300)
        st.download_button(
            label="📥 Download Executive Summary Text Report",
            data=rep_text,
            file_name="fleet_maintenance_report.txt",
            mime="text/plain"
        )


# ==============================================================================
# PAGE 5: WHAT-IF FLEET SIMULATOR
# ==============================================================================
elif page == "🧪 What-If Fleet Simulator":
    st.subheader("🧪 Fleet Operational Profile Simulator & Smart Charging Optimizer")
    st.write("Simulate how altering operational parameters (Fast Charging ratio, Ambient Temperature, DoD, driving style) impacts battery degradation velocity over 500 cycles.")

    c_sim1, c_sim2 = st.columns([4, 6])

    with c_sim1:
        st.markdown("##### 🎛️ Simulation Parameters")
        sim_fc = st.slider("DC Fast-Charge Ratio (%)", 0.0, 1.0, 0.50, 0.05)
        sim_temp = st.slider("Average Ambient / Operating Temp (°C)", 10.0, 48.0, 32.0, 1.0)
        sim_dod = st.slider("Daily Depth of Discharge (DoD %)", 0.20, 1.00, 0.80, 0.05)
        sim_drive = st.slider("Driver Aggressiveness Index", 0.5, 2.0, 1.1, 0.1)

        opt_result = optimizer.optimize_charging_strategy(sim_fc, sim_temp, sim_dod)

    with c_sim2:
        st.markdown("##### 💡 Smart Charging Optimization ROI")
        st.success(f"**Degradation Rate Reduction:** {opt_result['degradation_reduction_pct']}% slower battery capacity loss")
        st.info(f"**Lifespan Extension:** +{opt_result['extra_life_years']} Years (Extended from {opt_result['baseline_life_years']} yrs to {opt_result['extended_life_years']} yrs)")
        st.warning(f"**Estimated Financial Savings:** ${opt_result['estimated_savings_usd']:,.2f} per vehicle")

        st.markdown("##### 💡 Recommended Charging Policy Rules:")
        for rec in opt_result["recommendations"]:
            st.markdown(f"- {rec}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparative Lifespan Curves
    st.markdown("##### 📈 Baseline vs Simulated Battery Lifespan Projection")

    # Generate baseline vs simulated future curves
    cycles = np.arange(1, 501)
    
    # Baseline curve
    base_loss = (0.025 + 0.015 / np.sqrt(cycles)) * (1.0 + 0.05 * max(0, 25 - 32) ** 1.3) * (1.0 + 1.3 * (0.3 ** 1.8)) * ((0.7 / 0.8) ** 1.4)
    base_soh = np.clip(100.0 - np.cumsum(base_loss), 50.0, 100.0)

    # Simulated curve
    sim_t_pen = 1.0 + 0.05 * max(0, sim_temp - 32.0) ** 1.3 if sim_temp > 32 else (1.3 if sim_temp < 15 else 1.0)
    sim_fc_pen = 1.0 + 1.3 * (sim_fc ** 1.8)
    sim_dod_pen = (sim_dod / 0.8) ** 1.4
    sim_loss = (0.025 + 0.015 / np.sqrt(cycles)) * sim_t_pen * sim_fc_pen * sim_dod_pen * sim_drive
    simulated_soh = np.clip(100.0 - np.cumsum(sim_loss), 50.0, 100.0)

    df_sim_comp = pd.DataFrame({
        "Cycle": cycles,
        "Baseline Fleet Profile (30% Fast Charge, 25°C)": base_soh,
        "Simulated Operational Profile": simulated_soh
    })

    fig_sim = px.line(
        df_sim_comp,
        x="Cycle",
        y=["Baseline Fleet Profile (30% Fast Charge, 25°C)", "Simulated Operational Profile"],
        title="500-Cycle Battery Degradation Comparison",
        labels={"value": "State of Health (SoH %)"},
        color_discrete_map={
            "Baseline Fleet Profile (30% Fast Charge, 25°C)": "#10B981",
            "Simulated Operational Profile": "#EF4444" if sim_soh[-1] < base_soh[-1] else "#0EA5E9"
        }
    )
    fig_sim.add_hline(y=80.0, line_dash="dash", line_color="orange", annotation_text="80% Caution Threshold")
    fig_sim.add_hline(y=70.0, line_dash="dash", line_color="red", annotation_text="70% EoL Threshold")
    fig_sim.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_sim, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B;'>Antigravity AI Platform | EV Fleet Battery Health & Predictive Maintenance System</p>", unsafe_allow_html=True)
