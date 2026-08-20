import streamlit as st
import pandas as pd

def inject_custom_css():
    """Injects modern, ultra-clean executive UI styling for Streamlit."""
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Background & Overall Container */
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
        }

        /* Top Executive Header */
        .main-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0D9488 100%);
            padding: 2rem 2.2rem;
            border-radius: 16px;
            color: #FFFFFF;
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 2rem;
        }
        
        .main-header h1 {
            color: #F8FAFC !important;
            font-weight: 800;
            font-size: 2.3rem;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .main-header p {
            color: #94A3B8;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            margin-bottom: 0;
        }

        /* Metric Cards */
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.25s ease-in-out;
        }
        .metric-card:hover {
            border-color: #38BDF8;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
        }
        .metric-label {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.3rem;
            margin-bottom: 0.2rem;
        }
        .metric-delta {
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.2rem;
        }
        .metric-delta.positive { color: #34D399; }
        .metric-delta.negative { color: #F87171; }
        .metric-delta.warning { color: #FBBF24; }

        /* Status Badges */
        .badge-critical {
            background: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }
        .badge-high {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }
        .badge-medium {
            background: rgba(234, 179, 8, 0.15);
            color: #FACC15;
            border: 1px solid rgba(234, 179, 8, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }
        .badge-low {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }

        /* Glassmorphism Section Cards */
        .content-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 1.6rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }

        /* Custom Scrollbars & Streamlit Widgets */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0F172A;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_metric_card(label, value, delta=None, delta_type="positive"):
    """Renders a modern, high-contrast KPI metric card."""
    delta_class = f"metric-delta {delta_type}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ''
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def generate_csv_download(df):
    """Generates a downloadable CSV string from a Pandas dataframe."""
    return df.to_csv(index=False).encode('utf-8')

def generate_work_orders_report_text(work_orders):
    """Generates clean plain text report for maintenance dispatches."""
    lines = []
    lines.append("=========================================================================")
    lines.append("       EV FLEET PREDICTIVE MAINTENANCE WORK ORDER REPORT                 ")
    lines.append(f"       Generated Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}   ")
    lines.append("=========================================================================\n")

    if not work_orders:
        lines.append("No active maintenance work orders pending.")
    else:
        for idx, wo in enumerate(work_orders, 1):
            lines.append(f"--- WORK ORDER #{idx}: {wo['work_order_id']} [{wo['severity']}] ---")
            lines.append(f"Vehicle ID     : {wo['vehicle_id']} ({wo['model']})")
            lines.append(f"Risk Score     : {wo['risk_score']}/100")
            lines.append(f"Current SoH    : {wo['current_soh']:.1f}%")
            lines.append(f"Due Date       : {wo['due_date']}")
            lines.append(f"Est. Downtime  : {wo['estimated_downtime_hrs']} Hours")
            lines.append(f"Est. Cost      : ${wo['estimated_cost_usd']:,}")
            lines.append(f"Action Plan    : {wo['action_plan']}")
            lines.append("\n" + "-"*50 + "\n")

    return "\n".join(lines)
