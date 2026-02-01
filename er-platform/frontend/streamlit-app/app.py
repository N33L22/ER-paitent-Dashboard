"""
ER Patient Flow Intelligence Platform v2.0
Main Streamlit Application

Self-Service Analytics with CSV/Excel Upload and Real-Time Streaming
Hospital Command Center Decision Intelligence System

Authors: Neel, Harsh, Tanishk
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import random

# =============================================================================
# PAGE CONFIGURATION (MUST be first Streamlit command)
# =============================================================================

st.set_page_config(
    page_title="ER Intelligence Platform v2.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "ER Patient Flow Intelligence Platform v2.0 - Built by Neel, Harsh, Tanishk"
    }
)

# =============================================================================
# ENHANCED CUSTOM CSS - BEAUTIFUL UI
# =============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 100%;
        padding: 1rem 2rem 2rem 2rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: white;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.1;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    
    .kpi-delta-positive {
        color: #38a169;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .kpi-delta-negative {
        color: #e53e3e;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .kpi-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a202c;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .chart-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #718096;
        line-height: 1.5;
    }
    
    /* Alert Styling */
    .alert-card {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #fed7d7 0%, #fff5f5 100%);
        border-left: 4px solid #e53e3e;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fefcbf 0%, #fffff0 100%);
        border-left: 4px solid #d69e2e;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #bee3f8 0%, #ebf8ff 100%);
        border-left: 4px solid #3182ce;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 2px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .nav-card:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
    }
    
    .nav-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .nav-title {
        font-weight: 600;
        color: #2d3748;
        font-size: 0.95rem;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
        text-align: center;
    }
    
    .footer-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .footer-team {
        font-size: 1rem;
        color: #a0aec0;
    }
    
    .footer-team strong {
        color: #667eea;
    }
    
    /* Status Indicator */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    .status-dot-green {
        background-color: #38a169;
        box-shadow: 0 0 10px #38a169;
    }
    
    .status-dot-red {
        background-color: #e53e3e;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Quick Stats Bar */
    .quick-stats {
        background: linear-gradient(90deg, #1a202c 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .quick-stat-item {
        text-align: center;
        color: white;
    }
    
    .quick-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .quick-stat-label {
        font-size: 0.75rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Improve metric styling */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }
    
    div[data-testid="metric-container"] > div {
        color: #1a202c;
    }
    
    /* Plotly Charts */
    .js-plotly-plot, .plot-container {
        width: 100% !important;
    }
    
    .stPlotlyChart {
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DEMO DATA GENERATION FUNCTIONS
# =============================================================================

def generate_current_state():
    """Generate realistic current ED state"""
    hour = datetime.now().hour
    # More arrivals during peak hours (10am-8pm)
    if 10 <= hour <= 20:
        base_census = random.randint(30, 38)
    else:
        base_census = random.randint(20, 28)
    
    capacity = 40
    return {
        'total_patients': base_census,
        'beds_total': capacity,
        'bed_utilization': base_census / capacity,
        'mean_wait_time': random.randint(25, 55),
        'triage_queue': random.randint(1, 6),
        'forecast_4h': base_census + random.randint(10, 25),
        'lwbs_rate': random.uniform(0.02, 0.05),
        'patient_delta': random.choice([-2, -1, 1, 2, 3]),
        'bed_util_delta': random.uniform(-0.05, 0.08),
        'wait_delta': random.choice([-5, -3, 3, 5, 8]),
        'triage_delta': random.choice([-1, 0, 1, 2]),
        'forecast_uncertainty': random.randint(8, 15),
        'lwbs_delta': random.uniform(-0.01, 0.01)
    }

def generate_hourly_metrics(hours=72):
    """Generate realistic hourly metrics"""
    data = []
    base_time = datetime.now() - timedelta(hours=hours)
    
    for i in range(hours):
        t = base_time + timedelta(hours=i)
        hour = t.hour
        day = t.weekday()
        
        # Realistic arrival patterns
        if 10 <= hour <= 20:
            base_arrivals = 18
        elif 6 <= hour < 10:
            base_arrivals = 12
        else:
            base_arrivals = 8
        
        # Weekend adjustment
        if day >= 5:
            base_arrivals = int(base_arrivals * 0.85)
        
        arrivals = max(2, base_arrivals + random.randint(-4, 4))
        census = max(15, 28 + random.randint(-8, 10))
        
        data.append({
            'timestamp': t,
            'hour': i,
            'arrivals': arrivals,
            'census': census,
            'mean_wait': random.randint(20, 50),
            'mean_los': random.randint(150, 280)
        })
    
    return pd.DataFrame(data)

def generate_alerts():
    """Generate sample alerts"""
    alerts = []
    if random.random() > 0.6:
        alerts.append({
            'severity': 'warning',
            'message': 'High wait times detected in triage area (avg 45 min)',
            'time': datetime.now() - timedelta(minutes=random.randint(5, 30))
        })
    if random.random() > 0.8:
        alerts.append({
            'severity': 'critical',
            'message': 'Bed utilization exceeding 90% threshold',
            'time': datetime.now() - timedelta(minutes=random.randint(1, 15))
        })
    if random.random() > 0.5:
        alerts.append({
            'severity': 'info',
            'message': 'Shift change in 2 hours - staffing transition',
            'time': datetime.now() - timedelta(minutes=random.randint(10, 60))
        })
    return alerts


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'realtime_enabled' not in st.session_state:
    st.session_state.realtime_enabled = False

if 'data_source' not in st.session_state:
    st.session_state.data_source = "synthetic"

if 'refresh_key' not in st.session_state:
    st.session_state.refresh_key = 0


# =============================================================================
# BEAUTIFUL SIDEBAR
# =============================================================================

# Sidebar Header with Logo
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏥</div>
    <h1 style="color: white; font-size: 1.4rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;">
        ER Intelligence
    </h1>
    <p style="color: #a0aec0; font-size: 0.85rem; margin: 0.25rem 0 0 0;">
        Platform v2.0
    </p>
    <div style="margin-top: 0.75rem;">
        <span style="background: linear-gradient(90deg, #667eea, #764ba2); padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.75rem; color: white;">
            <span class="status-dot status-dot-green"></span>LIVE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Data Source Selection with nice styling
st.sidebar.markdown("""
<p style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
    📊 Data Source
</p>
""", unsafe_allow_html=True)

data_source = st.sidebar.radio(
    "Select Data Source",
    ["🎲 Synthetic Demo", "📁 Upload CSV/Excel", "🔴 Real-Time Stream"],
    index=0,
    label_visibility="collapsed"
)

# Handle data source selection
if data_source == "📁 Upload CSV/Excel":
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Drop your file here",
        type=['csv', 'xlsx', 'xls'],
        help="Upload ED patient data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            st.session_state.data_source = "uploaded"
            st.sidebar.success(f"✅ Loaded {len(df):,} rows")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

elif data_source == "🔴 Real-Time Stream":
    st.sidebar.markdown("---")
    update_interval = st.sidebar.slider("Update Interval", 5, 60, 10, format="%ds")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.realtime_enabled = True
    with col2:
        if st.button("⏸️ Stop", use_container_width=True):
            st.session_state.realtime_enabled = False
    
    if st.session_state.realtime_enabled:
        st.sidebar.markdown('<p style="color: #38a169; font-size: 0.85rem;">🔴 Stream Active</p>', unsafe_allow_html=True)

else:
    st.session_state.data_source = "synthetic"

# Status Section
st.sidebar.markdown("---")
st.sidebar.markdown("""
<p style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
    📡 System Status
</p>
""", unsafe_allow_html=True)

status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    st.markdown('<p style="color: #e2e8f0; font-size: 0.85rem;"><span style="color: #38a169;">●</span> Data Ready</p>', unsafe_allow_html=True)
with status_col2:
    stream_status = "🔴 Live" if st.session_state.realtime_enabled else "⏸️ Static"
    st.markdown(f'<p style="color: #e2e8f0; font-size: 0.85rem;">{stream_status}</p>', unsafe_allow_html=True)

# Navigation Section
st.sidebar.markdown("---")
st.sidebar.markdown("""
<p style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;">
    📑 Navigation
</p>
<div style="padding: 0.5rem 0;">
    <p style="color: #e2e8f0; font-size: 0.9rem; margin: 0.4rem 0;">🏠 <b>Dashboard</b> (Current)</p>
    <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.4rem 0;">🔬 Operational View</p>
    <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.4rem 0;">📈 Executive View</p>
    <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.4rem 0;">🎮 Simulation Lab</p>
    <p style="color: #a0aec0; font-size: 0.9rem; margin: 0.4rem 0;">🧪 Research Explorer</p>
</div>
""", unsafe_allow_html=True)

# Refresh Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.session_state.refresh_key += 1
    st.rerun()

# Team Credits
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <p style="color: #718096; font-size: 0.75rem; margin: 0;">Built with ❤️ by</p>
    <p style="color: #667eea; font-size: 0.9rem; font-weight: 600; margin: 0.25rem 0 0 0;">
        Neel, Harsh, Tanishk
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# Generate data
current_state = generate_current_state()
hourly_data = generate_hourly_metrics(72)
alerts = generate_alerts()

# Hero Header
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🏥 ER Patient Flow Intelligence</h1>
    <p class="hero-subtitle">Hospital Command Center • Real-Time Decision Support • AI-Powered Analytics</p>
    <div style="text-align: center;">
        <span class="hero-badge">📊 Self-Service Analytics Platform v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Stats Bar
census = current_state['total_patients']
capacity = current_state['beds_total']
utilization = (census / capacity) * 100

st.markdown(f"""
<div class="quick-stats">
    <div class="quick-stat-item">
        <div class="quick-stat-value">{datetime.now().strftime('%H:%M')}</div>
        <div class="quick-stat-label">Current Time</div>
    </div>
    <div class="quick-stat-item">
        <div class="quick-stat-value">{census}/{capacity}</div>
        <div class="quick-stat-label">Census/Capacity</div>
    </div>
    <div class="quick-stat-item">
        <div class="quick-stat-value">{utilization:.0f}%</div>
        <div class="quick-stat-label">Utilization</div>
    </div>
    <div class="quick-stat-item">
        <div class="quick-stat-value">{current_state['mean_wait_time']} min</div>
        <div class="quick-stat-label">Avg Wait</div>
    </div>
    <div class="quick-stat-item">
        <div class="quick-stat-value">{current_state['triage_queue']}</div>
        <div class="quick-stat-label">In Queue</div>
    </div>
    <div class="quick-stat-item">
        <div class="quick-stat-value">{datetime.now().strftime('%a, %b %d')}</div>
        <div class="quick-stat-label">Today</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Active Alerts
if alerts:
    for alert in alerts:
        alert_class = f"alert-{alert['severity']}"
        icon = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'warning' else "🔵"
        st.markdown(f"""
        <div class="alert-card {alert_class}">
            <span style="font-size: 1.25rem;">{icon}</span>
            <div>
                <strong>{alert['severity'].upper()}</strong>: {alert['message']}
                <span style="color: #718096; font-size: 0.8rem; margin-left: 0.5rem;">
                    {alert['time'].strftime('%H:%M')}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# KPI Section Header
st.markdown('<p class="section-header">📊 Key Performance Indicators</p>', unsafe_allow_html=True)

# KPI Cards Row
col1, col2, col3, col4, col5, col6 = st.columns(6)

kpi_data = [
    ("🏥", "Patients in ED", current_state['total_patients'], f"+{current_state['patient_delta']}" if current_state['patient_delta'] > 0 else str(current_state['patient_delta']), current_state['patient_delta'] > 0),
    ("🛏️", "Bed Utilization", f"{current_state['bed_utilization']:.0%}", f"{current_state['bed_util_delta']:+.1%}", current_state['bed_util_delta'] > 0),
    ("⏱️", "Mean Wait Time", f"{current_state['mean_wait_time']} min", f"{current_state['wait_delta']:+d} min", current_state['wait_delta'] > 0),
    ("📋", "Triage Queue", current_state['triage_queue'], f"{current_state['triage_delta']:+d}", current_state['triage_delta'] > 0),
    ("🔮", "4h Forecast", f"{current_state['forecast_4h']} pts", f"±{current_state['forecast_uncertainty']}", None),
    ("🚶", "LWBS Rate", f"{current_state['lwbs_rate']:.1%}", f"{current_state['lwbs_delta']:+.1%}", current_state['lwbs_delta'] > 0),
]

for col, (icon, label, value, delta, is_negative) in zip([col1, col2, col3, col4, col5, col6], kpi_data):
    with col:
        delta_class = "kpi-delta-negative" if is_negative else "kpi-delta-positive" if is_negative == False else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="{delta_class}">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Charts Section
st.markdown('<p class="section-header">📈 Real-Time Analytics</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 1])

with col1:
    # Arrivals & Census Chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🚑 Hourly Patient Arrivals (Last 72h)", "📊 ED Census Trend"),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Arrivals bars with gradient colors
    colors = ['#667eea' if x < 15 else '#764ba2' if x < 18 else '#f093fb' for x in hourly_data['arrivals']]
    
    fig.add_trace(
        go.Bar(
            x=hourly_data['timestamp'],
            y=hourly_data['arrivals'],
            name="Arrivals",
            marker_color=colors,
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Arrivals: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Census line with fill
    fig.add_trace(
        go.Scatter(
            x=hourly_data['timestamp'],
            y=hourly_data['census'],
            name="Census",
            mode='lines',
            line=dict(color='#e53e3e', width=2),
            fill='tozeroy',
            fillcolor='rgba(229, 62, 62, 0.15)',
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Census: %{y}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add capacity line
    fig.add_hline(y=capacity, line_dash="dash", line_color="#718096", 
                  annotation_text="Capacity", row=2, col=1)
    
    fig.update_layout(
        height=420,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#2d3748")
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Census Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=census,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Census", 'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}},
        number={'font': {'size': 48, 'color': '#1a202c', 'family': 'Inter'}},
        delta={'reference': capacity * 0.8, 'increasing': {'color': "#e53e3e"}, 'decreasing': {'color': "#38a169"}},
        gauge={
            'axis': {'range': [0, capacity], 'tickwidth': 1, 'tickcolor': "#718096"},
            'bar': {'color': '#667eea'},
            'bgcolor': "#f7fafc",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, capacity * 0.7], 'color': 'rgba(56, 161, 105, 0.2)'},
                {'range': [capacity * 0.7, capacity * 0.85], 'color': 'rgba(214, 158, 46, 0.2)'},
                {'range': [capacity * 0.85, capacity], 'color': 'rgba(229, 62, 62, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#e53e3e", 'width': 4},
                'thickness': 0.75,
                'value': capacity * 0.9
            }
        }
    ))
    
    fig_gauge.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Acuity Distribution
    acuity_labels = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']
    acuity_values = [random.randint(0, 3), random.randint(3, 8), random.randint(10, 16), 
                    random.randint(5, 10), random.randint(1, 4)]
    acuity_colors = ['#e53e3e', '#dd6b20', '#d69e2e', '#38a169', '#3182ce']
    
    fig_acuity = go.Figure(data=[
        go.Bar(
            x=acuity_labels,
            y=acuity_values,
            marker_color=acuity_colors,
            text=acuity_values,
            textposition='outside',
            textfont=dict(size=14, color='#2d3748', family='Inter'),
            hovertemplate="<b>%{x}</b><br>Patients: %{y}<extra></extra>"
        )
    ])
    
    fig_acuity.update_layout(
        title=dict(text="🎯 Acuity Distribution (ESI)", font=dict(size=16, color='#2d3748', family='Inter')),
        xaxis_title="",
        yaxis_title="Patients",
        height=200,
        margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#2d3748"),
        showlegend=False
    )
    
    fig_acuity.update_xaxes(showgrid=False)
    fig_acuity.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    
    st.plotly_chart(fig_acuity, use_container_width=True)

# Quick Start Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">🚀 Quick Start Guide</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📁</div>
        <div class="feature-title">Upload Your Data</div>
        <div class="feature-desc">
            1. Select "📁 Upload CSV/Excel" in sidebar<br>
            2. Drop your ED patient data file<br>
            3. System auto-detects schema<br>
            4. Navigate to analytics views<br><br>
            <b>Formats:</b> CSV, Excel (.xlsx, .xls)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔴</div>
        <div class="feature-title">Real-Time Streaming</div>
        <div class="feature-desc">
            1. Select "🔴 Real-Time Stream" in sidebar<br>
            2. Configure update interval<br>
            3. Click Start to begin streaming<br>
            4. Watch live updates across views<br><br>
            <b>Features:</b> Auto-refresh, live metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎲</div>
        <div class="feature-title">Synthetic Demo Mode</div>
        <div class="feature-desc">
            1. Select "🎲 Synthetic Demo" (default)<br>
            2. Explore all platform features<br>
            3. Test analytics and visualizations<br>
            4. See realistic ED patterns<br><br>
            <b>Based on:</b> MIMIC-IV-ED patterns
        </div>
    </div>
    """, unsafe_allow_html=True)

# Platform Capabilities
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">✨ Platform Capabilities</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Advanced Visualizations</div>
        <div class="feature-desc">
            • 3D Queue Evolution Surfaces<br>
            • Real-Time Sankey Flows<br>
            • Predictive Forecast Bands<br>
            • Heatmap Calendars<br>
            • Network Bottleneck Graphs<br>
            • Survival Curves
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">AI-Powered Insights</div>
        <div class="feature-desc">
            • Automated anomaly detection<br>
            • Statistical trend analysis<br>
            • Bottleneck identification<br>
            • Causal inference<br>
            • Natural language summaries<br>
            • Risk stratification
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Decision Support</div>
        <div class="feature-desc">
            • What-if scenario testing<br>
            • Resource optimization<br>
            • Capacity planning<br>
            • Staffing recommendations<br>
            • Real-time alerts<br>
            • Predictive warnings
        </div>
    </div>
    """, unsafe_allow_html=True)

# Navigation Cards
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">📑 Explore Views</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🔬</div>
        <div class="nav-title">Operational View</div>
        <p style="font-size: 0.8rem; color: #718096; margin: 0.25rem 0 0 0;">Real-time operations</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📈</div>
        <div class="nav-title">Executive View</div>
        <p style="font-size: 0.8rem; color: #718096; margin: 0.25rem 0 0 0;">Strategic insights</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🎮</div>
        <div class="nav-title">Simulation Lab</div>
        <p style="font-size: 0.8rem; color: #718096; margin: 0.25rem 0 0 0;">What-if analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🧪</div>
        <div class="nav-title">Research Explorer</div>
        <p style="font-size: 0.8rem; color: #718096; margin: 0.25rem 0 0 0;">Advanced analytics</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-title">ER Patient Flow Intelligence Platform v2.0</div>
    <div class="footer-team">
        Built with ❤️ by <strong>Neel, Harsh, and Tanishk</strong>
    </div>
    <p style="color: #718096; font-size: 0.85rem; margin-top: 0.5rem;">
        Self-Service Analytics • Real-Time Decision Support • Production-Grade Visualizations
    </p>
</div>
""", unsafe_allow_html=True)
