"""
Operational View - Real-time ED Operations Dashboard
ER Patient Flow Intelligence Platform v2.0

Authors: Neel, Harsh, Tanishk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Operational View | ER Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        max-width: 100%;
        padding: 1rem 2rem 2rem 2rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #2d3748 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    
    .page-header {
        background: linear-gradient(135deg, #38a169 0%, #2f855a 50%, #276749 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(56, 161, 105, 0.3);
    }
    
    .page-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .page-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.25rem;
    }
    
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #ffffff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #38a169;
        display: inline-block;
    }
    
    .footer {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 2rem;
        text-align: center;
    }
    
    /* Dark theme for metrics */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        padding: 1rem;
        border-radius: 12px;
    }
    
    div[data-testid="metric-container"] > div {
        color: #ffffff;
    }
    
    div[data-testid="metric-container"] label {
        color: #a0aec0 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA GENERATION
# =============================================================================

@st.cache_data(ttl=60)
def generate_queue_data(hours=24):
    """Generate queue evolution data"""
    data = []
    base_time = datetime.now() - timedelta(hours=hours)
    queue = random.randint(3, 8)
    
    for i in range(hours * 4):
        t = base_time + timedelta(minutes=i*15)
        hour = t.hour
        arrival_rate = 0.4 if 10 <= hour <= 20 else 0.25 if 6 <= hour <= 22 else 0.15
        arrivals = max(0, int(arrival_rate * 15 + random.gauss(0, 2)))
        departures = max(0, int(0.35 * 15 + random.gauss(0, 1))) if queue > 0 else 0
        queue = max(0, min(25, queue + arrivals - departures))
        data.append({'time': t, 'queue': queue, 'arrivals': arrivals, 'departures': departures})
    return pd.DataFrame(data)


@st.cache_data(ttl=30)
def generate_patient_list(n=30):
    """Generate current patient list"""
    complaints = ['Chest Pain', 'Abdominal Pain', 'SOB', 'Headache', 'Laceration', 
                  'Fall', 'Fever', 'Back Pain', 'Nausea/Vomiting', 'Weakness']
    patients = []
    for i in range(n):
        arrival = datetime.now() - timedelta(minutes=random.randint(5, 360))
        time_elapsed = (datetime.now() - arrival).total_seconds() / 60
        acuity = random.choices([1, 2, 3, 4, 5], weights=[3, 12, 45, 28, 12])[0]
        
        if time_elapsed < 30:
            status = 'Waiting'
        elif time_elapsed < 90:
            status = random.choice(['Waiting', 'In Treatment'])
        elif time_elapsed < 180:
            status = random.choice(['In Treatment', 'Awaiting Results'])
        else:
            status = random.choice(['Awaiting Results', 'Ready for Discharge'])
        
        patients.append({
            'ID': f'P{1000 + i}', 'Arrival': arrival.strftime('%H:%M'), 'ESI': acuity,
            'Status': status, 'Wait': int(time_elapsed) if status == 'Waiting' else random.randint(5, 25),
            'LOS': int(time_elapsed), 'Complaint': random.choice(complaints),
            'Bed': f'B{random.randint(1, 40)}' if status != 'Waiting' else '-'
        })
    return pd.DataFrame(patients).sort_values('ESI')


@st.cache_data(ttl=30)
def generate_resource_status():
    """Generate resource utilization"""
    return {
        'beds': {'used': random.randint(28, 36), 'total': 40},
        'physicians': {'on_duty': random.randint(3, 5), 'total': 6},
        'nurses': {'on_duty': random.randint(6, 10), 'total': 12},
        'resus_bays': {'used': random.randint(0, 2), 'total': 2},
        'trauma_bays': {'used': random.randint(1, 3), 'total': 4},
        'fast_track': {'used': random.randint(2, 5), 'total': 6}
    }


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">🔬</div>
    <h2 style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Operational View</h2>
    <p style="color: #a0aec0; font-size: 0.8rem;">Real-Time Operations</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Filters")
time_range = st.sidebar.selectbox("Time Range", ["Last 6 Hours", "Last 12 Hours", "Last 24 Hours"], index=2)
status_filter = st.sidebar.multiselect("Status", ["Waiting", "In Treatment", "Awaiting Results", "Ready for Discharge"], 
                                       default=["Waiting", "In Treatment", "Awaiting Results", "Ready for Discharge"])
acuity_filter = st.sidebar.multiselect("Acuity (ESI)", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.75rem;">
    Made by <strong style="color: #667eea;">Neel, Harsh, Tanishk</strong>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("""
<div class="page-header">
    <h1 class="page-title">🔬 Operational View</h1>
    <p class="page-subtitle">Real-Time Emergency Department Operations • Live Patient Tracking</p>
</div>
""", unsafe_allow_html=True)

# Generate data
hours_map = {"Last 6 Hours": 6, "Last 12 Hours": 12, "Last 24 Hours": 24}
queue_data = generate_queue_data(hours_map[time_range])
patients = generate_patient_list(35)
resources = generate_resource_status()
patients_filtered = patients[patients['Status'].isin(status_filter) & patients['ESI'].isin(acuity_filter)]

# Resource Metrics
st.markdown('<p class="section-title">📊 Resource Status</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
resource_metrics = [
    ("🛏️ Beds", resources['beds']['used'], resources['beds']['total']),
    ("👨‍⚕️ Physicians", resources['physicians']['on_duty'], resources['physicians']['total']),
    ("👩‍⚕️ Nurses", resources['nurses']['on_duty'], resources['nurses']['total']),
    ("🚨 Resus", resources['resus_bays']['used'], resources['resus_bays']['total']),
    ("🏥 Trauma", resources['trauma_bays']['used'], resources['trauma_bays']['total']),
    ("⚡ Fast Track", resources['fast_track']['used'], resources['fast_track']['total']),
]

for col, (label, used, total) in zip([col1, col2, col3, col4, col5, col6], resource_metrics):
    with col:
        pct = used / total * 100
        st.metric(label, f"{used}/{total}", f"{pct:.0f}%")


# Charts Row
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<p class="section-title">📈 Queue Evolution</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=queue_data['time'], y=queue_data['queue'],
        marker_color='#38a169', name='Queue Length'
    ))
    
    fig.update_layout(
        height=350, margin=dict(l=60, r=40, t=30, b=60),
        xaxis_title="Time", yaxis_title="Patients in Queue",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='white')
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">🎯 Status Distribution</p>', unsafe_allow_html=True)
    
    status_counts = patients['Status'].value_counts()
    colors = ['#e53e3e', '#38a169', '#d69e2e', '#3182ce']
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index, values=status_counts.values, hole=0.5,
        marker_colors=colors, textinfo='label+value', textposition='outside',
        textfont=dict(size=12, color='white')
    )])
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white')
    )
    st.plotly_chart(fig, use_container_width=True)


# Patient Stats
st.markdown('<p class="section-title">👥 Current Patients</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔴 Waiting", len(patients[patients['Status'] == 'Waiting']))
with col2:
    st.metric("🟢 In Treatment", len(patients[patients['Status'] == 'In Treatment']))
with col3:
    avg_wait = patients[patients['Status'] == 'Waiting']['Wait'].mean()
    st.metric("⏱️ Avg Wait", f"{avg_wait:.0f} min" if not pd.isna(avg_wait) else "0 min")
with col4:
    st.metric("🚨 High Acuity", len(patients[patients['ESI'] <= 2]))

st.dataframe(patients_filtered[['ID', 'Arrival', 'ESI', 'Status', 'Wait', 'LOS', 'Complaint', 'Bed']],
             use_container_width=True, height=300, hide_index=True)


# Arrivals vs Departures
st.markdown('<p class="section-title">🚑 Hourly Arrivals vs Departures</p>', unsafe_allow_html=True)

queue_data['hour'] = queue_data['time'].dt.floor('H')
hourly = queue_data.groupby('hour').agg({'arrivals': 'sum', 'departures': 'sum'}).reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['arrivals'], name='Arrivals', marker_color='#667eea'))
fig.add_trace(go.Bar(x=hourly['hour'], y=hourly['departures'], name='Departures', marker_color='#38a169'))

fig.update_layout(
    height=300, barmode='group', margin=dict(l=60, r=40, t=30, b=60),
    legend=dict(orientation="h", y=1.15, x=0.5, xanchor='center', font=dict(color='white')),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter", size=12, color='white')
)
fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color='white'))
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# REAL-TIME AUTO-REFRESH
# =============================================================================

# Session state for real-time updates
if 'op_realtime_enabled' not in st.session_state:
    st.session_state.op_realtime_enabled = False

if 'op_update_interval' not in st.session_state:
    st.session_state.op_update_interval = 5

if 'op_last_update' not in st.session_state:
    st.session_state.op_last_update = datetime.now()

# Real-time controls in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔴 Real-Time Streaming")

op_update_interval = st.sidebar.slider("Update Interval", 3, 30, st.session_state.op_update_interval, format="%ds", key="op_interval")
st.session_state.op_update_interval = op_update_interval

rt_col1, rt_col2 = st.sidebar.columns(2)
with rt_col1:
    if st.button("▶️ Start", use_container_width=True, key="op_start"):
        st.session_state.op_realtime_enabled = True
        st.session_state.op_last_update = datetime.now()
with rt_col2:
    if st.button("⏸️ Stop", use_container_width=True, key="op_stop"):
        st.session_state.op_realtime_enabled = False

if st.session_state.op_realtime_enabled:
    time_since = (datetime.now() - st.session_state.op_last_update).total_seconds()
    st.sidebar.markdown(f'''
    <div style="background: rgba(56, 161, 105, 0.2); border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;">
        <p style="color: #38a169; font-size: 0.85rem; margin: 0;">🔴 <b>LIVE</b> - Next update: {max(0, op_update_interval - int(time_since))}s</p>
    </div>
    ''', unsafe_allow_html=True)
    st.sidebar.progress(min(1.0, time_since / op_update_interval))

# Footer
st.markdown("""
<div class="footer">
    <strong>ER Patient Flow Intelligence Platform v2.0</strong><br>
    <span style="color: #a0aec0;">Built by <strong style="color: #667eea;">Neel, Harsh, and Tanishk</strong></span>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
import time

if st.session_state.op_realtime_enabled:
    time_since_update = (datetime.now() - st.session_state.op_last_update).total_seconds()
    
    if time_since_update >= st.session_state.op_update_interval:
        st.session_state.op_last_update = datetime.now()
        time.sleep(0.1)
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()
