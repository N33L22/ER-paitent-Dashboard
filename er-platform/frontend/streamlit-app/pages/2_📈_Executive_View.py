"""
Executive View - Strategic KPIs & Performance Analytics
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
    page_title="Executive View | ER Intelligence",
    page_icon="📈",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #6B46C1 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
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
        color: white;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
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
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA GENERATION
# =============================================================================

@st.cache_data(ttl=60)
def generate_weekly_trends(weeks=12):
    """Generate weekly performance trends"""
    data = []
    base_date = datetime.now() - timedelta(weeks=weeks)
    for i in range(weeks):
        week_start = base_date + timedelta(weeks=i)
        data.append({
            'week': week_start, 'week_label': f'W{i+1}',
            'avg_los': 180 + random.gauss(0, 15) - i * 0.8,
            'avg_wait': 45 + random.gauss(0, 8) - i * 0.3,
            'volume': 800 + random.randint(-50, 100) + i * 5,
            'lwbs': max(0.5, 3.5 + random.gauss(0, 0.5) - i * 0.1),
            'satisfaction': min(99, 85 + random.gauss(0, 2) + i * 0.3)
        })
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def generate_acuity_mix():
    """Generate acuity distribution"""
    return pd.DataFrame({
        'ESI': ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
        'Count': [random.randint(10, 25), random.randint(80, 120), random.randint(300, 400), 
                  random.randint(200, 280), random.randint(50, 100)],
        'Avg LOS': [240, 210, 180, 120, 60]
    })


@st.cache_data(ttl=60)
def generate_hourly_heatmap():
    """Generate hourly volume heatmap"""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data = []
    for d, day in enumerate(days):
        for hour in range(24):
            base = 3 if 2 <= hour <= 6 else 8 if 10 <= hour <= 20 else 5
            weekend_mult = 1.2 if day in ['Sat', 'Sun'] else 1.0
            data.append({'day': day, 'day_num': d, 'hour': hour, 'volume': int(base * weekend_mult * random.uniform(0.8, 1.2))})
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def generate_benchmarks():
    """Generate benchmark comparison data"""
    return pd.DataFrame({
        'Metric': ['Avg LOS (min)', 'Door-to-Provider (min)', 'LWBS Rate (%)', 'Admit Decision (min)', 'Satisfaction (%)'],
        'Our ED': [175, 22, 2.8, 45, 92],
        'Target': [180, 25, 3.0, 60, 90],
        'National Avg': [240, 35, 4.5, 90, 78],
        'Top 10%': [150, 15, 1.5, 30, 95]
    })


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">📈</div>
    <h2 style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Executive View</h2>
    <p style="color: #a0aec0; font-size: 0.8rem;">Strategic Analytics</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
date_range = st.sidebar.selectbox("Analysis Period", ["Last 4 Weeks", "Last 8 Weeks", "Last 12 Weeks"], index=2)

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
    <h1 class="page-title">📈 Executive View</h1>
    <p class="page-subtitle">Strategic Performance Dashboard • KPI Tracking & Benchmarking</p>
</div>
""", unsafe_allow_html=True)

# Generate data
weeks_map = {"Last 4 Weeks": 4, "Last 8 Weeks": 8, "Last 12 Weeks": 12}
trends = generate_weekly_trends(weeks_map[date_range])
acuity_mix = generate_acuity_mix()
heatmap_data = generate_hourly_heatmap()
benchmarks = generate_benchmarks()

# KPI Metrics
st.markdown('<p class="section-title">🎯 Key Performance Indicators</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
current, previous = trends.iloc[-1], trends.iloc[-2] if len(trends) > 1 else trends.iloc[-1]

with col1:
    st.metric("⏱️ Avg LOS", f"{current['avg_los']:.0f} min", f"{current['avg_los'] - previous['avg_los']:.1f}", delta_color="inverse")
with col2:
    st.metric("🚪 Avg Wait", f"{current['avg_wait']:.0f} min", f"{current['avg_wait'] - previous['avg_wait']:.1f}", delta_color="inverse")
with col3:
    st.metric("📊 Weekly Volume", f"{current['volume']:.0f}", f"{current['volume'] - previous['volume']:.0f}")
with col4:
    st.metric("🚶 LWBS Rate", f"{current['lwbs']:.1f}%", f"{current['lwbs'] - previous['lwbs']:.2f}", delta_color="inverse")
with col5:
    st.metric("⭐ Satisfaction", f"{current['satisfaction']:.0f}%", f"{current['satisfaction'] - previous['satisfaction']:.1f}")


# Trend Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">📉 LOS & Wait Time Trends</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trends['week_label'], y=trends['avg_los'], name='Avg LOS', marker_color='#667eea'))
    fig.add_trace(go.Bar(x=trends['week_label'], y=trends['avg_wait'], name='Avg Wait', marker_color='#38a169'))
    fig.add_hline(y=180, line_dash="dash", line_color="#e53e3e", annotation_text="LOS Target", 
                  annotation_font=dict(color="white", size=11))
    
    fig.update_layout(
        height=380, margin=dict(l=60, r=40, t=40, b=60), barmode='group',
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white', size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="Minutes", font=dict(family="Inter", size=12, color='white')
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color='white'))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">📊 Volume & Satisfaction</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=trends['week_label'], y=trends['volume'], name='Volume', marker_color='#667eea'))
    
    fig.update_layout(
        height=380, margin=dict(l=60, r=40, t=40, b=60),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white', size=11)),
        yaxis_title="Patients", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='white')
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color='white'))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
    st.plotly_chart(fig, use_container_width=True)


# Acuity & Heatmap
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">🏥 Acuity Distribution</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=acuity_mix['ESI'], y=acuity_mix['Count'], marker_color='#667eea',
        text=acuity_mix['Count'], textposition='outside', textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        height=350, margin=dict(l=60, r=40, t=40, b=60),
        yaxis_title="Patient Count", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='white')
    )
    fig.update_xaxes(tickfont=dict(size=12, color='white'))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<p class="section-title">🗓️ Volume Heatmap</p>', unsafe_allow_html=True)
    
    pivot_data = heatmap_data.pivot(index='day', columns='hour', values='volume')
    days_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_data = pivot_data.reindex(days_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values, x=[f"{h}" for h in range(24)], y=pivot_data.index,
        colorscale='RdYlGn_r', showscale=True,
        colorbar=dict(tickfont=dict(color='white'))
    ))
    fig.update_layout(
        height=350, margin=dict(l=60, r=40, t=40, b=60),
        xaxis_title="Hour", paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12, color='white')
    )
    fig.update_xaxes(tickfont=dict(size=10, color='white'))
    fig.update_yaxes(tickfont=dict(size=11, color='white'))
    st.plotly_chart(fig, use_container_width=True)


# Benchmark Comparison
st.markdown('<p class="section-title">🏆 Performance Benchmarking</p>', unsafe_allow_html=True)

fig = go.Figure()
colors = {'Our ED': '#667eea', 'Target': '#38a169', 'National Avg': '#e53e3e', 'Top 10%': '#f6ad55'}

for col_name in ['Our ED', 'Target', 'National Avg', 'Top 10%']:
    fig.add_trace(go.Bar(
        name=col_name, x=benchmarks['Metric'], y=benchmarks[col_name],
        marker_color=colors[col_name], text=benchmarks[col_name],
        textposition='outside', textfont=dict(color='white', size=10)
    ))

fig.update_layout(
    barmode='group', height=400, margin=dict(l=60, r=40, t=60, b=80),
    legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white', size=11)),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter", size=12, color='white')
)
fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color='white'), tickangle=-15)
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=11, color='white'))
st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("""
<div class="footer">
    <strong>ER Patient Flow Intelligence Platform v2.0</strong><br>
    <span style="color: #a0aec0;">Built by <strong style="color: #667eea;">Neel, Harsh, and Tanishk</strong></span>
</div>
""", unsafe_allow_html=True)
