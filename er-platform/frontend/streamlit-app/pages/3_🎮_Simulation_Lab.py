"""
Simulation Lab - What-If Analysis, Monte Carlo & Scenario Planning
ER Patient Flow Intelligence Platform v2.0

Authors: Neel, Harsh, Tanishk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Simulation Lab | ER Intelligence",
    page_icon="🎮",
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
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 50%, #c05621 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(237, 137, 54, 0.3);
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
        border-bottom: 2px solid #ed8936;
        display: inline-block;
    }
    
    .result-card {
        background: rgba(56, 161, 105, 0.15);
        border-left: 4px solid #38a169;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
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
# SIMULATION ENGINES
# =============================================================================

def run_discrete_event_simulation(config, duration_hours=24, time_steps=96):
    """Run discrete event simulation"""
    results = {'time': [], 'queue_length': [], 'avg_wait': [], 'utilization': [], 'lwbs_count': [], 'throughput': []}
    
    queue, lwbs, throughput = config.get('initial_queue', 5), 0, 0
    beds = config.get('beds', 40)
    staff_efficiency = config.get('staff_efficiency', 1.0)
    
    for step in range(time_steps):
        current_hour = (step * duration_hours / time_steps) % 24
        surge = config.get('surge_factor', 1.0)
        
        base_arrivals = 4 * surge if 10 <= current_hour <= 20 else 2.5 * surge if 6 <= current_hour <= 22 else 1.5 * surge
        arrivals = max(0, int(np.random.poisson(base_arrivals)))
        queue += arrivals
        
        service_capacity = beds * 0.15 * staff_efficiency
        served = min(queue, int(np.random.poisson(service_capacity)))
        queue = max(0, queue - served)
        throughput += served
        
        if queue > beds * 0.8:
            lwbs_count = int(queue * min(0.1, (queue - beds * 0.8) / beds * 0.2))
            queue -= lwbs_count
            lwbs += lwbs_count
        
        results['time'].append(step * duration_hours / time_steps)
        results['queue_length'].append(queue)
        results['avg_wait'].append(max(5, queue * 12 / max(1, service_capacity)))
        results['utilization'].append(min(100, (beds - queue * 0.5) / beds * 100 + 50))
        results['lwbs_count'].append(lwbs)
        results['throughput'].append(throughput)
    
    return pd.DataFrame(results)


def run_monte_carlo_simulation(config, n_simulations=500):
    """Run Monte Carlo simulation for risk analysis"""
    results = {'simulation': [], 'max_queue': [], 'avg_wait': [], 'total_lwbs': [], 'throughput': [], 'peak_utilization': []}
    
    for sim in range(n_simulations):
        beds = config.get('beds', 40) + np.random.randint(-3, 4)
        surge = config.get('surge_factor', 1.0) * np.random.uniform(0.8, 1.2)
        efficiency = config.get('staff_efficiency', 1.0) * np.random.uniform(0.85, 1.15)
        
        sim_result = run_discrete_event_simulation({'beds': beds, 'surge_factor': surge, 'staff_efficiency': efficiency}, 
                                                    duration_hours=24, time_steps=48)
        
        results['simulation'].append(sim)
        results['max_queue'].append(sim_result['queue_length'].max())
        results['avg_wait'].append(sim_result['avg_wait'].mean())
        results['total_lwbs'].append(sim_result['lwbs_count'].iloc[-1])
        results['throughput'].append(sim_result['throughput'].iloc[-1])
        results['peak_utilization'].append(sim_result['utilization'].max())
    
    return pd.DataFrame(results)


def run_staffing_optimization(base_config, steps=10):
    """Run staffing optimization"""
    results = {'staff_level': [], 'avg_wait': [], 'lwbs_rate': [], 'utilization': [], 'cost_index': []}
    
    for staff in np.linspace(0.6, 1.5, steps):
        sim = run_discrete_event_simulation({**base_config, 'staff_efficiency': staff}, duration_hours=24, time_steps=48)
        results['staff_level'].append(staff)
        results['avg_wait'].append(sim['avg_wait'].mean())
        results['lwbs_rate'].append(sim['lwbs_count'].iloc[-1] / max(1, sim['throughput'].iloc[-1]) * 100)
        results['utilization'].append(sim['utilization'].mean())
        results['cost_index'].append(staff * 100)
    
    return pd.DataFrame(results)


def run_capacity_analysis(base_config, steps=7):
    """Run capacity analysis"""
    results = {'beds': [], 'avg_queue': [], 'avg_wait': [], 'lwbs_rate': [], 'throughput': []}
    
    for beds in np.linspace(25, 55, steps):
        sim = run_discrete_event_simulation({**base_config, 'beds': int(beds)}, duration_hours=24, time_steps=48)
        results['beds'].append(int(beds))
        results['avg_queue'].append(sim['queue_length'].mean())
        results['avg_wait'].append(sim['avg_wait'].mean())
        results['lwbs_rate'].append(sim['lwbs_count'].iloc[-1] / max(1, sim['throughput'].iloc[-1]) * 100)
        results['throughput'].append(sim['throughput'].iloc[-1])
    
    return pd.DataFrame(results)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">🎮</div>
    <h2 style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Simulation Lab</h2>
    <p style="color: #a0aec0; font-size: 0.8rem;">What-If Analysis</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Baseline Config")
baseline_beds = st.sidebar.slider("🛏️ Total Beds", 20, 60, 40)
baseline_staff = st.sidebar.slider("👥 Staff Efficiency", 0.5, 1.5, 1.0, 0.1)
baseline_surge = st.sidebar.slider("📈 Surge Factor", 0.5, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Simulation Type")
sim_type = st.sidebar.selectbox("Select", ["📊 Scenario Comparison", "🎲 Monte Carlo Analysis", "👥 Staffing Optimization", "🛏️ Capacity Planning"])

if sim_type == "🎲 Monte Carlo Analysis":
    n_sims = st.sidebar.slider("Simulations", 100, 1000, 500, 100)
else:
    n_sims = 500

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
    <h1 class="page-title">🎮 Simulation Lab</h1>
    <p class="page-subtitle">Discrete Event Simulation • Monte Carlo Analysis • What-If Planning</p>
</div>
""", unsafe_allow_html=True)

base_config = {'beds': baseline_beds, 'staff_efficiency': baseline_staff, 'surge_factor': baseline_surge}


# =============================================================================
# SCENARIO COMPARISON
# =============================================================================
if sim_type == "📊 Scenario Comparison":
    st.markdown('<p class="section-title">🔧 Configure Scenarios</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Scenario A: High Surge")
        surge_a = st.slider("Surge", 0.5, 2.0, 1.5, 0.1, key="sa")
        beds_a = st.slider("Beds", 20, 60, baseline_beds, key="ba")
        eff_a = st.slider("Efficiency", 0.5, 1.5, baseline_staff, 0.1, key="ea")
    with col2:
        st.markdown("#### Scenario B: +10 Beds")
        surge_b = st.slider("Surge", 0.5, 2.0, 1.0, 0.1, key="sb")
        beds_b = st.slider("Beds", 20, 60, min(60, baseline_beds + 10), key="bb")
        eff_b = st.slider("Efficiency", 0.5, 1.5, baseline_staff, 0.1, key="eb")
    with col3:
        st.markdown("#### Scenario C: +30% Staff")
        surge_c = st.slider("Surge", 0.5, 2.0, 1.0, 0.1, key="sc")
        beds_c = st.slider("Beds", 20, 60, baseline_beds, key="bc")
        eff_c = st.slider("Efficiency", 0.5, 1.5, min(1.5, baseline_staff + 0.3), 0.1, key="ec")
    
    if st.button("🚀 Run Scenario Comparison", use_container_width=True, type="primary"):
        with st.spinner("Running simulations..."):
            progress = st.progress(0)
            baseline = run_discrete_event_simulation(base_config); progress.progress(25)
            scen_a = run_discrete_event_simulation({'beds': beds_a, 'staff_efficiency': eff_a, 'surge_factor': surge_a}); progress.progress(50)
            scen_b = run_discrete_event_simulation({'beds': beds_b, 'staff_efficiency': eff_b, 'surge_factor': surge_b}); progress.progress(75)
            scen_c = run_discrete_event_simulation({'beds': beds_c, 'staff_efficiency': eff_c, 'surge_factor': surge_c}); progress.progress(100)
            st.session_state['scenario_results'] = {'Baseline': baseline, 'Scenario A': scen_a, 'Scenario B': scen_b, 'Scenario C': scen_c}
        st.success("✅ Complete!")
    
    if 'scenario_results' in st.session_state:
        results = st.session_state['scenario_results']
        colors = {'Baseline': '#667eea', 'Scenario A': '#e53e3e', 'Scenario B': '#38a169', 'Scenario C': '#ed8936'}
        
        # Summary
        st.markdown('<p class="section-title">📊 Results Summary</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (name, df) in enumerate(results.items()):
            with cols[i]:
                st.metric(name, f"Queue: {df['queue_length'].mean():.1f}")
                st.metric("Wait", f"{df['avg_wait'].mean():.0f} min")
                st.metric("LWBS", f"{df['lwbs_count'].iloc[-1]}")
        
        # Queue Chart
        st.markdown('<p class="section-title">📈 Queue Length Comparison</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for name, df in results.items():
            fig.add_trace(go.Bar(x=df['time'], y=df['queue_length'], name=name, marker_color=colors[name], opacity=0.8))
        fig.update_layout(height=400, margin=dict(l=60,r=40,t=40,b=60), barmode='group',
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white')),
            xaxis_title="Time (hours)", yaxis_title="Queue Length",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Wait & Utilization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">⏱️ Average Wait Time</p>', unsafe_allow_html=True)
            fig = go.Figure()
            for name, df in results.items():
                fig.add_trace(go.Bar(x=[name], y=[df['avg_wait'].mean()], marker_color=colors[name],
                    text=[f"{df['avg_wait'].mean():.0f}"], textposition='outside', textfont=dict(color='white', size=14)))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), showlegend=False, yaxis_title="Wait (min)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white', size=12))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">📊 Resource Utilization</p>', unsafe_allow_html=True)
            fig = go.Figure()
            for name, df in results.items():
                fig.add_trace(go.Bar(x=[name], y=[df['utilization'].mean()], marker_color=colors[name],
                    text=[f"{df['utilization'].mean():.0f}%"], textposition='outside', textfont=dict(color='white', size=14)))
            fig.add_hline(y=85, line_dash="dash", line_color="#e53e3e", annotation_text="Max 85%", annotation_font=dict(color="white"))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), showlegend=False, yaxis_title="Utilization %", yaxis_range=[0,100],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white', size=12))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MONTE CARLO
# =============================================================================
elif sim_type == "🎲 Monte Carlo Analysis":
    st.markdown('<p class="section-title">🎲 Monte Carlo Risk Analysis</p>', unsafe_allow_html=True)
    st.info(f"Running {n_sims} simulations with randomized parameters to assess operational risk.")
    
    if st.button("🚀 Run Monte Carlo", use_container_width=True, type="primary"):
        with st.spinner(f"Running {n_sims} simulations..."):
            mc = run_monte_carlo_simulation(base_config, n_sims)
            st.session_state['mc_results'] = mc
        st.success("✅ Complete!")
    
    if 'mc_results' in st.session_state:
        mc = st.session_state['mc_results']
        
        # Stats
        st.markdown('<p class="section-title">📊 Risk Statistics</p>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Mean Max Queue", f"{mc['max_queue'].mean():.1f}", f"±{mc['max_queue'].std():.1f}")
        with col2: st.metric("Mean Avg Wait", f"{mc['avg_wait'].mean():.0f} min", f"±{mc['avg_wait'].std():.0f}")
        with col3: st.metric("P95 Wait", f"{mc['avg_wait'].quantile(0.95):.0f} min")
        with col4: st.metric("Mean LWBS", f"{mc['total_lwbs'].mean():.1f}", f"±{mc['total_lwbs'].std():.1f}")
        with col5: st.metric("Mean Throughput", f"{mc['throughput'].mean():.0f}")
        
        # Histograms
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">📈 Wait Time Distribution</p>', unsafe_allow_html=True)
            hist, bins = np.histogram(mc['avg_wait'], bins=25)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bins[:-1], y=hist, marker_color='#667eea', name='Frequency'))
            fig.add_vline(x=mc['avg_wait'].quantile(0.5), line_dash="dash", line_color="#38a169", annotation_text=f"Median: {mc['avg_wait'].quantile(0.5):.0f}", annotation_font=dict(color="white"))
            fig.add_vline(x=mc['avg_wait'].quantile(0.95), line_dash="dash", line_color="#e53e3e", annotation_text=f"P95: {mc['avg_wait'].quantile(0.95):.0f}", annotation_font=dict(color="white"))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Wait Time (min)", yaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'), showlegend=False)
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">📈 Max Queue Distribution</p>', unsafe_allow_html=True)
            hist, bins = np.histogram(mc['max_queue'], bins=25)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bins[:-1], y=hist, marker_color='#ed8936', name='Frequency'))
            fig.add_vline(x=mc['max_queue'].quantile(0.95), line_dash="dash", line_color="#e53e3e", annotation_text=f"P95: {mc['max_queue'].quantile(0.95):.0f}", annotation_font=dict(color="white"))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Max Queue", yaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'), showlegend=False)
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk Cards
        st.markdown('<p class="section-title">⚠️ Risk Assessment</p>', unsafe_allow_html=True)
        risks = [
            ("High Wait Risk", (mc['avg_wait'] > 60).mean() * 100, "Wait > 60 min"),
            ("Overcrowding", (mc['max_queue'] > baseline_beds * 0.8).mean() * 100, "Queue > 80% capacity"),
            ("LWBS Risk", (mc['total_lwbs'] > 5).mean() * 100, "LWBS > 5 patients")
        ]
        cols = st.columns(3)
        for i, (title, val, desc) in enumerate(risks):
            color = '#e53e3e' if val > 30 else '#ed8936' if val > 15 else '#38a169'
            with cols[i]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-left: 4px solid {color}; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <h3 style="color: white; margin: 0; font-size: 1rem;">{title}</h3>
                    <p style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 0.5rem 0;">{val:.1f}%</p>
                    <p style="color: #a0aec0; margin: 0; font-size: 0.85rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# STAFFING OPTIMIZATION
# =============================================================================
elif sim_type == "👥 Staffing Optimization":
    st.markdown('<p class="section-title">👥 Staffing Level Optimization</p>', unsafe_allow_html=True)
    st.info("Analyze trade-offs between staffing levels, wait times, and costs.")
    
    if st.button("🚀 Run Staffing Analysis", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            st.session_state['staff_results'] = run_staffing_optimization(base_config)
        st.success("✅ Complete!")
    
    if 'staff_results' in st.session_state:
        sr = st.session_state['staff_results']
        
        st.markdown('<p class="section-title">📈 Wait Time vs Staff Level</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{s:.1f}x" for s in sr['staff_level']], y=sr['avg_wait'], marker_color='#667eea',
            text=[f"{w:.0f}" for w in sr['avg_wait']], textposition='outside', textfont=dict(color='white', size=11)))
        fig.add_hline(y=30, line_dash="dash", line_color="#38a169", annotation_text="Target 30 min", annotation_font=dict(color="white"))
        fig.update_layout(height=400, margin=dict(l=60,r=40,t=60,b=60), xaxis_title="Staff Level", yaxis_title="Wait Time (min)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">💰 Cost Index</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"{s:.1f}x" for s in sr['staff_level']], y=sr['cost_index'], marker_color='#e53e3e',
                text=[f"{c:.0f}" for c in sr['cost_index']], textposition='outside', textfont=dict(color='white', size=10)))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Staff", yaxis_title="Cost",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white', size=10))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">🚶 LWBS Rate</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"{s:.1f}x" for s in sr['staff_level']], y=sr['lwbs_rate'], marker_color='#ed8936',
                text=[f"{l:.1f}%" for l in sr['lwbs_rate']], textposition='outside', textfont=dict(color='white', size=10)))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Staff", yaxis_title="LWBS %",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white', size=10))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        opt = sr[sr['avg_wait'] < 35]['cost_index'].idxmin() if any(sr['avg_wait'] < 35) else sr['avg_wait'].idxmin()
        st.markdown(f"""<div class="result-card"><h3 style="color: #38a169; margin: 0;">💡 Recommendation</h3>
            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Optimal: <strong>{sr.iloc[opt]['staff_level']:.1f}x</strong> baseline</p></div>""", unsafe_allow_html=True)


# =============================================================================
# CAPACITY PLANNING
# =============================================================================
elif sim_type == "🛏️ Capacity Planning":
    st.markdown('<p class="section-title">🛏️ Capacity Planning Analysis</p>', unsafe_allow_html=True)
    st.info("Analyze how bed capacity affects patient flow.")
    
    if st.button("🚀 Run Capacity Analysis", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            st.session_state['cap_results'] = run_capacity_analysis(base_config)
        st.success("✅ Complete!")
    
    if 'cap_results' in st.session_state:
        cr = st.session_state['cap_results']
        
        st.markdown('<p class="section-title">📈 Wait Time by Capacity</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{b}" for b in cr['beds']], y=cr['avg_wait'], marker_color='#667eea',
            text=[f"{w:.0f}" for w in cr['avg_wait']], textposition='outside', textfont=dict(color='white', size=12)))
        fig.add_hline(y=30, line_dash="dash", line_color="#38a169", annotation_text="Target 30 min", annotation_font=dict(color="white"))
        fig.update_layout(height=400, margin=dict(l=60,r=40,t=60,b=60), xaxis_title="Bed Capacity", yaxis_title="Wait (min)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">📊 Queue by Capacity</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"{b}" for b in cr['beds']], y=cr['avg_queue'], marker_color='#38a169',
                text=[f"{q:.1f}" for q in cr['avg_queue']], textposition='outside', textfont=dict(color='white', size=11)))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Beds", yaxis_title="Avg Queue",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white'))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-title">✅ Throughput</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"{b}" for b in cr['beds']], y=cr['throughput'], marker_color='#ed8936',
                text=[f"{t:.0f}" for t in cr['throughput']], textposition='outside', textfont=dict(color='white', size=11)))
            fig.update_layout(height=350, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Beds", yaxis_title="Daily Throughput",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white'))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        opt = cr[cr['avg_wait'] < 35]['beds'].idxmin() if any(cr['avg_wait'] < 35) else cr['avg_wait'].idxmin()
        st.markdown(f"""<div class="result-card"><h3 style="color: #38a169; margin: 0;">💡 Recommendation</h3>
            <p style="color: white; font-size: 1.1rem; margin: 0.5rem 0;">Recommended: <strong>{cr.iloc[opt]['beds']} beds</strong></p></div>""", unsafe_allow_html=True)


# Footer
st.markdown("""
<div class="footer">
    <strong>ER Patient Flow Intelligence Platform v2.0</strong><br>
    <span style="color: #a0aec0;">Built by <strong style="color: #667eea;">Neel, Harsh, and Tanishk</strong></span>
</div>
""", unsafe_allow_html=True)
