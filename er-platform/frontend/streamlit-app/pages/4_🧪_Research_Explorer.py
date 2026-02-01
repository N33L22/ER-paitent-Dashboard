"""
Research Explorer - ML Insights, SHAP, Survival Analysis & Causal Inference
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
    page_title="Research Explorer | ER Intelligence",
    page_icon="🧪",
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
        background: linear-gradient(135deg, #805AD5 0%, #6B46C1 50%, #553C9A 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(107, 70, 193, 0.3);
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
        border-bottom: 2px solid #805AD5;
        display: inline-block;
    }
    
    .insight-card {
        background: rgba(128, 90, 213, 0.15);
        border-left: 4px solid #805AD5;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
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

@st.cache_data(ttl=120)
def generate_shap_values():
    """Generate SHAP feature importance"""
    features = ['Queue Length', 'Time of Day', 'Day of Week', 'Acuity (ESI)', 'Available Beds', 
                'Staff on Duty', 'Arrival Rate', 'Avg Recent LOS', 'Weather', 'Nearby Hospitals', 'Holiday', 'Season']
    importance = [0.25, 0.18, 0.12, 0.11, 0.09, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.005]
    importance = [max(0, i + random.gauss(0, 0.02)) for i in importance]
    total = sum(importance)
    importance = [i / total for i in importance]
    
    return pd.DataFrame({
        'Feature': features, 'Importance': importance,
        'Direction': ['↑ Wait'] * 4 + ['↓ Wait'] * 2 + ['↑ Wait'] * 2 + ['Varies'] * 4
    }).sort_values('Importance', ascending=True)


@st.cache_data(ttl=120)
def generate_survival_data():
    """Generate survival analysis data"""
    data = []
    for t in range(0, 361, 15):
        for esi in [1, 2, 3, 4, 5]:
            base_rate = 0.003 if esi <= 2 else 0.005 if esi == 3 else 0.008
            data.append({'time': t, 'ESI': f'ESI {esi}', 'survival': np.exp(-base_rate * t)})
    return pd.DataFrame(data)


@st.cache_data(ttl=120)
def generate_prediction_accuracy():
    """Generate model accuracy metrics over time"""
    data = []
    for i in range(12):
        month = datetime.now() - timedelta(days=30 * (11 - i))
        data.append({
            'month': month.strftime('%b'), 'mae': 18 - i * 0.3 + random.gauss(0, 1),
            'rmse': 25 - i * 0.4 + random.gauss(0, 1.5), 'r2': 0.78 + i * 0.005 + random.gauss(0, 0.01)
        })
    return pd.DataFrame(data)


@st.cache_data(ttl=120)
def generate_error_by_acuity():
    """Generate prediction error by acuity"""
    return pd.DataFrame({
        'ESI': ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
        'MAE': [22, 18, 15, 12, 8],
        'Samples': [120, 450, 1800, 1200, 400]
    })


@st.cache_data(ttl=120)
def generate_cluster_data():
    """Generate patient clusters"""
    clusters = [
        ('Low Acuity - Quick', 150, 1.5, 1.0, '#38a169'),
        ('Moderate - Standard', 300, 3.0, 2.5, '#667eea'),
        ('Complex - Extended', 100, 4.0, 4.5, '#ed8936'),
        ('Critical - Intensive', 40, 1.2, 5.0, '#e53e3e'),
        ('Observation - Waiting', 80, 2.5, 3.8, '#805AD5')
    ]
    data = []
    for name, size, cx, cy, color in clusters:
        data.append({'Cluster': name, 'Size': size, 'Complexity': cx, 'LOS': cy, 'Color': color})
    return pd.DataFrame(data)


@st.cache_data(ttl=120)
def generate_causal_effects():
    """Generate causal inference results"""
    return pd.DataFrame({
        'Intervention': ['Add 5 Beds', 'Add 1 Physician', 'Add 2 Nurses', 'Fast Track Expansion', 'Triage Optimization'],
        'Effect on Wait': [-8.5, -5.2, -3.8, -6.1, -4.5],
        'Effect on LOS': [-12.3, -8.1, -5.5, -15.2, -7.8],
        'Effect on LWBS': [-1.2, -0.8, -0.5, -1.5, -0.6],
        'Confidence': [95, 92, 88, 90, 85]
    })


@st.cache_data(ttl=120)
def generate_model_performance():
    """Generate model performance metrics"""
    return {'mae': 15.3, 'rmse': 22.7, 'r2': 0.847, 'mape': 12.4, 'acc_15': 78, 'acc_30': 91}


@st.cache_data(ttl=120)
def generate_feature_interactions():
    """Generate feature interaction data"""
    return pd.DataFrame({
        'Feature Pair': ['Queue × Time', 'Acuity × Beds', 'Staff × Volume', 'Wait × Acuity', 'Beds × Staff'],
        'Interaction Strength': [0.85, 0.72, 0.68, 0.61, 0.55],
        'Impact Direction': ['Synergistic', 'Synergistic', 'Antagonistic', 'Synergistic', 'Synergistic']
    })


@st.cache_data(ttl=120)
def generate_residual_analysis():
    """Generate residual analysis data"""
    predictions = np.random.uniform(20, 120, 200)
    residuals = np.random.normal(0, 10, 200) + (predictions - 70) * 0.05
    return pd.DataFrame({'Predicted': predictions, 'Residual': residuals})


@st.cache_data(ttl=120)
def generate_sensitivity_analysis():
    """Generate sensitivity analysis"""
    return pd.DataFrame({
        'Parameter': ['Arrival Rate', 'Service Rate', 'Staff Level', 'Bed Count', 'Triage Speed'],
        'Base Value': [8.5, 0.35, 1.0, 40, 10],
        '+10% Effect': [12.5, -8.2, -6.5, -4.8, -3.2],
        '-10% Effect': [-10.8, 9.5, 7.8, 5.2, 3.8]
    })


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">🧪</div>
    <h2 style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Research Explorer</h2>
    <p style="color: #a0aec0; font-size: 0.8rem;">ML Insights</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis")
analysis = st.sidebar.selectbox("Select", ["All Analyses", "Feature Importance", "Survival Analysis", 
                                            "Model Performance", "Patient Clustering", "Causal Inference",
                                            "Feature Interactions", "Sensitivity Analysis"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Model")
model = st.sidebar.selectbox("Active", ["XGBoost (Production)", "LightGBM (Testing)", "Neural Network (Exp)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Advanced Options")
show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
show_residuals = st.sidebar.checkbox("Show Residual Analysis", value=False)

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
    <h1 class="page-title">🧪 Research Explorer</h1>
    <p class="page-subtitle">Machine Learning Insights • Explainable AI • Causal Inference</p>
</div>
""", unsafe_allow_html=True)

# Model Performance Overview
st.markdown('<p class="section-title">🎯 Model Performance Overview</p>', unsafe_allow_html=True)
metrics = generate_model_performance()

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1: st.metric("MAE", f"{metrics['mae']:.1f} min")
with col2: st.metric("RMSE", f"{metrics['rmse']:.1f} min")
with col3: st.metric("R² Score", f"{metrics['r2']:.3f}")
with col4: st.metric("MAPE", f"{metrics['mape']:.1f}%")
with col5: st.metric("±15 min Acc", f"{metrics['acc_15']}%")
with col6: st.metric("±30 min Acc", f"{metrics['acc_30']}%")


show_all = analysis == "All Analyses"


# =============================================================================
# FEATURE IMPORTANCE (SHAP)
# =============================================================================
if show_all or analysis == "Feature Importance":
    st.markdown('<p class="section-title">🔍 Feature Importance (SHAP Values)</p>', unsafe_allow_html=True)
    
    shap = generate_shap_values()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        colors = ['#38a169' if '↓' in d else '#e53e3e' if '↑' in d else '#718096' for d in shap['Direction']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=shap['Feature'], x=shap['Importance'], orientation='h', marker_color=colors,
            text=[f"{i:.1%}" for i in shap['Importance']], textposition='outside', textfont=dict(color='white', size=11)
        ))
        fig.update_layout(height=450, margin=dict(l=150, r=60, t=40, b=40), xaxis_title="Importance",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        fig.update_yaxes(tickfont=dict(size=11, color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">🔬 Key Insights</strong><br><br>
            <span style="color: #e53e3e;">■</span> Red = Increases wait<br>
            <span style="color: #38a169;">■</span> Green = Decreases wait<br>
            <span style="color: #718096;">■</span> Gray = Variable<br><br>
            <strong style="color: white;">Top Predictors:</strong><br>
            <span style="color: #a0aec0;">
            1. Queue Length (25%)<br>
            2. Time of Day (18%)<br>
            3. Day of Week (12%)</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SURVIVAL ANALYSIS
# =============================================================================
if show_all or analysis == "Survival Analysis":
    st.markdown('<p class="section-title">📉 Survival Analysis - Time to Discharge</p>', unsafe_allow_html=True)
    
    survival = generate_survival_data()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        colors = {'ESI 1': '#e53e3e', 'ESI 2': '#ed8936', 'ESI 3': '#ecc94b', 'ESI 4': '#38a169', 'ESI 5': '#4299e1'}
        
        fig = go.Figure()
        for esi in ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5']:
            df = survival[survival['ESI'] == esi]
            fig.add_trace(go.Bar(x=df['time'], y=df['survival'], name=esi, marker_color=colors[esi], opacity=0.8))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=40, b=60), barmode='group',
            xaxis_title="Time in ED (min)", yaxis_title="Survival Probability",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white', size=11)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">📊 Median LOS by Acuity</strong><br><br>
            <span style="color: #a0aec0;">
            • ESI 1: ~240 min<br>
            • ESI 2: ~210 min<br>
            • ESI 3: ~180 min<br>
            • ESI 4: ~120 min<br>
            • ESI 5: ~60 min</span><br><br>
            <span style="color: #718096; font-size: 0.85rem;">
            Curves show probability of patient still in ED at each time.</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MODEL PERFORMANCE TRENDS
# =============================================================================
if show_all or analysis == "Model Performance":
    st.markdown('<p class="section-title">📈 Model Performance Trends</p>', unsafe_allow_html=True)
    
    perf = generate_prediction_accuracy()
    errors = generate_error_by_acuity()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf['month'], y=perf['mae'], name='MAE', marker_color='#667eea',
            text=[f"{m:.1f}" for m in perf['mae']], textposition='outside', textfont=dict(color='white', size=10)))
        fig.add_trace(go.Bar(x=perf['month'], y=perf['rmse'], name='RMSE', marker_color='#38a169',
            text=[f"{r:.1f}" for r in perf['rmse']], textposition='outside', textfont=dict(color='white', size=10)))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=60, b=60), barmode='group',
            xaxis_title="Month", yaxis_title="Error (minutes)",
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor='center', font=dict(color='white')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=errors['ESI'], y=errors['MAE'], marker_color='#805AD5',
            text=errors['MAE'], textposition='outside', textfont=dict(color='white', size=12)))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=60, b=60),
            xaxis_title="Acuity Level", yaxis_title="MAE (minutes)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(color='white', size=12))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual Analysis
    if show_residuals:
        st.markdown('<p class="section-title">📊 Residual Analysis</p>', unsafe_allow_html=True)
        residuals = generate_residual_analysis()
        
        col1, col2 = st.columns(2)
        with col1:
            hist, bins = np.histogram(residuals['Residual'], bins=30)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bins[:-1], y=hist, marker_color='#667eea'))
            fig.update_layout(height=300, margin=dict(l=60,r=40,t=40,b=60), xaxis_title="Residual (min)", yaxis_title="Frequency",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white'))
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            bins_pred = np.linspace(residuals['Predicted'].min(), residuals['Predicted'].max(), 10)
            residuals['bin'] = pd.cut(residuals['Predicted'], bins=bins_pred)
            grouped = residuals.groupby('bin')['Residual'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[str(b)[:10] for b in grouped['bin']], y=grouped['Residual'], marker_color='#38a169'))
            fig.add_hline(y=0, line_dash="dash", line_color="#e53e3e")
            fig.update_layout(height=300, margin=dict(l=60,r=40,t=40,b=80), xaxis_title="Predicted Range", yaxis_title="Mean Residual",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
            fig.update_xaxes(tickfont=dict(color='white', size=8), tickangle=-45)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PATIENT CLUSTERING
# =============================================================================
if show_all or analysis == "Patient Clustering":
    st.markdown('<p class="section-title">🎨 Patient Clustering Analysis</p>', unsafe_allow_html=True)
    
    clusters = generate_cluster_data()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=clusters['Cluster'], y=clusters['Size'],
            marker_color=clusters['Color'],
            text=clusters['Size'], textposition='outside', textfont=dict(color='white', size=12)
        ))
        
        fig.update_layout(height=400, margin=dict(l=60, r=40, t=40, b=100),
            xaxis_title="Patient Segment", yaxis_title="Patient Count",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(size=10, color='white'), tickangle=-20)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">🎯 5 Patient Segments</strong><br><br>
            <span style="color: #38a169;">■</span> <strong style="color: white;">Low Acuity</strong><br>
            <span style="color: #a0aec0;">Fast track, <1hr LOS</span><br><br>
            <span style="color: #667eea;">■</span> <strong style="color: white;">Moderate</strong><br>
            <span style="color: #a0aec0;">Standard care, 2-3hr</span><br><br>
            <span style="color: #ed8936;">■</span> <strong style="color: white;">Complex</strong><br>
            <span style="color: #a0aec0;">Extended workup, 4-5hr</span><br><br>
            <span style="color: #e53e3e;">■</span> <strong style="color: white;">Critical</strong><br>
            <span style="color: #a0aec0;">Resource intensive</span><br><br>
            <span style="color: #805AD5;">■</span> <strong style="color: white;">Observation</strong><br>
            <span style="color: #a0aec0;">Awaiting results, 3-4hr</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# CAUSAL INFERENCE
# =============================================================================
if show_all or analysis == "Causal Inference":
    st.markdown('<p class="section-title">🔬 Causal Inference - Intervention Effects</p>', unsafe_allow_html=True)
    
    causal = generate_causal_effects()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        colors = ['#38a169' if v < -5 else '#667eea' for v in causal['Effect on Wait']]
        fig.add_trace(go.Bar(
            x=causal['Intervention'], y=causal['Effect on Wait'].abs(),
            marker_color=colors,
            text=[f"-{abs(v):.1f} min" for v in causal['Effect on Wait']], 
            textposition='outside', textfont=dict(color='white', size=11)
        ))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=40, b=100),
            xaxis_title="Intervention", yaxis_title="Wait Time Reduction (min)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(size=10, color='white'), tickangle=-20)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        colors = ['#38a169' if v < -10 else '#667eea' for v in causal['Effect on LOS']]
        fig.add_trace(go.Bar(
            x=causal['Intervention'], y=causal['Effect on LOS'].abs(),
            marker_color=colors,
            text=[f"-{abs(v):.1f} min" for v in causal['Effect on LOS']], 
            textposition='outside', textfont=dict(color='white', size=11)
        ))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=40, b=100),
            xaxis_title="Intervention", yaxis_title="LOS Reduction (min)",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(size=10, color='white'), tickangle=-20)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Causal Summary Table
    st.markdown('<p class="section-title">📊 Intervention Impact Summary</p>', unsafe_allow_html=True)
    
    st.dataframe(
        causal.style.format({
            'Effect on Wait': '{:.1f} min',
            'Effect on LOS': '{:.1f} min', 
            'Effect on LWBS': '{:.1f}%',
            'Confidence': '{:.0f}%'
        }),
        use_container_width=True, hide_index=True
    )


# =============================================================================
# FEATURE INTERACTIONS
# =============================================================================
if show_all or analysis == "Feature Interactions":
    st.markdown('<p class="section-title">🔗 Feature Interactions</p>', unsafe_allow_html=True)
    
    interactions = generate_feature_interactions()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        colors = ['#38a169' if d == 'Synergistic' else '#e53e3e' for d in interactions['Impact Direction']]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=interactions['Feature Pair'], y=interactions['Interaction Strength'],
            marker_color=colors,
            text=[f"{s:.2f}" for s in interactions['Interaction Strength']],
            textposition='outside', textfont=dict(color='white', size=12)
        ))
        
        fig.update_layout(height=380, margin=dict(l=60, r=40, t=40, b=80),
            xaxis_title="Feature Pair", yaxis_title="Interaction Strength",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
        fig.update_xaxes(tickfont=dict(size=11, color='white'), tickangle=-15)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">🔗 Interaction Types</strong><br><br>
            <span style="color: #38a169;">■</span> <strong style="color: white;">Synergistic</strong><br>
            <span style="color: #a0aec0;">Features reinforce each other's effect</span><br><br>
            <span style="color: #e53e3e;">■</span> <strong style="color: white;">Antagonistic</strong><br>
            <span style="color: #a0aec0;">Features counteract each other</span><br><br>
            <span style="color: #718096; font-size: 0.85rem;">
            Strongest interaction: Queue × Time of Day (0.85)</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================
if show_all or analysis == "Sensitivity Analysis":
    st.markdown('<p class="section-title">📊 Sensitivity Analysis</p>', unsafe_allow_html=True)
    
    sensitivity = generate_sensitivity_analysis()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='+10% Change', x=sensitivity['Parameter'], y=sensitivity['+10% Effect'],
        marker_color='#38a169', text=[f"{v:+.1f}%" for v in sensitivity['+10% Effect']],
        textposition='outside', textfont=dict(color='white', size=11)
    ))
    fig.add_trace(go.Bar(
        name='-10% Change', x=sensitivity['Parameter'], y=sensitivity['-10% Effect'],
        marker_color='#e53e3e', text=[f"{v:+.1f}%" for v in sensitivity['-10% Effect']],
        textposition='outside', textfont=dict(color='white', size=11)
    ))
    
    fig.update_layout(height=400, margin=dict(l=60, r=40, t=60, b=80), barmode='group',
        xaxis_title="Parameter", yaxis_title="Effect on Wait Time (%)",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor='center', font=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color='white'))
    fig.update_xaxes(tickfont=dict(size=11, color='white'), tickangle=-15)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-card">
        <strong style="color: white;">📌 Key Finding</strong><br>
        <span style="color: #a0aec0;">Arrival Rate has the highest sensitivity - a 10% increase leads to 12.5% longer wait times, 
        making demand management the most impactful lever for wait time reduction.</span>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div class="footer">
    <strong>ER Patient Flow Intelligence Platform v2.0</strong><br>
    <span style="color: #a0aec0;">Built by <strong style="color: #667eea;">Neel, Harsh, and Tanishk</strong></span>
</div>
""", unsafe_allow_html=True)
