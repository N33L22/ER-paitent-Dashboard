"""
Fairness & Evaluation Matrix - Model Performance, Bias Detection, and Comparison
ER Patient Flow Intelligence Platform v2.0

Authors: Neel, Harsh, Tanishk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Fairness & Evaluation | ER Intelligence",
    page_icon="⚖️",
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
        background: linear-gradient(135deg, #ED8936 0%, #DD6B20 50%, #C05621 100%);
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
        border-bottom: 2px solid #ED8936;
        display: inline-block;
    }
    
    .insight-card {
        background: rgba(237, 137, 54, 0.15);
        border-left: 4px solid #ED8936;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .fairness-card {
        background: rgba(56, 161, 105, 0.15);
        border-left: 4px solid #38a169;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: rgba(229, 62, 62, 0.15);
        border-left: 4px solid #e53e3e;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .grade-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .grade-a { background: #38a169; color: white; }
    .grade-b { background: #68D391; color: #1a1f2e; }
    .grade-c { background: #ECC94B; color: #1a1f2e; }
    .grade-d { background: #ED8936; color: white; }
    .grade-f { background: #e53e3e; color: white; }
    
    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
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
# DATA GENERATION FUNCTIONS
# =============================================================================

@st.cache_data(ttl=120)
def generate_model_predictions():
    """Generate sample model predictions for evaluation"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate ground truth
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate
    
    # Generate predictions with some noise
    y_pred_proba = np.clip(y_true * 0.7 + np.random.normal(0.15, 0.2, n_samples), 0, 1)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate protected attributes
    age_groups = np.random.choice(['18-40', '41-64', '65+'], n_samples, p=[0.3, 0.4, 0.3])
    sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
    insurance = np.random.choice(['Private', 'Medicare', 'Medicaid', 'Uninsured'], 
                                  n_samples, p=[0.45, 0.25, 0.20, 0.10])
    acuity = np.random.choice(['ESI 1-2', 'ESI 3', 'ESI 4-5'], n_samples, p=[0.15, 0.50, 0.35])
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'age_group': age_groups,
        'sex': sex,
        'insurance': insurance,
        'acuity': acuity,
        'n_samples': n_samples
    }


@st.cache_data(ttl=120)
def generate_confusion_matrix_data():
    """Generate confusion matrix data"""
    data = generate_model_predictions()
    
    # Calculate confusion matrix
    tp = np.sum((data['y_true'] == 1) & (data['y_pred'] == 1))
    tn = np.sum((data['y_true'] == 0) & (data['y_pred'] == 0))
    fp = np.sum((data['y_true'] == 0) & (data['y_pred'] == 1))
    fn = np.sum((data['y_true'] == 1) & (data['y_pred'] == 0))
    
    return {
        'matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


@st.cache_data(ttl=120)
def generate_model_comparison():
    """Generate model comparison data"""
    models = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Neural Network', 'LightGBM']
    
    return pd.DataFrame({
        'Model': models,
        'Accuracy': [0.847, 0.823, 0.789, 0.831, 0.852],
        'Precision': [0.823, 0.801, 0.756, 0.812, 0.834],
        'Recall': [0.789, 0.756, 0.723, 0.778, 0.798],
        'F1 Score': [0.806, 0.778, 0.739, 0.795, 0.816],
        'ROC-AUC': [0.891, 0.867, 0.834, 0.876, 0.897],
        'Training Time (s)': [12.3, 8.5, 1.2, 45.6, 10.1],
        'Inference Time (ms)': [2.1, 3.4, 0.5, 8.2, 1.8]
    })


@st.cache_data(ttl=120)
def generate_roc_curves():
    """Generate ROC curve data for multiple models"""
    np.random.seed(42)
    
    fpr_base = np.linspace(0, 1, 100)
    
    models = {
        'XGBoost': {'auc': 0.891, 'color': '#667eea'},
        'Random Forest': {'auc': 0.867, 'color': '#38a169'},
        'Logistic Regression': {'auc': 0.834, 'color': '#ED8936'},
        'Neural Network': {'auc': 0.876, 'color': '#e53e3e'},
        'LightGBM': {'auc': 0.897, 'color': '#805AD5'}
    }
    
    curves = {}
    for model, props in models.items():
        # Generate realistic ROC curve
        tpr = 1 - (1 - fpr_base) ** (1 / (1 - props['auc'] + 0.1))
        tpr = np.clip(tpr + np.random.normal(0, 0.02, len(tpr)), 0, 1)
        tpr = np.sort(tpr)
        tpr[0] = 0
        tpr[-1] = 1
        curves[model] = {'fpr': fpr_base, 'tpr': tpr, **props}
    
    return curves


@st.cache_data(ttl=120)
def generate_fairness_metrics():
    """Generate fairness metrics across protected attributes"""
    data = generate_model_predictions()
    
    fairness_results = {}
    
    for attr in ['age_group', 'sex', 'insurance', 'acuity']:
        groups = np.unique(data[attr])
        group_metrics = {}
        
        for group in groups:
            mask = data[attr] == group
            y_true_g = data['y_true'][mask]
            y_pred_g = data['y_pred'][mask]
            
            tp = np.sum((y_true_g == 1) & (y_pred_g == 1))
            tn = np.sum((y_true_g == 0) & (y_pred_g == 0))
            fp = np.sum((y_true_g == 0) & (y_pred_g == 1))
            fn = np.sum((y_true_g == 1) & (y_pred_g == 0))
            
            group_metrics[group] = {
                'count': int(mask.sum()),
                'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'positive_rate': (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            }
        
        fairness_results[attr] = group_metrics
    
    return fairness_results


@st.cache_data(ttl=120)
def calculate_fairness_score():
    """Calculate overall fairness score and identify disparities"""
    metrics = generate_fairness_metrics()
    
    disparities = []
    total_checks = 0
    fair_checks = 0
    
    for attr, groups in metrics.items():
        group_names = list(groups.keys())
        if len(group_names) < 2:
            continue
            
        reference = group_names[0]
        ref_metrics = groups[reference]
        
        for group in group_names[1:]:
            comp_metrics = groups[group]
            
            for metric in ['accuracy', 'tpr', 'fpr', 'positive_rate']:
                total_checks += 1
                ref_val = ref_metrics[metric]
                comp_val = comp_metrics[metric]
                
                ratio = comp_val / (ref_val + 1e-10)
                diff = abs(comp_val - ref_val)
                
                is_fair = 0.8 <= ratio <= 1.25 and diff < 0.1
                if is_fair:
                    fair_checks += 1
                else:
                    disparities.append({
                        'attribute': attr,
                        'reference': reference,
                        'comparison': group,
                        'metric': metric,
                        'ref_value': ref_val,
                        'comp_value': comp_val,
                        'ratio': ratio,
                        'severity': 'High' if diff > 0.15 else 'Medium' if diff > 0.1 else 'Low'
                    })
    
    score = (fair_checks / total_checks * 100) if total_checks > 0 else 100
    
    if score >= 90:
        grade = 'A'
    elif score >= 80:
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    
    return {
        'score': score,
        'grade': grade,
        'total_checks': total_checks,
        'fair_checks': fair_checks,
        'disparities': disparities
    }


@st.cache_data(ttl=120)
def generate_bias_over_time():
    """Generate bias metrics over time"""
    months = ['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
    
    return pd.DataFrame({
        'Month': months,
        'Age Parity': [0.82, 0.85, 0.88, 0.87, 0.89, 0.91],
        'Sex Parity': [0.94, 0.93, 0.95, 0.96, 0.95, 0.97],
        'Insurance Parity': [0.75, 0.78, 0.80, 0.82, 0.85, 0.87],
        'Acuity Parity': [0.88, 0.87, 0.89, 0.90, 0.91, 0.92]
    })


@st.cache_data(ttl=120)
def generate_calibration_data():
    """Generate calibration curve data"""
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Well-calibrated model (close to diagonal)
    perfect = bin_centers
    actual = bin_centers + np.random.normal(0, 0.03, len(bin_centers))
    actual = np.clip(actual, 0, 1)
    
    return pd.DataFrame({
        'bin': bin_centers,
        'perfect': perfect,
        'actual': actual
    })


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem;">⚖️</div>
    <h2 style="color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;">Fairness & Evaluation</h2>
    <p style="color: #a0aec0; font-size: 0.8rem;">Model Metrics & Bias Analysis</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Type")
analysis_type = st.sidebar.selectbox(
    "Select View",
    ["Complete Dashboard", "Confusion Matrix", "Model Comparison", 
     "Fairness Analysis", "ROC Curves", "Calibration"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Model Selection")
primary_model = st.sidebar.selectbox(
    "Primary Model",
    ["XGBoost (Production)", "LightGBM", "Random Forest", "Neural Network", "Logistic Regression"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Fairness Settings")
fairness_threshold = st.sidebar.slider("Fairness Threshold (4/5 rule)", 0.7, 0.9, 0.8)
show_all_disparities = st.sidebar.checkbox("Show All Disparities", value=False)

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
    <h1 class="page-title">⚖️ Fairness & Evaluation Matrix</h1>
    <p class="page-subtitle">Model Performance Metrics • Bias Detection • Multi-Model Comparison</p>
</div>
""", unsafe_allow_html=True)

show_all = analysis_type == "Complete Dashboard"


# =============================================================================
# FAIRNESS SCORECARD OVERVIEW
# =============================================================================

if show_all or analysis_type == "Fairness Analysis":
    st.markdown('<p class="section-title">📋 Fairness Scorecard</p>', unsafe_allow_html=True)
    
    fairness_score = calculate_fairness_score()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        grade_class = f"grade-{fairness_score['grade'].lower()}"
        st.markdown(f"""
        <div class="metric-box">
            <div class="grade-badge {grade_class}">{fairness_score['grade']}</div>
            <p style="color: #a0aec0; margin-top: 0.5rem; margin-bottom: 0;">Fairness Grade</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Fairness Score", f"{fairness_score['score']:.1f}%", 
                  delta="+3.2%" if fairness_score['score'] > 80 else "-2.1%")
    
    with col3:
        st.metric("Checks Passed", f"{fairness_score['fair_checks']}/{fairness_score['total_checks']}")
    
    with col4:
        st.metric("Disparities Found", len(fairness_score['disparities']),
                  delta=f"-2" if len(fairness_score['disparities']) < 5 else None, delta_color="inverse")
    
    with col5:
        high_sev = len([d for d in fairness_score['disparities'] if d['severity'] == 'High'])
        st.metric("High Severity", high_sev, delta=None, delta_color="inverse")


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

if show_all or analysis_type == "Confusion Matrix":
    st.markdown('<p class="section-title">🔢 Confusion Matrix & Classification Metrics</p>', unsafe_allow_html=True)
    
    cm_data = generate_confusion_matrix_data()
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Interactive Confusion Matrix
        cm = np.array(cm_data['matrix'])
        labels = ['Negative (0)', 'Positive (1)']
        
        # Create annotations with counts and percentages
        total = cm.sum()
        annotations = []
        for i in range(2):
            for j in range(2):
                val = cm[i][j]
                pct = val / total * 100
                text = f"{val}<br>({pct:.1f}%)"
                annotations.append(
                    dict(
                        x=labels[j], y=labels[i],
                        text=text,
                        showarrow=False,
                        font=dict(size=16, color='white')
                    )
                )
        
        # Color scale: TN/TP are green, FP/FN are red
        colors = [[0, '#38a169'], [0.5, '#718096'], [1, '#38a169']]
        z_normalized = [[0.8, 0.2], [0.2, 0.8]]  # For coloring
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, '#1a1f2e'], [0.5, '#4a5568'], [1, '#38a169']],
            showscale=False,
            hoverongaps=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text='Confusion Matrix', font=dict(color='white', size=16)),
            xaxis=dict(title='Predicted Label', tickfont=dict(color='white'), title_font=dict(color='white')),
            yaxis=dict(title='Actual Label', tickfont=dict(color='white'), title_font=dict(color='white')),
            annotations=annotations,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">📊 Matrix Breakdown</strong>
        </div>
        """, unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)'],
            'Count': [cm_data['tp'], cm_data['tn'], cm_data['fp'], cm_data['fn']],
            'Description': ['Correctly predicted positive', 'Correctly predicted negative', 
                           'Type I Error', 'Type II Error']
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    # Classification Metrics
    st.markdown('<p class="section-title">📈 Classification Performance Metrics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{cm_data['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{cm_data['precision']:.1%}")
    with col3:
        st.metric("Recall (Sensitivity)", f"{cm_data['recall']:.1%}")
    with col4:
        st.metric("Specificity", f"{cm_data['specificity']:.1%}")
    with col5:
        st.metric("F1 Score", f"{cm_data['f1']:.3f}")
    
    # Metrics explanation
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="fairness-card">
            <strong style="color: white;">✅ Strengths</strong><br>
            <span style="color: #a0aec0;">
            • High specificity ({:.1%}) - Good at identifying negatives<br>
            • Balanced accuracy across classes<br>
            • Low false positive rate minimizes unnecessary alerts
            </span>
        </div>
        """.format(cm_data['specificity']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
            <strong style="color: white;">⚠️ Areas for Improvement</strong><br>
            <span style="color: #a0aec0;">
            • Recall could be improved for critical cases<br>
            • Consider threshold adjustment for high-risk patients<br>
            • Monitor false negatives in ESI 1-2 patients
            </span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

if show_all or analysis_type == "Model Comparison":
    st.markdown('<p class="section-title">🏆 Multi-Model Performance Comparison</p>', unsafe_allow_html=True)
    
    comparison_df = generate_model_comparison()
    
    # Radar chart for model comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        colors = {'XGBoost': '#667eea', 'Random Forest': '#38a169', 
                  'Logistic Regression': '#ED8936', 'Neural Network': '#e53e3e', 'LightGBM': '#805AD5'}
        
        for _, row in comparison_df.iterrows():
            values = [row['Accuracy'], row['Precision'], row['Recall'], row['F1 Score'], row['ROC-AUC']]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                name=row['Model'],
                fill='toself',
                opacity=0.6,
                line=dict(color=colors.get(row['Model'], '#718096'))
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.7, 1], tickfont=dict(color='white')),
                angularaxis=dict(tickfont=dict(color='white'))
            ),
            showlegend=True,
            legend=dict(font=dict(color='white'), orientation='h', y=-0.1, x=0.5, xanchor='center'),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">🏆 Model Rankings</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Sort by F1 Score
        ranked = comparison_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            st.markdown(f"{medal} **{row['Model']}** - F1: {row['F1 Score']:.3f}")
    
    # Detailed comparison table
    st.markdown('<p class="section-title">📊 Detailed Metrics Table</p>', unsafe_allow_html=True)
    
    # Style the dataframe
    styled_df = comparison_df.style.format({
        'Accuracy': '{:.1%}',
        'Precision': '{:.1%}',
        'Recall': '{:.1%}',
        'F1 Score': '{:.3f}',
        'ROC-AUC': '{:.3f}',
        'Training Time (s)': '{:.1f}',
        'Inference Time (ms)': '{:.1f}'
    }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'], 
                           cmap='Greens', vmin=0.7, vmax=1.0)
    
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    # Performance vs Speed tradeoff
    st.markdown('<p class="section-title">⚡ Performance vs Speed Tradeoff</p>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=comparison_df['Inference Time (ms)'],
        y=comparison_df['F1 Score'],
        mode='markers+text',
        marker=dict(size=comparison_df['Training Time (s)'] * 2 + 10,
                   color=['#667eea', '#38a169', '#ED8936', '#e53e3e', '#805AD5'],
                   opacity=0.7),
        text=comparison_df['Model'],
        textposition='top center',
        textfont=dict(color='white', size=11),
        hovertemplate="<b>%{text}</b><br>F1: %{y:.3f}<br>Inference: %{x:.1f}ms<extra></extra>"
    ))
    
    fig.update_layout(
        xaxis=dict(title='Inference Time (ms)', tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(title='F1 Score', tickfont=dict(color='white'), title_font=dict(color='white'), range=[0.7, 0.85]),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color='white')
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ROC CURVES
# =============================================================================

if show_all or analysis_type == "ROC Curves":
    st.markdown('<p class="section-title">📈 ROC Curves & AUC Comparison</p>', unsafe_allow_html=True)
    
    roc_data = generate_roc_curves()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(dash='dash', color='#718096')
        ))
        
        # Add ROC curves for each model
        for model, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                mode='lines',
                name=f"{model} (AUC = {data['auc']:.3f})",
                line=dict(color=data['color'], width=2),
                fill='tonexty' if model == 'LightGBM' else None,
                fillcolor='rgba(128, 90, 213, 0.1)' if model == 'LightGBM' else None
            ))
        
        fig.update_layout(
            xaxis=dict(title='False Positive Rate', tickfont=dict(color='white'), 
                      title_font=dict(color='white'), range=[0, 1]),
            yaxis=dict(title='True Positive Rate', tickfont=dict(color='white'), 
                      title_font=dict(color='white'), range=[0, 1]),
            legend=dict(font=dict(color='white', size=10), x=0.6, y=0.05),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">📊 AUC Interpretation</strong><br><br>
            <span style="color: #a0aec0;">
            • <strong style="color: #38a169;">0.9-1.0</strong>: Excellent<br>
            • <strong style="color: #68D391;">0.8-0.9</strong>: Good<br>
            • <strong style="color: #ECC94B;">0.7-0.8</strong>: Fair<br>
            • <strong style="color: #ED8936;">0.6-0.7</strong>: Poor<br>
            • <strong style="color: #e53e3e;">0.5-0.6</strong>: Fail
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # AUC ranking
        st.markdown("**AUC Rankings:**")
        sorted_models = sorted(roc_data.items(), key=lambda x: x[1]['auc'], reverse=True)
        for i, (model, data) in enumerate(sorted_models):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            st.markdown(f"{medal} {model}: **{data['auc']:.3f}**")


# =============================================================================
# FAIRNESS ANALYSIS DETAILED
# =============================================================================

if show_all or analysis_type == "Fairness Analysis":
    st.markdown('<p class="section-title">🔍 Detailed Fairness Analysis by Protected Attribute</p>', unsafe_allow_html=True)
    
    fairness_data = generate_fairness_metrics()
    
    # Create tabs for each attribute
    tabs = st.tabs(["👥 Age Group", "⚧ Sex", "💳 Insurance", "🏥 Acuity Level"])
    
    for tab, (attr, display_name) in zip(tabs, [('age_group', 'Age Group'), ('sex', 'Sex'), 
                                                  ('insurance', 'Insurance'), ('acuity', 'Acuity')]):
        with tab:
            group_data = fairness_data[attr]
            groups = list(group_data.keys())
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy by group
                accuracies = [group_data[g]['accuracy'] for g in groups]
                counts = [group_data[g]['count'] for g in groups]
                
                fig = go.Figure()
                
                # Determine colors based on fairness
                max_acc = max(accuracies)
                min_acc = min(accuracies)
                colors = ['#38a169' if (max_acc - a) < 0.05 else '#ED8936' if (max_acc - a) < 0.1 else '#e53e3e' 
                         for a in accuracies]
                
                fig.add_trace(go.Bar(
                    x=groups, y=accuracies,
                    marker_color=colors,
                    text=[f"{a:.1%}" for a in accuracies],
                    textposition='outside',
                    textfont=dict(color='white', size=12),
                    hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<br>Samples: " + 
                                  "<br>".join([f"{c}" for c in counts]) + "<extra></extra>"
                ))
                
                # Add fairness threshold line
                fig.add_hline(y=max_acc * 0.8, line_dash="dash", line_color="#ED8936",
                             annotation_text="4/5 Rule Threshold", annotation_font_color="white")
                
                fig.update_layout(
                    title=dict(text=f'Accuracy by {display_name}', font=dict(color='white', size=14)),
                    xaxis=dict(tickfont=dict(color='white')),
                    yaxis=dict(title='Accuracy', tickfont=dict(color='white'), range=[0.6, 1]),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='white')
                )
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # TPR/FPR comparison
                tprs = [group_data[g]['tpr'] for g in groups]
                fprs = [group_data[g]['fpr'] for g in groups]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='True Positive Rate',
                    x=groups, y=tprs,
                    marker_color='#38a169',
                    text=[f"{t:.1%}" for t in tprs],
                    textposition='outside',
                    textfont=dict(color='white', size=10)
                ))
                
                fig.add_trace(go.Bar(
                    name='False Positive Rate',
                    x=groups, y=fprs,
                    marker_color='#e53e3e',
                    text=[f"{f:.1%}" for f in fprs],
                    textposition='outside',
                    textfont=dict(color='white', size=10)
                ))
                
                fig.update_layout(
                    title=dict(text=f'TPR vs FPR by {display_name}', font=dict(color='white', size=14)),
                    barmode='group',
                    xaxis=dict(tickfont=dict(color='white')),
                    yaxis=dict(title='Rate', tickfont=dict(color='white')),
                    legend=dict(font=dict(color='white'), orientation='h', y=1.15),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", color='white')
                )
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Group statistics table
            st.markdown(f"**{display_name} Group Statistics:**")
            stats_df = pd.DataFrame([
                {
                    'Group': g,
                    'Sample Size': group_data[g]['count'],
                    'Accuracy': f"{group_data[g]['accuracy']:.1%}",
                    'TPR (Recall)': f"{group_data[g]['tpr']:.1%}",
                    'FPR': f"{group_data[g]['fpr']:.1%}",
                    'Precision': f"{group_data[g]['precision']:.1%}",
                    'Positive Rate': f"{group_data[g]['positive_rate']:.1%}"
                }
                for g in groups
            ])
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # Disparities found
    fairness_score = calculate_fairness_score()
    
    if fairness_score['disparities']:
        st.markdown('<p class="section-title">⚠️ Identified Disparities</p>', unsafe_allow_html=True)
        
        disp_df = pd.DataFrame(fairness_score['disparities'])
        
        # Filter by severity if not showing all
        if not show_all_disparities:
            disp_df = disp_df[disp_df['severity'].isin(['High', 'Medium'])]
        
        if len(disp_df) > 0:
            disp_df['ratio_display'] = disp_df['ratio'].apply(lambda x: f"{x:.2f}")
            disp_df['ref_display'] = disp_df['ref_value'].apply(lambda x: f"{x:.1%}")
            disp_df['comp_display'] = disp_df['comp_value'].apply(lambda x: f"{x:.1%}")
            
            display_df = disp_df[['attribute', 'metric', 'reference', 'comparison', 
                                   'ref_display', 'comp_display', 'ratio_display', 'severity']]
            display_df.columns = ['Attribute', 'Metric', 'Reference Group', 'Comparison Group',
                                  'Ref Value', 'Comp Value', 'Ratio', 'Severity']
            
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # Recommendations
        st.markdown('<p class="section-title">💡 Recommendations</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="fairness-card">
                <strong style="color: white;">✅ Mitigation Strategies</strong><br><br>
                <span style="color: #a0aec0;">
                • Re-weight training samples for underrepresented groups<br>
                • Apply adversarial debiasing techniques<br>
                • Use threshold adjustment per demographic group<br>
                • Collect more balanced training data<br>
                • Consider fairness constraints in model training
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-card">
                <strong style="color: white;">📋 Monitoring Actions</strong><br><br>
                <span style="color: #a0aec0;">
                • Set up automated fairness monitoring pipeline<br>
                • Track disparity metrics over time<br>
                • Alert when metrics cross thresholds<br>
                • Regular bias audits (monthly recommended)<br>
                • Document and explain any remaining disparities
                </span>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# CALIBRATION
# =============================================================================

if show_all or analysis_type == "Calibration":
    st.markdown('<p class="section-title">🎯 Model Calibration Analysis</p>', unsafe_allow_html=True)
    
    cal_data = generate_calibration_data()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=cal_data['bin'], y=cal_data['perfect'],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='#718096')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=cal_data['bin'], y=cal_data['actual'],
            mode='lines+markers',
            name='XGBoost Model',
            line=dict(color='#667eea', width=2),
            marker=dict(size=8)
        ))
        
        # Fill area between
        fig.add_trace(go.Scatter(
            x=list(cal_data['bin']) + list(cal_data['bin'][::-1]),
            y=list(cal_data['perfect']) + list(cal_data['actual'][::-1]),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            xaxis=dict(title='Predicted Probability', tickfont=dict(color='white'), 
                      title_font=dict(color='white')),
            yaxis=dict(title='Actual Frequency', tickfont=dict(color='white'), 
                      title_font=dict(color='white')),
            legend=dict(font=dict(color='white')),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card">
            <strong style="color: white;">📊 Calibration Metrics</strong><br><br>
            <span style="color: #a0aec0;">
            <strong>Brier Score:</strong> 0.142<br>
            <strong>ECE:</strong> 0.031<br>
            <strong>MCE:</strong> 0.058<br><br>
            <span style="font-size: 0.85rem;">
            A well-calibrated model has predictions that match actual frequencies.
            The closer the curve is to the diagonal, the better the calibration.
            </span>
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="fairness-card">
            <strong style="color: white;">✅ Calibration Status</strong><br>
            <span style="color: #a0aec0;">
            Model is well-calibrated with minor over-confidence in the 0.6-0.8 probability range.
            </span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# BIAS TRENDS OVER TIME
# =============================================================================

if show_all:
    st.markdown('<p class="section-title">📈 Fairness Trends Over Time</p>', unsafe_allow_html=True)
    
    bias_trends = generate_bias_over_time()
    
    fig = go.Figure()
    
    colors = {'Age Parity': '#667eea', 'Sex Parity': '#38a169', 
              'Insurance Parity': '#ED8936', 'Acuity Parity': '#805AD5'}
    
    for col in ['Age Parity', 'Sex Parity', 'Insurance Parity', 'Acuity Parity']:
        fig.add_trace(go.Scatter(
            x=bias_trends['Month'], y=bias_trends[col],
            mode='lines+markers',
            name=col,
            line=dict(color=colors[col], width=2),
            marker=dict(size=8)
        ))
    
    # Add threshold line
    fig.add_hline(y=0.8, line_dash="dash", line_color="#e53e3e",
                  annotation_text="Fairness Threshold (0.8)", annotation_font_color="white")
    
    fig.update_layout(
        xaxis=dict(title='Month', tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(title='Parity Ratio', tickfont=dict(color='white'), 
                  title_font=dict(color='white'), range=[0.7, 1.0]),
        legend=dict(font=dict(color='white'), orientation='h', y=-0.15, x=0.5, xanchor='center'),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color='white')
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="insight-card">
        <strong style="color: white;">📌 Trend Analysis</strong><br>
        <span style="color: #a0aec0;">
        All fairness metrics show improvement over the past 6 months. Insurance parity 
        has shown the most improvement (+16%), while sex parity remains consistently high.
        The team should continue monitoring age parity as it approaches but hasn't yet 
        crossed the 0.9 threshold for Grade A fairness.
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="footer">
    <strong>ER Patient Flow Intelligence Platform v2.0</strong><br>
    <span style="color: #a0aec0;">Fairness & Evaluation Matrix | Built by <strong style="color: #667eea;">Neel, Harsh, and Tanishk</strong></span>
</div>
""", unsafe_allow_html=True)
