"""
Production-Grade Visualization Base Classes
ER Patient Flow Intelligence Platform

Features:
- Responsive layouts (no truncation)
- Automated insight extraction
- Interactive annotations
- Publication-ready styling
- Accessibility (WCAG AAA)

Authors: Neel, Harsh, Tanishk
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class ChartInsight:
    """Structured insight from chart analysis"""
    icon: str
    title: str
    message: str
    severity: str = "info"  # info, warning, critical, success
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    
    def to_markdown(self) -> str:
        return f"{self.icon} **{self.title}**: {self.message}"


class ProductionVisualizer:
    """
    Base class for production-grade visualizations
    
    Features:
    1. Responsive layouts (no truncation)
    2. Automated insight extraction
    3. Interactive annotations
    4. Publication-ready styling
    5. Accessibility (WCAG AAA)
    """
    
    # Professional color palettes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'info': '#17becf',
        'neutral': '#7f7f7f',
        'light': '#f0f0f0',
        'dark': '#2c3e50',
        # Sequential palettes
        'sequential': px.colors.sequential.Blues,
        'sequential_warm': px.colors.sequential.Oranges,
        'sequential_cool': px.colors.sequential.Purples,
        # Diverging palettes
        'diverging': px.colors.diverging.RdYlGn,
        'diverging_rdbu': px.colors.diverging.RdBu,
        # Qualitative palettes
        'qualitative': px.colors.qualitative.Set2,
        'qualitative_bold': px.colors.qualitative.Bold,
        # Acuity colors (ESI 1-5)
        'acuity': {
            1: '#d62728',  # Red - Resuscitation
            2: '#ff7f0e',  # Orange - Emergent
            3: '#ffbb33',  # Yellow - Urgent
            4: '#2ca02c',  # Green - Less Urgent
            5: '#1f77b4'   # Blue - Non-Urgent
        },
        # Status colors
        'status': {
            'good': '#00C851',
            'warning': '#ffbb33',
            'danger': '#ff4444',
            'neutral': '#666666'
        }
    }
    
    # Standard layout settings for consistency
    LAYOUT_CONFIG = {
        'font': {
            'family': 'Arial, Helvetica, sans-serif',
            'size': 12,
            'color': '#2c3e50'
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80},
        'hovermode': 'closest',
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5,
            'bgcolor': 'rgba(255,255,255,0.8)',
            'bordercolor': '#ddd',
            'borderwidth': 1
        }
    }
    
    def __init__(self):
        self.insight_engine = InsightEngine()
    
    def create_responsive_layout(
        self,
        title: str,
        subtitle: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        show_legend: bool = True
    ) -> Dict:
        """
        Create responsive layout that prevents truncation
        
        Args:
            title: Main chart title
            subtitle: Optional subtitle
            height: Fixed height or None for auto
            width: Fixed width or None for auto (responsive)
            show_legend: Whether to show legend
        
        Returns:
            Layout dictionary for Plotly
        """
        layout = self.LAYOUT_CONFIG.copy()
        layout['font'] = self.LAYOUT_CONFIG['font'].copy()
        layout['legend'] = self.LAYOUT_CONFIG['legend'].copy()
        layout['margin'] = self.LAYOUT_CONFIG['margin'].copy()
        
        # Title configuration
        if subtitle:
            title_text = f"<b>{title}</b><br><span style='font-size:12px;color:#666'>{subtitle}</span>"
        else:
            title_text = f"<b>{title}</b>"
        
        layout['title'] = {
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        }
        
        # Size handling
        if height:
            layout['height'] = height
        if width:
            layout['width'] = width
        else:
            layout['autosize'] = True
        
        layout['showlegend'] = show_legend
        
        return layout
    
    def get_axis_config(
        self,
        title: str,
        show_grid: bool = True,
        zero_line: bool = False,
        tick_format: Optional[str] = None
    ) -> Dict:
        """Create consistent axis configuration"""
        config = {
            'title': {'text': title, 'font': {'size': 12}},
            'gridcolor': 'rgba(200,200,200,0.3)',
            'showgrid': show_grid,
            'zeroline': zero_line,
            'zerolinecolor': 'rgba(0,0,0,0.2)',
            'linecolor': 'rgba(0,0,0,0.2)',
            'linewidth': 1
        }
        
        if tick_format:
            config['tickformat'] = tick_format
        
        return config
    
    def get_hover_template(self, fields: List[Tuple[str, str]]) -> str:
        """
        Generate consistent hover template
        
        Args:
            fields: List of (label, format) tuples, e.g., [('Time', '%{x}'), ('Value', '%{y:.1f}')]
        """
        lines = [f"<b>{label}</b>: {fmt}" for label, fmt in fields]
        return "<br>".join(lines) + "<extra></extra>"
    
    def add_threshold_line(
        self,
        fig: go.Figure,
        value: float,
        label: str,
        color: str = 'gray',
        dash: str = 'dot',
        annotation_position: str = 'right'
    ) -> go.Figure:
        """Add a horizontal threshold line with annotation"""
        fig.add_hline(
            y=value,
            line_dash=dash,
            line_color=color,
            line_width=1.5,
            annotation_text=label,
            annotation_position=annotation_position,
            annotation_font_size=10
        )
        return fig
    
    def add_annotations(
        self,
        fig: go.Figure,
        annotations: List[Dict]
    ) -> go.Figure:
        """Add multiple annotations to figure"""
        for ann in annotations:
            fig.add_annotation(
                x=ann.get('x'),
                y=ann.get('y'),
                text=ann.get('text'),
                showarrow=ann.get('showarrow', True),
                arrowhead=ann.get('arrowhead', 2),
                ax=ann.get('ax', 0),
                ay=ann.get('ay', -40),
                font=dict(size=ann.get('font_size', 10), color=ann.get('color', '#333'))
            )
        return fig
    
    def extract_insights(
        self,
        data: pd.DataFrame,
        viz_type: str
    ) -> List[ChartInsight]:
        """
        Automatically extract insights from data
        
        Args:
            data: DataFrame to analyze
            viz_type: Type of visualization ('time_series', 'distribution', 'comparison', 'correlation')
        
        Returns:
            List of ChartInsight objects
        """
        return self.insight_engine.analyze(data, viz_type)
    
    def insights_to_markdown(self, insights: List[ChartInsight]) -> List[str]:
        """Convert insights to markdown strings"""
        return [i.to_markdown() for i in insights]


class InsightEngine:
    """
    AI-powered insight extraction from visualizations
    Provides automated statistical analysis and natural language insights
    """
    
    def analyze(self, data: pd.DataFrame, viz_type: str) -> List[ChartInsight]:
        """
        Generate automated insights based on visualization type
        
        Args:
            data: DataFrame to analyze
            viz_type: One of 'time_series', 'distribution', 'comparison', 'correlation', 'queue'
        
        Returns:
            List of ChartInsight objects
        """
        insights = []
        
        try:
            if viz_type == 'time_series':
                insights.extend(self._analyze_time_series(data))
            elif viz_type == 'distribution':
                insights.extend(self._analyze_distribution(data))
            elif viz_type == 'comparison':
                insights.extend(self._analyze_comparison(data))
            elif viz_type == 'correlation':
                insights.extend(self._analyze_correlation(data))
            elif viz_type == 'queue':
                insights.extend(self._analyze_queue(data))
            elif viz_type == 'flow':
                insights.extend(self._analyze_flow(data))
        except Exception as e:
            insights.append(ChartInsight(
                icon="⚠️",
                title="Analysis Note",
                message=f"Some insights could not be generated: {str(e)}",
                severity="warning"
            ))
        
        return insights
    
    def _analyze_time_series(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract time-series insights"""
        insights = []
        
        # Find first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return insights
        
        metric_col = numeric_cols[0]
        values = data[metric_col].dropna()
        
        if len(values) < 3:
            return insights
        
        # Trend detection using linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if p_value < 0.05:  # Significant trend
            pct_change = ((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100) if values.iloc[0] != 0 else 0
            
            if slope > 0:
                insights.append(ChartInsight(
                    icon="📈",
                    title="Upward Trend Detected",
                    message=f"{metric_col} increased by {abs(pct_change):.1f}% over the period (statistically significant, p={p_value:.3f})",
                    severity="info" if pct_change < 20 else "warning",
                    metric_name=metric_col,
                    metric_value=pct_change
                ))
            else:
                insights.append(ChartInsight(
                    icon="📉",
                    title="Downward Trend Detected",
                    message=f"{metric_col} decreased by {abs(pct_change):.1f}% over the period (statistically significant, p={p_value:.3f})",
                    severity="info" if abs(pct_change) < 20 else "success",
                    metric_name=metric_col,
                    metric_value=pct_change
                ))
        
        # Volatility analysis (coefficient of variation)
        mean_val = values.mean()
        if mean_val != 0:
            cv = values.std() / mean_val
            if cv > 0.3:
                insights.append(ChartInsight(
                    icon="⚠️",
                    title="High Volatility",
                    message=f"{metric_col} shows high variability (CV={cv:.2f}), indicating unstable patterns",
                    severity="warning",
                    metric_name="coefficient_of_variation",
                    metric_value=cv
                ))
        
        # Peak detection
        peak_idx = values.idxmax()
        peak_value = values.max()
        
        if peak_value > mean_val * 1.5:
            insights.append(ChartInsight(
                icon="🔺",
                title="Significant Peak",
                message=f"Maximum value ({peak_value:.1f}) is {(peak_value/mean_val):.1f}x the average, indicating a notable surge",
                severity="info",
                metric_name="peak_ratio",
                metric_value=peak_value/mean_val
            ))
        
        # Recent change detection
        if len(values) >= 2:
            recent_change = ((values.iloc[-1] - values.iloc[-2]) / values.iloc[-2] * 100) if values.iloc[-2] != 0 else 0
            if abs(recent_change) > 15:
                direction = "increased" if recent_change > 0 else "decreased"
                insights.append(ChartInsight(
                    icon="📊",
                    title="Recent Change",
                    message=f"Latest value {direction} by {abs(recent_change):.1f}% from previous period",
                    severity="warning" if abs(recent_change) > 25 else "info",
                    metric_name="recent_change_pct",
                    metric_value=recent_change
                ))
        
        return insights
    
    def _analyze_distribution(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract distribution insights"""
        insights = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return insights
        
        metric_col = numeric_cols[0]
        values = data[metric_col].dropna()
        
        if len(values) < 5:
            return insights
        
        # Central tendency analysis
        mean_val = values.mean()
        median_val = values.median()
        
        if abs(mean_val - median_val) / max(median_val, 0.001) > 0.2:
            if mean_val > median_val:
                insights.append(ChartInsight(
                    icon="📊",
                    title="Right-Skewed Distribution",
                    message=f"Mean ({mean_val:.1f}) > Median ({median_val:.1f}), indicating outliers on the high end",
                    severity="info"
                ))
            else:
                insights.append(ChartInsight(
                    icon="📊",
                    title="Left-Skewed Distribution",
                    message=f"Mean ({mean_val:.1f}) < Median ({median_val:.1f}), indicating outliers on the low end",
                    severity="info"
                ))
        
        # Outlier detection using IQR
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = values[(values < Q1 - 1.5*IQR) | (values > Q3 + 1.5*IQR)]
        
        if len(outliers) > 0:
            pct_outliers = len(outliers) / len(values) * 100
            insights.append(ChartInsight(
                icon="⚠️",
                title="Outliers Detected",
                message=f"{len(outliers)} values ({pct_outliers:.1f}%) fall outside 1.5×IQR range and may warrant investigation",
                severity="warning" if pct_outliers > 5 else "info",
                metric_name="outlier_count",
                metric_value=len(outliers)
            ))
        
        # Tail risk analysis
        p95 = values.quantile(0.95)
        if p95 > median_val * 2:
            insights.append(ChartInsight(
                icon="📈",
                title="High Tail Risk",
                message=f"95th percentile ({p95:.1f}) is {(p95/median_val):.1f}x the median, indicating significant high-end variability",
                severity="warning"
            ))
        
        return insights
    
    def _analyze_comparison(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract comparison insights"""
        insights = []
        
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        num_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(cat_cols) == 0 or len(num_cols) == 0:
            return insights
        
        cat_col = cat_cols[0]
        num_col = num_cols[0]
        
        # Group statistics
        group_stats = data.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
        
        if len(group_stats) < 2:
            return insights
        
        # Best/worst performer
        best_group = group_stats['mean'].idxmax()
        best_value = group_stats['mean'].max()
        worst_group = group_stats['mean'].idxmin()
        worst_value = group_stats['mean'].min()
        
        if worst_value != 0:
            gap = ((best_value - worst_value) / worst_value * 100)
            
            insights.append(ChartInsight(
                icon="🏆",
                title="Performance Gap",
                message=f"{best_group} ({best_value:.1f}) outperforms {worst_group} ({worst_value:.1f}) by {gap:.1f}%",
                severity="info"
            ))
        
        # Statistical significance test (ANOVA if n > 30)
        if all(group_stats['count'] >= 30):
            groups = [data[data[cat_col] == g][num_col].dropna().values for g in data[cat_col].unique()]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
                
                if p_value < 0.05:
                    insights.append(ChartInsight(
                        icon="✅",
                        title="Statistically Significant",
                        message=f"Differences between groups are statistically significant (F={f_stat:.2f}, p={p_value:.4f})",
                        severity="success"
                    ))
                else:
                    insights.append(ChartInsight(
                        icon="ℹ️",
                        title="No Significant Difference",
                        message=f"Differences between groups are not statistically significant (p={p_value:.4f})",
                        severity="info"
                    ))
        
        return insights
    
    def _analyze_correlation(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract correlation insights"""
        insights = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return insights
        
        corr_matrix = data[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'corr': corr_matrix.iloc[i, j]
                })
        
        # Sort by absolute correlation
        corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['corr']), reverse=True)
        
        # Report top correlations
        for pair in corr_pairs[:3]:
            if abs(pair['corr']) > 0.7:
                direction = "positive" if pair['corr'] > 0 else "negative"
                insights.append(ChartInsight(
                    icon="🔗",
                    title="Strong Correlation",
                    message=f"{pair['var1']} and {pair['var2']} show strong {direction} correlation (r={pair['corr']:.2f})",
                    severity="info",
                    metric_name="correlation",
                    metric_value=pair['corr']
                ))
            elif abs(pair['corr']) > 0.5:
                direction = "positive" if pair['corr'] > 0 else "negative"
                insights.append(ChartInsight(
                    icon="🔗",
                    title="Moderate Correlation",
                    message=f"{pair['var1']} and {pair['var2']} show moderate {direction} correlation (r={pair['corr']:.2f})",
                    severity="info"
                ))
        
        return insights
    
    def _analyze_queue(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract queue evolution insights"""
        insights = []
        
        # Look for queue and wait time columns
        queue_col = None
        wait_col = None
        hour_col = None
        
        for col in data.columns:
            col_lower = col.lower()
            if 'queue' in col_lower:
                queue_col = col
            if 'wait' in col_lower:
                wait_col = col
            if 'hour' in col_lower:
                hour_col = col
        
        if wait_col and queue_col:
            # Find critical threshold (nonlinear escalation point)
            data_sorted = data.sort_values(queue_col).copy()
            data_sorted['wait_delta'] = data_sorted[wait_col].diff()
            data_sorted['queue_delta'] = data_sorted[queue_col].diff()
            data_sorted['acceleration'] = data_sorted['wait_delta'] / data_sorted['queue_delta'].replace(0, np.nan)
            
            critical = data_sorted[data_sorted['acceleration'] > 5]
            if len(critical) > 0:
                critical_queue = critical.iloc[0][queue_col]
                insights.append(ChartInsight(
                    icon="⚠️",
                    title="Critical Threshold Identified",
                    message=f"Wait times escalate rapidly when queue exceeds {critical_queue:.0f} patients (nonlinear regime detected)",
                    severity="warning",
                    metric_name="critical_queue_threshold",
                    metric_value=critical_queue
                ))
        
        if wait_col and hour_col:
            # Peak congestion hour
            peak_idx = data[wait_col].idxmax()
            peak_hour = data.loc[peak_idx, hour_col]
            peak_wait = data[wait_col].max()
            avg_wait = data[wait_col].mean()
            
            insights.append(ChartInsight(
                icon="🕐",
                title="Peak Congestion Hour",
                message=f"Hour {peak_hour:.0f} experiences highest wait times ({peak_wait:.1f} min, {((peak_wait/avg_wait - 1)*100):.0f}% above average)",
                severity="warning" if peak_wait > 60 else "info"
            ))
            
            # Low congestion window
            low_threshold = avg_wait * 0.7
            low_data = data[data[wait_col] < low_threshold]
            if len(low_data) > 0 and hour_col:
                low_hours = low_data[hour_col].values
                insights.append(ChartInsight(
                    icon="✅",
                    title="Optimal Arrival Window",
                    message=f"Hours {int(min(low_hours))}-{int(max(low_hours))} have consistently lower wait times (<{low_threshold:.0f} min)",
                    severity="success"
                ))
        
        return insights
    
    def _analyze_flow(self, data: pd.DataFrame) -> List[ChartInsight]:
        """Extract patient flow insights"""
        insights = []
        
        if 'source' in data.columns and 'target' in data.columns and 'value' in data.columns:
            # Total throughput
            arrivals = data[data['source'].str.contains('Arrival', case=False, na=False)]['value'].sum()
            
            if arrivals > 0:
                # Disposition analysis
                discharged = data[data['target'].str.contains('Discharge', case=False, na=False)]['value'].sum()
                admitted = data[data['target'].str.contains('Admit', case=False, na=False)]['value'].sum()
                
                discharge_rate = discharged / arrivals * 100
                admission_rate = admitted / arrivals * 100
                
                insights.append(ChartInsight(
                    icon="📊",
                    title="Disposition Rates",
                    message=f"{discharge_rate:.1f}% discharged, {admission_rate:.1f}% admitted from {arrivals:.0f} total arrivals",
                    severity="info"
                ))
            
            # Bottleneck identification (if avg_time column exists)
            if 'avg_time' in data.columns:
                bottlenecks = data.nlargest(3, 'avg_time')
                
                for _, row in bottlenecks.iterrows():
                    if row['avg_time'] > 30:  # More than 30 min is concerning
                        insights.append(ChartInsight(
                            icon="🚦",
                            title="Bottleneck Identified",
                            message=f"{row['source']} → {row['target']} averages {row['avg_time']:.1f} min with {row['value']:.0f} patients",
                            severity="warning" if row['avg_time'] > 45 else "info"
                        ))
        
        return insights


# Export classes
__all__ = ['ProductionVisualizer', 'InsightEngine', 'ChartInsight']
