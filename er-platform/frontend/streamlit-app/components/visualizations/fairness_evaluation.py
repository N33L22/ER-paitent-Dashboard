"""
Fairness and Evaluation Visualizations
Enhanced visualizations for model fairness and performance metrics

Authors: Neel, Harsh, Tanishk
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from .base_visualizer import BaseVisualizer


class FairnessEvaluationVisualizer(BaseVisualizer):
    """
    Comprehensive visualizations for model fairness and evaluation.
    
    Includes:
    - Confusion matrix heatmaps
    - ROC curve comparisons
    - Fairness disparity charts
    - Model comparison radars
    - Calibration curves
    - Bias trend analysis
    """
    
    def __init__(self):
        super().__init__()
        self.fairness_colors = {
            'fair': '#38a169',
            'warning': '#ED8936',
            'unfair': '#e53e3e',
            'neutral': '#718096'
        }
        self.model_colors = [
            '#667eea', '#38a169', '#ED8936', '#e53e3e', 
            '#805AD5', '#4299e1', '#ECC94B', '#ed64a6'
        ]
    
    def create_confusion_matrix(
        self,
        matrix: List[List[int]],
        labels: List[str] = None,
        title: str = "Confusion Matrix",
        show_percentages: bool = True
    ) -> go.Figure:
        """
        Create an interactive confusion matrix heatmap.
        
        Parameters
        ----------
        matrix : List[List[int]]
            2D confusion matrix
        labels : List[str], optional
            Class labels
        title : str
            Chart title
        show_percentages : bool
            Whether to show percentages in cells
        
        Returns
        -------
        go.Figure
        """
        matrix = np.array(matrix)
        total = matrix.sum()
        
        if labels is None:
            labels = [f'Class {i}' for i in range(len(matrix))]
        
        # Create annotations
        annotations = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                val = matrix[i][j]
                pct = val / total * 100
                
                if show_percentages:
                    text = f"{val}<br>({pct:.1f}%)"
                else:
                    text = str(val)
                
                annotations.append(
                    dict(
                        x=labels[j], y=labels[i],
                        text=text,
                        showarrow=False,
                        font=dict(size=14, color='white')
                    )
                )
        
        # Custom colorscale for confusion matrix
        # Diagonal (correct) = green, off-diagonal (errors) = red
        colorscale = [[0, '#1a1f2e'], [0.5, '#4a5568'], [1, '#38a169']]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title='Count',
                title_font=dict(color='white'),
                tickfont=dict(color='white')
            ),
            hoverongaps=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title='Predicted Label',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                side='bottom'
            ),
            yaxis=dict(
                title='Actual Label',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                autorange='reversed'
            ),
            annotations=annotations,
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_roc_curve_comparison(
        self,
        roc_data: Dict[str, Dict],
        title: str = "ROC Curve Comparison"
    ) -> go.Figure:
        """
        Create ROC curves for multiple models.
        
        Parameters
        ----------
        roc_data : Dict[str, Dict]
            Dictionary of model_name -> {fpr, tpr, auc, color}
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random (AUC = 0.5)',
            line=dict(dash='dash', color='#718096', width=1),
            hoverinfo='skip'
        ))
        
        # Add ROC curve for each model
        for i, (model, data) in enumerate(roc_data.items()):
            color = data.get('color', self.model_colors[i % len(self.model_colors)])
            auc = data.get('auc', 0)
            
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                mode='lines',
                name=f"{model} (AUC = {auc:.3f})",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{model}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title='False Positive Rate',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                range=[0, 1],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='True Positive Rate',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                range=[0, 1],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            legend=dict(
                font=dict(color='white', size=11),
                x=0.55, y=0.05,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            height=450,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_model_comparison_radar(
        self,
        models_data: pd.DataFrame,
        metrics: List[str] = None,
        title: str = "Model Performance Comparison"
    ) -> go.Figure:
        """
        Create radar chart comparing multiple models.
        
        Parameters
        ----------
        models_data : pd.DataFrame
            DataFrame with 'Model' column and metric columns
        metrics : List[str], optional
            Metrics to include in comparison
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        if metrics is None:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
        
        fig = go.Figure()
        
        for i, row in models_data.iterrows():
            values = [row.get(m, 0) for m in metrics]
            values.append(values[0])  # Close the polygon
            
            color = self.model_colors[i % len(self.model_colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=row['Model'],
                fill='toself',
                opacity=0.6,
                line=dict(color=color, width=2),
                fillcolor=color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in color else color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.6, 1],
                    tickfont=dict(color='white', size=10),
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                angularaxis=dict(
                    tickfont=dict(color='white', size=11),
                    gridcolor='rgba(255,255,255,0.2)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=11),
                orientation='h',
                y=-0.15,
                x=0.5,
                xanchor='center'
            ),
            title=dict(text=title, font=dict(color='white', size=16)),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_fairness_disparity_chart(
        self,
        disparities: List[Dict],
        threshold: float = 0.8,
        title: str = "Fairness Disparity Analysis"
    ) -> go.Figure:
        """
        Create chart showing disparities across protected attributes.
        
        Parameters
        ----------
        disparities : List[Dict]
            List of disparity dictionaries
        threshold : float
            Fairness threshold (4/5 rule default)
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        if not disparities:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No disparities detected - Model meets fairness criteria",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#38a169')
            )
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        df = pd.DataFrame(disparities)
        
        # Create labels for y-axis
        df['label'] = df['attribute'] + ': ' + df['comparison'] + ' vs ' + df['reference']
        
        # Determine colors based on severity
        colors = []
        for _, row in df.iterrows():
            ratio = row.get('ratio', 1)
            if ratio >= threshold and ratio <= 1/threshold:
                colors.append(self.fairness_colors['fair'])
            elif ratio >= 0.7 or ratio <= 1/0.7:
                colors.append(self.fairness_colors['warning'])
            else:
                colors.append(self.fairness_colors['unfair'])
        
        fig = go.Figure()
        
        # Add ratio bars
        fig.add_trace(go.Bar(
            y=df['label'],
            x=df['ratio'],
            orientation='h',
            marker_color=colors,
            text=[f"{r:.2f}" for r in df['ratio']],
            textposition='outside',
            textfont=dict(color='white', size=11),
            hovertemplate="<b>%{y}</b><br>Ratio: %{x:.3f}<extra></extra>"
        ))
        
        # Add threshold lines
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ED8936",
                      annotation_text=f"Lower threshold ({threshold})",
                      annotation_font_color="#ED8936")
        fig.add_vline(x=1/threshold, line_dash="dash", line_color="#ED8936",
                      annotation_text=f"Upper threshold ({1/threshold:.2f})",
                      annotation_font_color="#ED8936")
        fig.add_vline(x=1.0, line_dash="solid", line_color="#38a169",
                      annotation_text="Perfect parity",
                      annotation_font_color="#38a169")
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title='Disparity Ratio',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                tickfont=dict(color='white', size=10),
                automargin=True
            ),
            height=max(300, 50 * len(df)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white'),
            margin=dict(l=200)
        )
        
        return fig
    
    def create_group_performance_chart(
        self,
        group_metrics: Dict[str, Dict],
        metric: str = 'accuracy',
        attribute_name: str = "Group",
        title: str = None
    ) -> go.Figure:
        """
        Create bar chart showing metric across demographic groups.
        
        Parameters
        ----------
        group_metrics : Dict[str, Dict]
            Dictionary of group -> metrics
        metric : str
            Metric to display
        attribute_name : str
            Name of the protected attribute
        title : str, optional
            Chart title
        
        Returns
        -------
        go.Figure
        """
        groups = list(group_metrics.keys())
        values = [group_metrics[g].get(metric, 0) for g in groups]
        counts = [group_metrics[g].get('count', 0) for g in groups]
        
        if title is None:
            title = f"{metric.replace('_', ' ').title()} by {attribute_name}"
        
        # Color based on disparity from max
        max_val = max(values)
        colors = []
        for v in values:
            if max_val > 0:
                ratio = v / max_val
                if ratio >= 0.9:
                    colors.append(self.fairness_colors['fair'])
                elif ratio >= 0.8:
                    colors.append(self.fairness_colors['warning'])
                else:
                    colors.append(self.fairness_colors['unfair'])
            else:
                colors.append(self.fairness_colors['neutral'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=groups,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" if v <= 1 else f"{v:.1f}" for v in values],
            textposition='outside',
            textfont=dict(color='white', size=12),
            hovertemplate="<b>%{x}</b><br>" + 
                         f"{metric}: " + "%{y:.3f}<br>" +
                         "Samples: " + "%{customdata}<extra></extra>",
            customdata=counts
        ))
        
        # Add 4/5 rule threshold line
        if max_val > 0:
            threshold = max_val * 0.8
            fig.add_hline(y=threshold, line_dash="dash", line_color="#ED8936",
                          annotation_text="4/5 Rule Threshold",
                          annotation_font_color="#ED8936")
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=14)),
            xaxis=dict(
                tickfont=dict(color='white'),
                title=attribute_name
            ),
            yaxis=dict(
                title=metric.replace('_', ' ').title(),
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_calibration_curve(
        self,
        bin_centers: np.ndarray,
        actual_freq: np.ndarray,
        model_name: str = "Model",
        title: str = "Calibration Curve"
    ) -> go.Figure:
        """
        Create calibration curve (reliability diagram).
        
        Parameters
        ----------
        bin_centers : array
            Predicted probability bin centers
        actual_freq : array
            Actual frequency for each bin
        model_name : str
            Model name for legend
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='#718096', width=1)
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=bin_centers, y=actual_freq,
            mode='lines+markers',
            name=model_name,
            line=dict(color='#667eea', width=2.5),
            marker=dict(size=8, color='#667eea')
        ))
        
        # Fill area between curves
        fig.add_trace(go.Scatter(
            x=list(bin_centers) + list(bin_centers[::-1]),
            y=list(bin_centers) + list(actual_freq[::-1]),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title='Mean Predicted Probability',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                range=[0, 1],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Fraction of Positives',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                range=[0, 1],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            legend=dict(font=dict(color='white')),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_fairness_trend_chart(
        self,
        trend_data: pd.DataFrame,
        metrics: List[str],
        threshold: float = 0.8,
        title: str = "Fairness Metrics Over Time"
    ) -> go.Figure:
        """
        Create line chart showing fairness trends.
        
        Parameters
        ----------
        trend_data : pd.DataFrame
            DataFrame with time column and metric columns
        metrics : List[str]
            Metrics to display
        threshold : float
            Fairness threshold
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        
        time_col = trend_data.columns[0]
        
        for i, metric in enumerate(metrics):
            if metric in trend_data.columns:
                color = self.model_colors[i % len(self.model_colors)]
                
                fig.add_trace(go.Scatter(
                    x=trend_data[time_col],
                    y=trend_data[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color)
                ))
        
        # Add threshold line
        fig.add_hline(y=threshold, line_dash="dash", line_color="#e53e3e",
                      annotation_text=f"Threshold ({threshold})",
                      annotation_font_color="#e53e3e")
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title='Time Period',
                tickfont=dict(color='white'),
                title_font=dict(color='white')
            ),
            yaxis=dict(
                title='Parity Ratio',
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                range=[0.65, 1.05],
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            legend=dict(
                font=dict(color='white'),
                orientation='h',
                y=-0.15,
                x=0.5,
                xanchor='center'
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig
    
    def create_performance_speed_tradeoff(
        self,
        models_data: pd.DataFrame,
        x_metric: str = 'Inference Time (ms)',
        y_metric: str = 'F1 Score',
        size_metric: str = 'Training Time (s)',
        title: str = "Performance vs Speed Tradeoff"
    ) -> go.Figure:
        """
        Create bubble chart for performance/speed tradeoff.
        
        Parameters
        ----------
        models_data : pd.DataFrame
            DataFrame with model metrics
        x_metric : str
            Metric for x-axis (typically speed)
        y_metric : str
            Metric for y-axis (typically performance)
        size_metric : str
            Metric for bubble size
        title : str
            Chart title
        
        Returns
        -------
        go.Figure
        """
        fig = go.Figure()
        
        sizes = models_data[size_metric] * 3 + 15 if size_metric in models_data.columns else [30] * len(models_data)
        colors = [self.model_colors[i % len(self.model_colors)] for i in range(len(models_data))]
        
        fig.add_trace(go.Scatter(
            x=models_data[x_metric],
            y=models_data[y_metric],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=models_data['Model'],
            textposition='top center',
            textfont=dict(color='white', size=11),
            hovertemplate="<b>%{text}</b><br>" +
                         f"{x_metric}: %{{x:.2f}}<br>" +
                         f"{y_metric}: %{{y:.3f}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color='white', size=16)),
            xaxis=dict(
                title=x_metric,
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title=y_metric,
                tickfont=dict(color='white'),
                title_font=dict(color='white'),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", color='white')
        )
        
        return fig


# Singleton instance
fairness_visualizer = FairnessEvaluationVisualizer()
