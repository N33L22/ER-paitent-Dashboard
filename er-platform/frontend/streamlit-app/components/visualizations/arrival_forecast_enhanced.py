"""
Enhanced Arrival Forecast Visualization with Confidence Bands
Production-grade predictive visualization

Authors: Neel, Harsh, Tanishk
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from .base_visualizer import ProductionVisualizer, ChartInsight


class ArrivalForecastViz(ProductionVisualizer):
    """
    Enhanced arrival forecast with:
    - Historical actuals
    - Point forecast
    - 50%, 80%, 95% prediction intervals
    - Anomaly highlighting
    - Automated commentary
    """
    
    def create(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        actual_col: str = 'actual_arrivals',
        forecast_col: str = 'forecast',
        capacity_threshold: Optional[float] = None
    ) -> Tuple[go.Figure, List[ChartInsight]]:
        """
        Create arrival forecast visualization
        
        Args:
            historical: DataFrame with historical actuals
            forecast: DataFrame with forecast and confidence intervals
            timestamp_col: Column name for timestamp
            actual_col: Column name for actual arrivals
            forecast_col: Column name for forecast values
            capacity_threshold: Optional capacity threshold line
        
        Returns:
            (figure, insights)
        """
        fig = go.Figure()
        
        # Determine column names
        if timestamp_col not in historical.columns:
            timestamp_col = historical.columns[0]
        if actual_col not in historical.columns:
            actual_col = [c for c in historical.columns if 'actual' in c.lower() or 'arrival' in c.lower()]
            actual_col = actual_col[0] if actual_col else historical.select_dtypes(include=[np.number]).columns[0]
        
        # Historical actuals
        fig.add_trace(go.Scatter(
            x=historical[timestamp_col],
            y=historical[actual_col],
            mode='lines+markers',
            name='Historical Actuals',
            line=dict(color=self.COLORS['primary'], width=2),
            marker=dict(size=5, symbol='circle'),
            hovertemplate='<b>%{x}</b><br>Arrivals: %{y:.0f}<extra></extra>'
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast[timestamp_col],
            y=forecast[forecast_col],
            mode='lines',
            name='Forecast',
            line=dict(color=self.COLORS['danger'], width=2.5, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Forecast: %{y:.1f}<extra></extra>'
        ))
        
        # Add confidence intervals (if columns exist)
        ci_configs = [
            ('upper_95', 'lower_95', '95% Confidence', 'rgba(214, 39, 40, 0.1)'),
            ('upper_80', 'lower_80', '80% Confidence', 'rgba(255, 193, 7, 0.15)'),
            ('upper_50', 'lower_50', '50% Confidence (Median)', 'rgba(46, 160, 44, 0.2)')
        ]
        
        for upper, lower, name, fill_color in ci_configs:
            if upper in forecast.columns and lower in forecast.columns:
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=forecast[timestamp_col],
                    y=forecast[upper],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Lower bound with fill
                fig.add_trace(go.Scatter(
                    x=forecast[timestamp_col],
                    y=forecast[lower],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=fill_color,
                    name=name,
                    hovertemplate=f'{name}: %{{y:.1f}}<extra></extra>'
                ))
        
        # Add capacity threshold line
        if capacity_threshold is None:
            capacity_threshold = historical[actual_col].quantile(0.85)
        
        fig.add_hline(
            y=capacity_threshold,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            annotation_text=f"Target Capacity ({capacity_threshold:.0f}/hr)",
            annotation_position="right",
            annotation_font_size=10
        )
        
        # Mark capacity exceedance points
        if 'upper_80' in forecast.columns:
            exceedance_mask = forecast['upper_80'] > capacity_threshold
            if exceedance_mask.any():
                exceedance_times = forecast[exceedance_mask][timestamp_col]
                exceedance_values = forecast[exceedance_mask]['upper_80']
                
                fig.add_trace(go.Scatter(
                    x=exceedance_times,
                    y=exceedance_values,
                    mode='markers',
                    name='Capacity Risk',
                    marker=dict(
                        size=10,
                        color='rgba(214, 39, 40, 0.7)',
                        symbol='triangle-up',
                        line=dict(width=1, color='red')
                    ),
                    hovertemplate='<b>⚠️ Capacity Risk</b><br>%{x}<br>Could reach: %{y:.1f}/hr<extra></extra>'
                ))
        
        # Layout
        layout = self.create_responsive_layout(
            title="Patient Arrival Forecast",
            subtitle="Next 24 Hours with 50%, 80%, 95% Prediction Intervals",
            height=500
        )
        
        layout.update({
            'xaxis': self.get_axis_config('Time', show_grid=True),
            'yaxis': self.get_axis_config('Patient Arrivals per Hour', show_grid=True, zero_line=True),
            'hovermode': 'x unified'
        })
        
        layout['xaxis']['type'] = 'date'
        layout['yaxis']['rangemode'] = 'tozero'
        
        fig.update_layout(layout)
        
        # Extract insights
        insights = self._extract_forecast_insights(historical, forecast, actual_col, forecast_col, capacity_threshold)
        
        return fig, insights
    
    def _extract_forecast_insights(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        actual_col: str,
        forecast_col: str,
        capacity_threshold: float
    ) -> List[ChartInsight]:
        """Extract forecast-specific insights"""
        insights = []
        
        try:
            # Compare forecast to recent average
            recent_avg = historical[actual_col].tail(24).mean()
            forecast_avg = forecast[forecast_col].mean()
            
            if recent_avg > 0:
                pct_change = ((forecast_avg - recent_avg) / recent_avg * 100)
                
                if abs(pct_change) > 10:
                    direction = "higher" if pct_change > 0 else "lower"
                    icon = "📈" if pct_change > 0 else "📉"
                    severity = "warning" if pct_change > 20 else "info"
                    
                    insights.append(ChartInsight(
                        icon=icon,
                        title="Forecast vs. Recent",
                        message=f"Next 24h forecast ({forecast_avg:.1f}/hr) is {abs(pct_change):.1f}% {direction} than recent 24h average ({recent_avg:.1f}/hr)",
                        severity=severity,
                        metric_name="forecast_change_pct",
                        metric_value=pct_change
                    ))
            
            # Identify peak forecast hour
            peak_idx = forecast[forecast_col].idxmax()
            peak_time = pd.to_datetime(forecast.loc[peak_idx, 'timestamp'])
            peak_value = forecast.loc[peak_idx, forecast_col]
            
            if 'upper_95' in forecast.columns:
                peak_upper = forecast.loc[peak_idx, 'upper_95']
                insights.append(ChartInsight(
                    icon="🔺",
                    title="Peak Forecast",
                    message=f"{peak_time.strftime('%I:%M %p')} is expected peak ({peak_value:.1f} patients/hr, could reach {peak_upper:.1f} at 95% CI)",
                    severity="warning" if peak_upper > capacity_threshold else "info"
                ))
            else:
                insights.append(ChartInsight(
                    icon="🔺",
                    title="Peak Forecast",
                    message=f"{peak_time.strftime('%I:%M %p')} is expected peak ({peak_value:.1f} patients/hr)",
                    severity="info"
                ))
            
            # Identify capacity exceedance risk
            if 'upper_80' in forecast.columns:
                exceedances = forecast[forecast['upper_80'] > capacity_threshold]
                
                if len(exceedances) > 0:
                    hours_at_risk = len(exceedances)
                    insights.append(ChartInsight(
                        icon="⚠️",
                        title="Capacity Risk",
                        message=f"{hours_at_risk} hours have 80% probability of exceeding {capacity_threshold:.0f} arrivals/hr threshold",
                        severity="warning" if hours_at_risk > 4 else "info",
                        metric_name="hours_at_capacity_risk",
                        metric_value=hours_at_risk
                    ))
            
            # Uncertainty assessment
            if 'upper_95' in forecast.columns and 'lower_95' in forecast.columns:
                avg_width = (forecast['upper_95'] - forecast['lower_95']).mean()
                avg_forecast = forecast[forecast_col].mean()
                
                if avg_forecast > 0:
                    uncertainty_ratio = avg_width / avg_forecast
                    
                    if uncertainty_ratio > 0.5:
                        insights.append(ChartInsight(
                            icon="❓",
                            title="High Uncertainty",
                            message=f"Prediction intervals are wide ({uncertainty_ratio*100:.0f}% of forecast), suggesting volatile conditions or limited historical patterns",
                            severity="warning"
                        ))
                    else:
                        insights.append(ChartInsight(
                            icon="✅",
                            title="Confident Forecast",
                            message=f"Narrow prediction intervals ({uncertainty_ratio*100:.0f}% of forecast) indicate stable, predictable patterns",
                            severity="success"
                        ))
            
            # Low volume periods
            min_idx = forecast[forecast_col].idxmin()
            min_time = pd.to_datetime(forecast.loc[min_idx, 'timestamp'])
            min_value = forecast.loc[min_idx, forecast_col]
            
            insights.append(ChartInsight(
                icon="🌙",
                title="Lowest Expected Volume",
                message=f"{min_time.strftime('%I:%M %p')} is projected low point ({min_value:.1f} patients/hr) - optimal for shift changes/maintenance",
                severity="info"
            ))
        
        except Exception as e:
            insights.append(ChartInsight(
                icon="ℹ️",
                title="Analysis Note",
                message=f"Some forecast insights could not be computed",
                severity="info"
            ))
        
        return insights
    
    def create_demo_data(
        self,
        hours_historical: int = 48,
        hours_forecast: int = 24
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate demo historical and forecast data"""
        np.random.seed(42)
        
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Historical data
        hist_times = [now - timedelta(hours=i) for i in range(hours_historical, 0, -1)]
        
        # Base arrival pattern (sinusoidal with day/night variation)
        hist_arrivals = []
        for t in hist_times:
            hour = t.hour
            base = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            noise = np.random.normal(0, 3)
            hist_arrivals.append(max(2, base + noise))
        
        historical = pd.DataFrame({
            'timestamp': hist_times,
            'actual_arrivals': hist_arrivals
        })
        
        # Forecast data
        forecast_times = [now + timedelta(hours=i) for i in range(1, hours_forecast + 1)]
        
        forecast_values = []
        for t in forecast_times:
            hour = t.hour
            base = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            forecast_values.append(base)
        
        forecast_arr = np.array(forecast_values)
        
        # Generate confidence intervals
        forecast = pd.DataFrame({
            'timestamp': forecast_times,
            'forecast': forecast_arr,
            'lower_95': forecast_arr * 0.6,
            'upper_95': forecast_arr * 1.5,
            'lower_80': forecast_arr * 0.75,
            'upper_80': forecast_arr * 1.35,
            'lower_50': forecast_arr * 0.85,
            'upper_50': forecast_arr * 1.2
        })
        
        return historical, forecast
