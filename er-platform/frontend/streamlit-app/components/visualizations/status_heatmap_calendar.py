"""
Status Heatmap Calendar Visualization
Weekly congestion pattern analysis

Authors: Neel, Harsh, Tanishk
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from .base_visualizer import ProductionVisualizer, ChartInsight


class StatusHeatmapCalendar(ProductionVisualizer):
    """
    Heatmap calendar showing ED congestion levels
    
    Rows: Hours of day (0-23)
    Columns: Days of week
    Color: Congestion score (0-100)
    """
    
    DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def create(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        score_col: str = 'congestion_score'
    ) -> Tuple[go.Figure, List[ChartInsight]]:
        """
        Create status heatmap calendar
        
        Args:
            data: DataFrame with timestamp and congestion score
            timestamp_col: Column name for timestamp
            score_col: Column name for congestion score
        
        Returns:
            (figure, insights)
        """
        # Prepare data
        df = data.copy()
        
        # Parse timestamp if needed
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        else:
            # Use index or generate
            df[timestamp_col] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
        
        # Determine score column
        if score_col not in df.columns:
            # Try to find a suitable column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                score_col = numeric_cols[0]
            else:
                # Generate demo scores
                df[score_col] = np.random.uniform(20, 80, len(df))
        
        # Add temporal features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_name'] = df[timestamp_col].dt.day_name()
        
        # Pivot to create matrix
        heatmap_data = df.pivot_table(
            index='hour',
            columns='day_name',
            values=score_col,
            aggfunc='mean'
        )
        
        # Reorder columns (Monday-Sunday)
        available_days = [d for d in self.DAY_ORDER if d in heatmap_data.columns]
        heatmap_data = heatmap_data[available_days]
        
        # Fill missing hours
        all_hours = list(range(24))
        heatmap_data = heatmap_data.reindex(all_hours)
        heatmap_data = heatmap_data.fillna(heatmap_data.mean().mean())
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=[f"{h:02d}:00" for h in heatmap_data.index],
            colorscale=[
                [0, 'rgb(46, 160, 44)'],      # Green (low congestion)
                [0.4, 'rgb(255, 235, 59)'],   # Yellow
                [0.6, 'rgb(255, 152, 0)'],    # Orange
                [0.8, 'rgb(255, 87, 34)'],    # Deep Orange
                [1, 'rgb(214, 39, 40)']       # Red (high congestion)
            ],
            colorbar=dict(
                title="Congestion<br>Score",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=20,
                len=0.8,
                thickness=20,
                ticksuffix=""
            ),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Time: %{y}<br>'
                'Congestion Score: %{z:.1f}<br>'
                '<extra></extra>'
            ),
            zmin=0,
            zmax=100
        ))
        
        # Add annotations for extreme values
        for i, hour in enumerate(heatmap_data.index):
            for j, day in enumerate(heatmap_data.columns):
                value = heatmap_data.loc[hour, day]
                if value > 80:  # High congestion
                    fig.add_annotation(
                        x=day,
                        y=f"{hour:02d}:00",
                        text="⚠️",
                        showarrow=False,
                        font=dict(size=12)
                    )
                elif value < 30:  # Low congestion
                    fig.add_annotation(
                        x=day,
                        y=f"{hour:02d}:00",
                        text="✓",
                        showarrow=False,
                        font=dict(size=10, color='green')
                    )
        
        # Layout
        layout = self.create_responsive_layout(
            title="ED Congestion Heatmap",
            subtitle="Weekly Pattern Analysis - Hour by Hour, Day by Day",
            height=650
        )
        
        layout.update({
            'xaxis': {
                'title': 'Day of Week',
                'side': 'bottom',
                'tickangle': 0,
                'tickfont': {'size': 11}
            },
            'yaxis': {
                'title': 'Hour of Day',
                'autorange': 'reversed',  # 00:00 at top
                'tickfont': {'size': 10},
                'dtick': 2
            }
        })
        
        fig.update_layout(layout)
        
        # Extract insights
        insights = self._extract_heatmap_insights(heatmap_data)
        
        return fig, insights
    
    def _extract_heatmap_insights(self, heatmap_data: pd.DataFrame) -> List[ChartInsight]:
        """Extract heatmap insights"""
        insights = []
        
        try:
            # Find highest congestion cell
            max_val = heatmap_data.max().max()
            max_loc = heatmap_data.stack().idxmax()
            
            insights.append(ChartInsight(
                icon="🔴",
                title="Peak Congestion",
                message=f"{max_loc[1]} at {max_loc[0]:02d}:00 has highest congestion (score: {max_val:.1f}/100)",
                severity="critical" if max_val > 85 else "warning",
                metric_name="peak_congestion",
                metric_value=max_val
            ))
            
            # Find lowest congestion window
            min_val = heatmap_data.min().min()
            min_loc = heatmap_data.stack().idxmin()
            
            insights.append(ChartInsight(
                icon="🟢",
                title="Lowest Congestion",
                message=f"{min_loc[1]} at {min_loc[0]:02d}:00 has lowest congestion (score: {min_val:.1f}/100)",
                severity="success",
                metric_name="min_congestion",
                metric_value=min_val
            ))
            
            # Day-level analysis
            day_avg = heatmap_data.mean()
            worst_day = day_avg.idxmax()
            best_day = day_avg.idxmin()
            
            insights.append(ChartInsight(
                icon="📅",
                title="Day Comparison",
                message=f"{worst_day} is busiest (avg: {day_avg[worst_day]:.1f}), {best_day} is quietest (avg: {day_avg[best_day]:.1f})",
                severity="info"
            ))
            
            # Hour-level analysis
            hour_avg = heatmap_data.mean(axis=1)
            peak_hours = hour_avg[hour_avg > hour_avg.quantile(0.75)].index.tolist()
            
            if peak_hours:
                insights.append(ChartInsight(
                    icon="🕐",
                    title="Peak Hours",
                    message=f"{min(peak_hours):02d}:00-{max(peak_hours):02d}:00 consistently show high congestion across all days",
                    severity="warning"
                ))
            
            # Low activity hours
            low_hours = hour_avg[hour_avg < hour_avg.quantile(0.25)].index.tolist()
            if low_hours:
                insights.append(ChartInsight(
                    icon="🌙",
                    title="Low Activity Period",
                    message=f"{min(low_hours):02d}:00-{max(low_hours):02d}:00 are consistently low congestion - ideal for maintenance",
                    severity="success"
                ))
            
            # Weekend vs weekday comparison
            weekend_cols = [c for c in heatmap_data.columns if c in ['Saturday', 'Sunday']]
            weekday_cols = [c for c in heatmap_data.columns if c not in weekend_cols]
            
            if weekend_cols and weekday_cols:
                weekend_avg = heatmap_data[weekend_cols].mean().mean()
                weekday_avg = heatmap_data[weekday_cols].mean().mean()
                
                diff_pct = ((weekday_avg - weekend_avg) / weekend_avg * 100)
                
                if abs(diff_pct) > 10:
                    higher_period = "Weekdays" if diff_pct > 0 else "Weekends"
                    insights.append(ChartInsight(
                        icon="📊",
                        title="Weekly Pattern",
                        message=f"{higher_period} show {abs(diff_pct):.1f}% higher congestion on average",
                        severity="info"
                    ))
            
            # Variability analysis
            overall_std = heatmap_data.std().mean()
            overall_mean = heatmap_data.mean().mean()
            cv = overall_std / overall_mean
            
            if cv > 0.3:
                insights.append(ChartInsight(
                    icon="📈",
                    title="High Variability",
                    message=f"Congestion varies significantly throughout the week (CV={cv:.2f}), requiring adaptive staffing",
                    severity="warning"
                ))
        
        except Exception as e:
            insights.append(ChartInsight(
                icon="ℹ️",
                title="Analysis Note",
                message=f"Some heatmap insights could not be computed",
                severity="info"
            ))
        
        return insights
    
    def create_demo_data(self, weeks: int = 4) -> pd.DataFrame:
        """Generate demo congestion data"""
        np.random.seed(42)
        
        now = datetime.now()
        start = now - timedelta(weeks=weeks)
        
        # Generate hourly data
        times = pd.date_range(start=start, end=now, freq='H')
        
        scores = []
        for t in times:
            hour = t.hour
            dow = t.dayofweek
            
            # Base pattern: busier during day, quieter at night
            hour_effect = 30 + 35 * np.sin(np.pi * (hour - 6) / 12)
            hour_effect = max(10, min(80, hour_effect))
            
            # Weekend effect
            if dow in [5, 6]:
                hour_effect *= 1.1
            
            # Monday effect
            if dow == 0:
                hour_effect *= 1.15
            
            # Random noise
            noise = np.random.normal(0, 8)
            
            score = max(5, min(95, hour_effect + noise))
            scores.append(score)
        
        return pd.DataFrame({
            'timestamp': times,
            'congestion_score': scores
        })
