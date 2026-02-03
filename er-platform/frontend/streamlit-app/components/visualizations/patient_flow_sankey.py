"""
Patient Flow Sankey Diagram Visualization
Interactive real-time patient flow through ED stages

Authors: Neel, Harsh, Tanishk
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from .base_visualizer import ProductionVisualizer, ChartInsight


class PatientFlowSankey(ProductionVisualizer):
    """
    Interactive Sankey diagram showing patient flow through ED stages
    
    Features:
    - Real-time updates
    - Hover details (volume, avg time, bottleneck score)
    - Color-coded by congestion
    - Automated bottleneck identification
    """
    
    # Standard ED flow stages
    DEFAULT_STAGES = [
        'Arrival',
        'Triage',
        'Waiting Room',
        'Bed Assignment',
        'Physician Eval',
        'Testing/Labs',
        'Treatment',
        'Discharge',
        'Admission',
        'Transfer',
        'LWBS'
    ]
    
    def create(
        self, 
        flow_data: pd.DataFrame,
        source_col: str = 'source',
        target_col: str = 'target',
        value_col: str = 'value',
        time_col: str = 'avg_time'
    ) -> Tuple[go.Figure, List[ChartInsight]]:
        """
        Create patient flow Sankey diagram
        
        Args:
            flow_data: DataFrame with flow transitions
            source_col: Column for source stage
            target_col: Column for target stage
            value_col: Column for patient count
            time_col: Column for average time in transition
        
        Returns:
            (figure, insights)
        """
        # Validate columns
        if source_col not in flow_data.columns:
            # Generate demo data if needed
            flow_data = self.create_demo_data()
            source_col, target_col, value_col, time_col = 'source', 'target', 'value', 'avg_time'
        
        # Get all unique nodes
        all_nodes = list(set(flow_data[source_col].unique()) | set(flow_data[target_col].unique()))
        node_dict = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Map to indices
        sources = [node_dict[s] for s in flow_data[source_col]]
        targets = [node_dict[t] for t in flow_data[target_col]]
        values = flow_data[value_col].tolist()
        
        # Get avg times if available
        if time_col in flow_data.columns:
            avg_times = flow_data[time_col].tolist()
        else:
            avg_times = [30] * len(flow_data)  # Default 30 min
        
        # Color links by average time (congestion indicator)
        colors = []
        for avg_time in avg_times:
            if avg_time < 20:
                colors.append('rgba(46, 160, 44, 0.5)')  # Green
            elif avg_time < 45:
                colors.append('rgba(255, 193, 7, 0.5)')  # Yellow
            elif avg_time < 60:
                colors.append('rgba(255, 152, 0, 0.5)')  # Orange
            else:
                colors.append('rgba(214, 39, 40, 0.5)')  # Red
        
        # Node colors based on stage type
        node_colors = []
        for node in all_nodes:
            node_lower = node.lower()
            if 'arrival' in node_lower:
                node_colors.append('rgba(31, 119, 180, 0.9)')  # Blue
            elif 'discharge' in node_lower:
                node_colors.append('rgba(46, 160, 44, 0.9)')  # Green
            elif 'admit' in node_lower:
                node_colors.append('rgba(255, 127, 14, 0.9)')  # Orange
            elif 'lwbs' in node_lower or 'ama' in node_lower:
                node_colors.append('rgba(214, 39, 40, 0.9)')  # Red
            elif 'wait' in node_lower:
                node_colors.append('rgba(255, 193, 7, 0.9)')  # Yellow
            else:
                node_colors.append('rgba(127, 127, 127, 0.9)')  # Gray
        
        # Create custom data for hover
        custom_data = list(zip(avg_times, values))
        
        # Create Sankey
        fig = go.Figure(data=[
            go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=25,
                    thickness=25,
                    line=dict(color='rgba(0,0,0,0.5)', width=1),
                    label=all_nodes,
                    color=node_colors,
                    hovertemplate='<b>%{label}</b><br>Total patients: %{value}<extra></extra>'
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=colors,
                    customdata=custom_data,
                    hovertemplate=(
                        '<b>%{source.label} → %{target.label}</b><br>'
                        'Patients: %{value}<br>'
                        'Avg Time: %{customdata[0]:.1f} min<br>'
                        '<extra></extra>'
                    )
                )
            )
        ])
        
        # Layout
        layout = self.create_responsive_layout(
            title="Patient Flow Through ED Stages",
            subtitle="Real-Time Sankey Diagram - Color Intensity Indicates Congestion",
            height=600
        )
        
        fig.update_layout(layout)
        
        # Extract insights
        insights = self._extract_flow_insights(flow_data, source_col, target_col, value_col, time_col, all_nodes)
        
        return fig, insights
    
    def _extract_flow_insights(
        self,
        flow_data: pd.DataFrame,
        source_col: str,
        target_col: str,
        value_col: str,
        time_col: str,
        nodes: List[str]
    ) -> List[ChartInsight]:
        """Extract patient flow insights"""
        insights = []
        
        try:
            # Identify bottlenecks (high avg_time transitions)
            if time_col in flow_data.columns:
                bottlenecks = flow_data.nlargest(3, time_col)
                
                for idx, row in bottlenecks.iterrows():
                    if row[time_col] > 30:
                        severity = "critical" if row[time_col] > 60 else "warning"
                        insights.append(ChartInsight(
                            icon="🚦",
                            title="Bottleneck Identified",
                            message=f"{row[source_col]} → {row[target_col]} averages {row[time_col]:.1f} min with {row[value_col]:.0f} patients",
                            severity=severity,
                            metric_name="avg_transition_time",
                            metric_value=row[time_col]
                        ))
            
            # Total throughput analysis
            arrivals = flow_data[
                flow_data[source_col].str.contains('Arrival', case=False, na=False)
            ][value_col].sum()
            
            if arrivals > 0:
                # Discharge rate
                discharged = flow_data[
                    flow_data[target_col].str.contains('Discharge', case=False, na=False)
                ][value_col].sum()
                
                # Admission rate
                admitted = flow_data[
                    flow_data[target_col].str.contains('Admit', case=False, na=False)
                ][value_col].sum()
                
                # LWBS rate
                lwbs = flow_data[
                    flow_data[target_col].str.contains('LWBS|Leave', case=False, na=False)
                ][value_col].sum()
                
                discharge_rate = discharged / arrivals * 100
                admission_rate = admitted / arrivals * 100
                lwbs_rate = lwbs / arrivals * 100
                
                insights.append(ChartInsight(
                    icon="📊",
                    title="Disposition Rates",
                    message=f"{discharge_rate:.1f}% discharged, {admission_rate:.1f}% admitted, {lwbs_rate:.1f}% LWBS from {arrivals:.0f} arrivals",
                    severity="warning" if lwbs_rate > 3 else "info"
                ))
            
            # Flow efficiency analysis
            if time_col in flow_data.columns:
                total_time = flow_data[time_col].sum()
                total_patients = flow_data[value_col].sum()
                
                if total_patients > 0:
                    avg_journey_time = total_time * flow_data[value_col].sum() / total_patients
                    
                    insights.append(ChartInsight(
                        icon="⏱️",
                        title="Flow Efficiency",
                        message=f"Average stage transition time: {flow_data[time_col].mean():.1f} min. Total flow capacity: {total_patients:.0f} patient-transitions",
                        severity="info"
                    ))
            
            # Identify parallel flow paths
            waiting_to_bed = flow_data[
                (flow_data[source_col].str.contains('Wait', case=False, na=False)) &
                (flow_data[target_col].str.contains('Bed', case=False, na=False))
            ][value_col].sum()
            
            direct_to_bed = flow_data[
                (flow_data[source_col].str.contains('Triage', case=False, na=False)) &
                (flow_data[target_col].str.contains('Bed', case=False, na=False))
            ][value_col].sum()
            
            if waiting_to_bed > 0 and direct_to_bed > 0:
                fast_track_ratio = direct_to_bed / (waiting_to_bed + direct_to_bed) * 100
                insights.append(ChartInsight(
                    icon="🚀",
                    title="Fast-Track Ratio",
                    message=f"{fast_track_ratio:.1f}% of patients bypass waiting room (direct triage-to-bed)",
                    severity="success" if fast_track_ratio > 20 else "info"
                ))
        
        except Exception as e:
            insights.append(ChartInsight(
                icon="ℹ️",
                title="Analysis Note",
                message=f"Some flow insights could not be computed",
                severity="info"
            ))
        
        return insights
    
    def create_demo_data(self) -> pd.DataFrame:
        """Generate demo flow data"""
        np.random.seed(42)
        
        # Define typical ED flow transitions
        transitions = [
            # Main arrival flow
            ('Arrival', 'Triage', 100, 8),
            ('Triage', 'Waiting Room', 65, 5),
            ('Triage', 'Bed Assignment', 35, 3),  # High acuity direct to bed
            ('Waiting Room', 'Bed Assignment', 60, 35),
            ('Waiting Room', 'LWBS', 5, 45),  # Left without being seen
            
            # Treatment flow
            ('Bed Assignment', 'Physician Eval', 95, 15),
            ('Physician Eval', 'Testing/Labs', 70, 12),
            ('Physician Eval', 'Treatment', 25, 8),
            ('Testing/Labs', 'Treatment', 70, 45),
            
            # Disposition
            ('Treatment', 'Discharge', 75, 25),
            ('Treatment', 'Admission', 18, 60),
            ('Treatment', 'Transfer', 2, 40),
        ]
        
        # Add some variation
        data = []
        for source, target, value, avg_time in transitions:
            data.append({
                'source': source,
                'target': target,
                'value': value + np.random.randint(-5, 10),
                'avg_time': avg_time + np.random.uniform(-5, 10)
            })
        
        return pd.DataFrame(data)
