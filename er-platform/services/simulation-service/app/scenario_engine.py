"""
Scenario Engine for What-If Analysis
Builds and executes multiple simulation scenarios
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from loguru import logger

from .discrete_event_sim import EDSimulation, SimulationConfig, SimulationResults


@dataclass
class ScenarioDefinition:
    """Definition of a simulation scenario"""
    name: str
    description: str = ""
    
    # Resource modifications (multipliers)
    physicians_multiplier: float = 1.0
    nurses_multiplier: float = 1.0
    beds_multiplier: float = 1.0
    triage_nurses_multiplier: float = 1.0
    
    # Arrival modifications
    arrival_multiplier: float = 1.0
    
    # Duration
    simulation_hours: int = 168


@dataclass
class ScenarioResult:
    """Result of a scenario simulation"""
    scenario: ScenarioDefinition
    results: SimulationResults
    
    # Comparison to baseline
    los_change_percent: Optional[float] = None
    wait_change_percent: Optional[float] = None
    lwbs_change_percent: Optional[float] = None


class ScenarioEngine:
    """
    Engine for running what-if scenarios
    
    Supports:
    - Single scenario execution
    - Scenario comparison
    - Scenario matrix (grid search)
    """
    
    # Predefined scenarios
    PRESET_SCENARIOS = {
        'baseline': ScenarioDefinition(
            name='Baseline',
            description='Current state configuration'
        ),
        'understaffed': ScenarioDefinition(
            name='Understaffed',
            description='20% reduction in physicians',
            physicians_multiplier=0.8
        ),
        'overstaffed': ScenarioDefinition(
            name='Overstaffed',
            description='20% increase in physicians',
            physicians_multiplier=1.2
        ),
        'surge': ScenarioDefinition(
            name='Demand Surge',
            description='30% increase in arrivals',
            arrival_multiplier=1.3
        ),
        'expansion': ScenarioDefinition(
            name='Bed Expansion',
            description='25% more beds',
            beds_multiplier=1.25
        ),
        'crisis': ScenarioDefinition(
            name='Crisis Mode',
            description='50% surge with 20% staff reduction',
            arrival_multiplier=1.5,
            physicians_multiplier=0.8,
            nurses_multiplier=0.8
        ),
        'optimal': ScenarioDefinition(
            name='Optimized',
            description='Optimized staffing for typical demand',
            physicians_multiplier=1.1,
            beds_multiplier=1.1
        )
    }
    
    def __init__(self, base_config: Optional[SimulationConfig] = None):
        """
        Initialize scenario engine
        
        Args:
            base_config: Base configuration for all scenarios
        """
        self.base_config = base_config or SimulationConfig()
        self.baseline_results: Optional[SimulationResults] = None
        self.scenario_results: Dict[str, ScenarioResult] = {}
    
    def apply_scenario(
        self,
        scenario: ScenarioDefinition
    ) -> SimulationConfig:
        """
        Apply scenario modifications to base config
        
        Args:
            scenario: Scenario definition
            
        Returns:
            Modified SimulationConfig
        """
        config = SimulationConfig(
            num_triage_nurses=max(1, int(self.base_config.num_triage_nurses * scenario.triage_nurses_multiplier)),
            num_physicians=max(1, int(self.base_config.num_physicians * scenario.physicians_multiplier)),
            num_nurses=max(1, int(self.base_config.num_nurses * scenario.nurses_multiplier)),
            num_beds=max(1, int(self.base_config.num_beds * scenario.beds_multiplier)),
            num_lab_techs=self.base_config.num_lab_techs,
            num_imaging_techs=self.base_config.num_imaging_techs,
            simulation_hours=scenario.simulation_hours,
            arrival_rate_per_hour=self.base_config.arrival_rate_per_hour,
            arrival_multiplier=scenario.arrival_multiplier,
            seed=self.base_config.seed
        )
        
        return config
    
    def run_scenario(
        self,
        scenario: ScenarioDefinition,
        compare_to_baseline: bool = True
    ) -> ScenarioResult:
        """
        Run a single scenario
        
        Args:
            scenario: Scenario to run
            compare_to_baseline: Whether to compare results to baseline
            
        Returns:
            ScenarioResult
        """
        logger.info(f"Running scenario: {scenario.name}")
        
        # Apply scenario to get config
        config = self.apply_scenario(scenario)
        
        # Run simulation
        sim = EDSimulation(config)
        results = sim.run()
        
        # Create result
        scenario_result = ScenarioResult(
            scenario=scenario,
            results=results
        )
        
        # Compare to baseline if available
        if compare_to_baseline and self.baseline_results:
            baseline = self.baseline_results
            
            if baseline.mean_los_minutes > 0:
                scenario_result.los_change_percent = (
                    (results.mean_los_minutes - baseline.mean_los_minutes) / 
                    baseline.mean_los_minutes * 100
                )
            
            if baseline.mean_wait_to_bed > 0:
                scenario_result.wait_change_percent = (
                    (results.mean_wait_to_bed - baseline.mean_wait_to_bed) / 
                    baseline.mean_wait_to_bed * 100
                )
            
            if baseline.lwbs_count > 0:
                scenario_result.lwbs_change_percent = (
                    (results.lwbs_count - baseline.lwbs_count) / 
                    baseline.lwbs_count * 100
                )
        
        # Store result
        self.scenario_results[scenario.name] = scenario_result
        
        return scenario_result
    
    def run_baseline(self) -> SimulationResults:
        """Run baseline scenario"""
        baseline_scenario = self.PRESET_SCENARIOS['baseline']
        result = self.run_scenario(baseline_scenario, compare_to_baseline=False)
        self.baseline_results = result.results
        return result.results
    
    def run_preset(self, preset_name: str) -> ScenarioResult:
        """
        Run a predefined scenario
        
        Args:
            preset_name: Name of preset scenario
            
        Returns:
            ScenarioResult
        """
        if preset_name not in self.PRESET_SCENARIOS:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        # Ensure baseline exists
        if self.baseline_results is None:
            self.run_baseline()
        
        scenario = self.PRESET_SCENARIOS[preset_name]
        return self.run_scenario(scenario)
    
    def run_all_presets(self) -> Dict[str, ScenarioResult]:
        """Run all predefined scenarios"""
        # Run baseline first
        self.run_baseline()
        
        # Run all others
        for name in self.PRESET_SCENARIOS:
            if name != 'baseline':
                self.run_preset(name)
        
        return self.scenario_results
    
    def run_scenario_matrix(
        self,
        staffing_multipliers: List[float],
        arrival_multipliers: List[float],
        simulation_hours: int = 168
    ) -> Dict[Tuple[float, float], ScenarioResult]:
        """
        Run a matrix of scenarios
        
        Args:
            staffing_multipliers: List of staffing multipliers to test
            arrival_multipliers: List of arrival multipliers to test
            simulation_hours: Duration for each simulation
            
        Returns:
            Dictionary mapping (staffing, arrivals) to results
        """
        logger.info(f"Running scenario matrix: {len(staffing_multipliers)}x{len(arrival_multipliers)}")
        
        # Ensure baseline
        if self.baseline_results is None:
            self.run_baseline()
        
        matrix_results = {}
        
        for staff_mult in staffing_multipliers:
            for arrival_mult in arrival_multipliers:
                scenario = ScenarioDefinition(
                    name=f"Staff_{staff_mult:.2f}_Arrivals_{arrival_mult:.2f}",
                    description=f"Staffing: {staff_mult:.0%}, Arrivals: {arrival_mult:.0%}",
                    physicians_multiplier=staff_mult,
                    nurses_multiplier=staff_mult,
                    arrival_multiplier=arrival_mult,
                    simulation_hours=simulation_hours
                )
                
                result = self.run_scenario(scenario)
                matrix_results[(staff_mult, arrival_mult)] = result
        
        return matrix_results
    
    def get_comparison_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary comparison of all scenarios
        
        Returns:
            List of scenario summaries
        """
        summaries = []
        
        for name, result in self.scenario_results.items():
            summary = {
                'name': name,
                'description': result.scenario.description,
                'total_patients': result.results.total_patients,
                'mean_los_minutes': round(result.results.mean_los_minutes, 1),
                'median_los_minutes': round(result.results.median_los_minutes, 1),
                'p90_los_minutes': round(result.results.p90_los_minutes, 1),
                'mean_wait_to_bed': round(result.results.mean_wait_to_bed, 1),
                'lwbs_count': result.results.lwbs_count,
                'lwbs_rate': round(result.results.lwbs_count / max(1, result.results.total_patients) * 100, 2),
                'los_change_percent': round(result.los_change_percent, 1) if result.los_change_percent else None,
                'wait_change_percent': round(result.wait_change_percent, 1) if result.wait_change_percent else None,
                'configuration': {
                    'physicians': int(self.base_config.num_physicians * result.scenario.physicians_multiplier),
                    'nurses': int(self.base_config.num_nurses * result.scenario.nurses_multiplier),
                    'beds': int(self.base_config.num_beds * result.scenario.beds_multiplier),
                    'arrival_rate': round(self.base_config.arrival_rate_per_hour * result.scenario.arrival_multiplier, 1)
                }
            }
            summaries.append(summary)
        
        return summaries
    
    def get_matrix_heatmap_data(
        self,
        matrix_results: Dict[Tuple[float, float], ScenarioResult],
        metric: str = 'mean_los_minutes'
    ) -> Dict[str, Any]:
        """
        Get heatmap data for scenario matrix
        
        Args:
            matrix_results: Results from run_scenario_matrix
            metric: Metric to visualize
            
        Returns:
            Heatmap-ready data
        """
        staffing_levels = sorted(set(k[0] for k in matrix_results.keys()))
        arrival_levels = sorted(set(k[1] for k in matrix_results.keys()))
        
        values = []
        for staff in staffing_levels:
            row = []
            for arrival in arrival_levels:
                result = matrix_results.get((staff, arrival))
                if result:
                    val = getattr(result.results, metric, 0)
                    row.append(float(val))
                else:
                    row.append(None)
            values.append(row)
        
        return {
            'x_labels': [f"{a:.0%}" for a in arrival_levels],
            'y_labels': [f"{s:.0%}" for s in staffing_levels],
            'x_title': 'Arrival Rate',
            'y_title': 'Staffing Level',
            'values': values,
            'metric': metric
        }
