"""
Monte Carlo Simulation Module
Quantifies uncertainty through repeated stochastic simulations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from .discrete_event_sim import EDSimulation, SimulationConfig, SimulationResults


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    num_runs: int = 100
    base_config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Parameter variation (for sensitivity analysis)
    vary_arrival_rate: bool = False
    arrival_rate_std: float = 0.1  # 10% std
    
    vary_service_times: bool = False
    service_time_std: float = 0.15  # 15% std
    
    # Random seeds
    seed_start: int = 42
    
    # Parallel execution
    max_workers: int = 4


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation"""
    config: MonteCarloConfig
    num_runs: int
    
    # LOS statistics
    mean_los_distribution: List[float] = field(default_factory=list)
    median_los_distribution: List[float] = field(default_factory=list)
    p90_los_distribution: List[float] = field(default_factory=list)
    
    # Wait time distributions
    mean_wait_distribution: List[float] = field(default_factory=list)
    
    # LWBS distributions
    lwbs_count_distribution: List[int] = field(default_factory=list)
    lwbs_rate_distribution: List[float] = field(default_factory=list)
    
    # Patient count distribution
    total_patients_distribution: List[int] = field(default_factory=list)
    
    # Summary statistics
    los_mean: float = 0
    los_std: float = 0
    los_p5: float = 0
    los_p50: float = 0
    los_p95: float = 0
    
    wait_mean: float = 0
    wait_std: float = 0
    wait_p5: float = 0
    wait_p95: float = 0
    
    lwbs_rate_mean: float = 0
    lwbs_rate_std: float = 0
    
    # Confidence intervals
    los_ci_95: tuple = (0, 0)
    wait_ci_95: tuple = (0, 0)
    
    # Individual run results
    all_results: List[SimulationResults] = field(default_factory=list)
    
    def compute_summary_statistics(self):
        """Compute summary statistics from distributions"""
        if self.mean_los_distribution:
            self.los_mean = float(np.mean(self.mean_los_distribution))
            self.los_std = float(np.std(self.mean_los_distribution))
            self.los_p5 = float(np.percentile(self.mean_los_distribution, 5))
            self.los_p50 = float(np.percentile(self.mean_los_distribution, 50))
            self.los_p95 = float(np.percentile(self.mean_los_distribution, 95))
            
            # 95% confidence interval
            self.los_ci_95 = (
                float(np.percentile(self.mean_los_distribution, 2.5)),
                float(np.percentile(self.mean_los_distribution, 97.5))
            )
        
        if self.mean_wait_distribution:
            self.wait_mean = float(np.mean(self.mean_wait_distribution))
            self.wait_std = float(np.std(self.mean_wait_distribution))
            self.wait_p5 = float(np.percentile(self.mean_wait_distribution, 5))
            self.wait_p95 = float(np.percentile(self.mean_wait_distribution, 95))
            
            self.wait_ci_95 = (
                float(np.percentile(self.mean_wait_distribution, 2.5)),
                float(np.percentile(self.mean_wait_distribution, 97.5))
            )
        
        if self.lwbs_rate_distribution:
            self.lwbs_rate_mean = float(np.mean(self.lwbs_rate_distribution))
            self.lwbs_rate_std = float(np.std(self.lwbs_rate_distribution))


class MonteCarloSimulator:
    """
    Monte Carlo simulation for uncertainty quantification
    
    Runs multiple simulations with:
    - Different random seeds
    - Optional parameter variation
    - Statistical analysis of results
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            config: Monte Carlo configuration
        """
        self.config = config or MonteCarloConfig()
        self.results: Optional[MonteCarloResults] = None
    
    def _run_single_simulation(
        self,
        run_index: int
    ) -> SimulationResults:
        """
        Run a single simulation with modified seed
        
        Args:
            run_index: Index of this run
            
        Returns:
            SimulationResults
        """
        # Create config with unique seed
        config = SimulationConfig(
            num_triage_nurses=self.config.base_config.num_triage_nurses,
            num_physicians=self.config.base_config.num_physicians,
            num_nurses=self.config.base_config.num_nurses,
            num_beds=self.config.base_config.num_beds,
            num_lab_techs=self.config.base_config.num_lab_techs,
            num_imaging_techs=self.config.base_config.num_imaging_techs,
            simulation_hours=self.config.base_config.simulation_hours,
            arrival_rate_per_hour=self.config.base_config.arrival_rate_per_hour,
            arrival_multiplier=self.config.base_config.arrival_multiplier,
            seed=self.config.seed_start + run_index
        )
        
        # Vary parameters if configured
        if self.config.vary_arrival_rate:
            variation = 1 + np.random.normal(0, self.config.arrival_rate_std)
            config.arrival_multiplier *= max(0.5, variation)
        
        # Run simulation
        sim = EDSimulation(config)
        return sim.run()
    
    def run(self, show_progress: bool = True) -> MonteCarloResults:
        """
        Run Monte Carlo simulation
        
        Args:
            show_progress: Whether to log progress
            
        Returns:
            MonteCarloResults
        """
        logger.info(f"Starting Monte Carlo simulation with {self.config.num_runs} runs")
        
        # Initialize results
        self.results = MonteCarloResults(
            config=self.config,
            num_runs=self.config.num_runs
        )
        
        # Run simulations (sequential for determinism)
        for i in range(self.config.num_runs):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{self.config.num_runs} runs")
            
            result = self._run_single_simulation(i)
            
            # Collect distributions
            self.results.mean_los_distribution.append(result.mean_los_minutes)
            self.results.median_los_distribution.append(result.median_los_minutes)
            self.results.p90_los_distribution.append(result.p90_los_minutes)
            self.results.mean_wait_distribution.append(result.mean_wait_to_bed)
            self.results.lwbs_count_distribution.append(result.lwbs_count)
            
            lwbs_rate = result.lwbs_count / max(1, result.total_patients) * 100
            self.results.lwbs_rate_distribution.append(lwbs_rate)
            self.results.total_patients_distribution.append(result.total_patients)
            
            self.results.all_results.append(result)
        
        # Compute summary statistics
        self.results.compute_summary_statistics()
        
        logger.info(f"Monte Carlo complete. Mean LOS: {self.results.los_mean:.1f} ± {self.results.los_std:.1f} min")
        
        return self.results
    
    def get_risk_analysis(self) -> Dict[str, Any]:
        """
        Perform risk analysis on results
        
        Returns:
            Risk metrics and probabilities
        """
        if self.results is None:
            raise ValueError("No results available. Run simulation first.")
        
        los_dist = np.array(self.results.mean_los_distribution)
        wait_dist = np.array(self.results.mean_wait_distribution)
        lwbs_dist = np.array(self.results.lwbs_rate_distribution)
        
        return {
            'los_analysis': {
                'mean': float(np.mean(los_dist)),
                'std': float(np.std(los_dist)),
                'min': float(np.min(los_dist)),
                'max': float(np.max(los_dist)),
                'p5': float(np.percentile(los_dist, 5)),
                'p25': float(np.percentile(los_dist, 25)),
                'p50': float(np.percentile(los_dist, 50)),
                'p75': float(np.percentile(los_dist, 75)),
                'p95': float(np.percentile(los_dist, 95)),
                'prob_exceeds_120min': float(np.mean(los_dist > 120)),
                'prob_exceeds_180min': float(np.mean(los_dist > 180)),
                'prob_exceeds_240min': float(np.mean(los_dist > 240))
            },
            'wait_analysis': {
                'mean': float(np.mean(wait_dist)),
                'std': float(np.std(wait_dist)),
                'p5': float(np.percentile(wait_dist, 5)),
                'p50': float(np.percentile(wait_dist, 50)),
                'p95': float(np.percentile(wait_dist, 95)),
                'prob_exceeds_30min': float(np.mean(wait_dist > 30)),
                'prob_exceeds_60min': float(np.mean(wait_dist > 60)),
                'prob_exceeds_90min': float(np.mean(wait_dist > 90))
            },
            'lwbs_analysis': {
                'mean_rate': float(np.mean(lwbs_dist)),
                'std_rate': float(np.std(lwbs_dist)),
                'prob_exceeds_2pct': float(np.mean(lwbs_dist > 2)),
                'prob_exceeds_5pct': float(np.mean(lwbs_dist > 5))
            },
            'value_at_risk': {
                'los_var_95': float(np.percentile(los_dist, 95)),
                'wait_var_95': float(np.percentile(wait_dist, 95)),
                'los_cvar_95': float(np.mean(los_dist[los_dist >= np.percentile(los_dist, 95)]))
            }
        }
    
    def get_distribution_data(self) -> Dict[str, Any]:
        """
        Get distribution data for visualization
        
        Returns:
            Histogram-ready data
        """
        if self.results is None:
            raise ValueError("No results available")
        
        return {
            'los': {
                'values': self.results.mean_los_distribution,
                'bins': np.histogram(self.results.mean_los_distribution, bins=20)[1].tolist(),
                'counts': np.histogram(self.results.mean_los_distribution, bins=20)[0].tolist()
            },
            'wait': {
                'values': self.results.mean_wait_distribution,
                'bins': np.histogram(self.results.mean_wait_distribution, bins=20)[1].tolist(),
                'counts': np.histogram(self.results.mean_wait_distribution, bins=20)[0].tolist()
            },
            'lwbs_rate': {
                'values': self.results.lwbs_rate_distribution,
                'bins': np.histogram(self.results.lwbs_rate_distribution, bins=20)[1].tolist(),
                'counts': np.histogram(self.results.lwbs_rate_distribution, bins=20)[0].tolist()
            }
        }
    
    def compare_scenarios(
        self,
        scenario_configs: Dict[str, SimulationConfig],
        runs_per_scenario: int = 50
    ) -> Dict[str, MonteCarloResults]:
        """
        Compare multiple scenarios with Monte Carlo
        
        Args:
            scenario_configs: Dictionary of scenario name to config
            runs_per_scenario: Number of runs per scenario
            
        Returns:
            Dictionary of scenario results
        """
        results = {}
        
        for name, config in scenario_configs.items():
            logger.info(f"Running Monte Carlo for scenario: {name}")
            
            mc_config = MonteCarloConfig(
                num_runs=runs_per_scenario,
                base_config=config,
                seed_start=self.config.seed_start
            )
            
            simulator = MonteCarloSimulator(mc_config)
            results[name] = simulator.run(show_progress=False)
        
        return results
