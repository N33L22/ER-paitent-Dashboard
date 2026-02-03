"""
Granger Causality Analysis
Uncover causal relationships between time-series variables
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from loguru import logger

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.api import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available, Granger causality will use fallback")


@dataclass
class GrangerResult:
    """Result of Granger causality test"""
    cause_var: str
    effect_var: str
    max_lag: int
    optimal_lag: int
    p_values: Dict[int, float]
    f_statistics: Dict[int, float]
    is_significant: bool
    strength: float  # 1 - min(p_value)
    direction: str  # 'causes', 'caused_by', 'bidirectional', 'none'


@dataclass
class CausalNetwork:
    """Network of causal relationships"""
    nodes: List[str]
    edges: List[Dict[str, Any]]
    adjacency: Dict[str, Dict[str, float]]
    root_causes: List[str]
    downstream_effects: Dict[str, List[str]]


class GrangerCausalityAnalyzer:
    """
    Granger Causality Analysis for ED metrics
    
    Tests whether one time series helps predict another,
    suggesting (but not proving) causal relationships.
    """
    
    def __init__(
        self,
        max_lag: int = 12,
        significance_level: float = 0.05
    ):
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.results: List[GrangerResult] = []
        self.causal_network: Optional[CausalNetwork] = None
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        variables: List[str],
        diff_order: int = 1
    ) -> pd.DataFrame:
        """
        Prepare time series data for Granger causality testing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Time series data
        variables : list
            Column names to include
        diff_order : int
            Differencing order to achieve stationarity
        
        Returns
        -------
        pd.DataFrame
            Stationary time series data
        """
        data = df[variables].copy()
        
        # Check stationarity and difference if needed
        if STATSMODELS_AVAILABLE:
            for col in variables:
                adf_result = adfuller(data[col].dropna())
                if adf_result[1] > 0.05:  # Not stationary
                    # Apply differencing
                    for _ in range(diff_order):
                        data[col] = data[col].diff()
        else:
            # Simple differencing without stationarity test
            if diff_order > 0:
                for col in variables:
                    for _ in range(diff_order):
                        data[col] = data[col].diff()
        
        # Drop NaN from differencing
        data = data.dropna()
        
        return data
    
    def test_granger_causality(
        self,
        data: pd.DataFrame,
        cause_var: str,
        effect_var: str
    ) -> GrangerResult:
        """
        Test if cause_var Granger-causes effect_var.
        
        Parameters
        ----------
        data : pd.DataFrame
            Stationary time series data
        cause_var : str
            Potential cause variable
        effect_var : str
            Potential effect variable
        
        Returns
        -------
        GrangerResult
            Test results
        """
        test_data = data[[effect_var, cause_var]].dropna()
        
        if len(test_data) < self.max_lag * 3:
            logger.warning(f"Insufficient data for Granger test: {len(test_data)} rows")
            return GrangerResult(
                cause_var=cause_var,
                effect_var=effect_var,
                max_lag=self.max_lag,
                optimal_lag=0,
                p_values={},
                f_statistics={},
                is_significant=False,
                strength=0.0,
                direction='none'
            )
        
        p_values = {}
        f_statistics = {}
        
        if STATSMODELS_AVAILABLE:
            try:
                gc_results = grangercausalitytests(
                    test_data.values,
                    maxlag=self.max_lag,
                    verbose=False
                )
                
                for lag in range(1, self.max_lag + 1):
                    if lag in gc_results:
                        # Use F-test results
                        test_result = gc_results[lag][0]['ssr_ftest']
                        f_statistics[lag] = test_result[0]
                        p_values[lag] = test_result[1]
                        
            except Exception as e:
                logger.warning(f"Granger test failed: {e}")
        else:
            # Fallback: simple correlation-based pseudo-causality
            for lag in range(1, min(self.max_lag + 1, len(test_data) // 2)):
                shifted = test_data[cause_var].shift(lag)
                valid = ~shifted.isna()
                corr = np.corrcoef(
                    shifted[valid].values,
                    test_data[effect_var][valid].values
                )[0, 1]
                
                # Convert correlation to pseudo p-value
                n = valid.sum()
                t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
                from scipy import stats
                p_values[lag] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                f_statistics[lag] = t_stat**2
        
        # Determine significance
        if p_values:
            min_p = min(p_values.values())
            optimal_lag = min(p_values, key=p_values.get)
            is_significant = min_p < self.significance_level
            strength = 1 - min_p
        else:
            min_p = 1.0
            optimal_lag = 0
            is_significant = False
            strength = 0.0
        
        return GrangerResult(
            cause_var=cause_var,
            effect_var=effect_var,
            max_lag=self.max_lag,
            optimal_lag=optimal_lag,
            p_values=p_values,
            f_statistics=f_statistics,
            is_significant=is_significant,
            strength=strength,
            direction='causes' if is_significant else 'none'
        )
    
    def build_causal_network(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> CausalNetwork:
        """
        Build a network of causal relationships.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data
        variables : list
            Variables to test
        
        Returns
        -------
        CausalNetwork
            Network of causal relationships
        """
        # Prepare data
        prepared = self.prepare_data(data, variables)
        
        # Test all pairs
        edges = []
        adjacency = {v: {} for v in variables}
        self.results = []
        
        for cause in variables:
            for effect in variables:
                if cause != effect:
                    result = self.test_granger_causality(prepared, cause, effect)
                    self.results.append(result)
                    
                    if result.is_significant:
                        edges.append({
                            'source': cause,
                            'target': effect,
                            'strength': result.strength,
                            'optimal_lag': result.optimal_lag,
                            'p_value': min(result.p_values.values()) if result.p_values else 1.0
                        })
                        adjacency[cause][effect] = result.strength
        
        # Find bidirectional relationships
        for r1 in self.results:
            for r2 in self.results:
                if (r1.cause_var == r2.effect_var and 
                    r1.effect_var == r2.cause_var and
                    r1.is_significant and r2.is_significant):
                    r1.direction = 'bidirectional'
                    r2.direction = 'bidirectional'
        
        # Identify root causes (nodes with only outgoing edges)
        incoming = {v: 0 for v in variables}
        outgoing = {v: 0 for v in variables}
        
        for edge in edges:
            outgoing[edge['source']] += 1
            incoming[edge['target']] += 1
        
        root_causes = [v for v in variables if outgoing[v] > 0 and incoming[v] == 0]
        
        # Map downstream effects
        downstream_effects = {}
        for v in variables:
            effects = [e['target'] for e in edges if e['source'] == v]
            if effects:
                downstream_effects[v] = effects
        
        self.causal_network = CausalNetwork(
            nodes=variables,
            edges=edges,
            adjacency=adjacency,
            root_causes=root_causes,
            downstream_effects=downstream_effects
        )
        
        return self.causal_network
    
    def get_impulse_response(
        self,
        data: pd.DataFrame,
        variables: List[str],
        shock_var: str,
        periods: int = 24
    ) -> Dict[str, List[float]]:
        """
        Compute impulse response functions.
        
        Shows how a shock to one variable propagates to others.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data
        variables : list
            Variables in the system
        shock_var : str
            Variable to shock
        periods : int
            Periods to forecast
        
        Returns
        -------
        dict
            Impulse responses for each variable
        """
        prepared = self.prepare_data(data, variables)
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels required for impulse response")
            return {v: [0.0] * periods for v in variables}
        
        try:
            model = VAR(prepared)
            fitted = model.fit(maxlags=min(self.max_lag, len(prepared) // 5))
            
            irf = fitted.irf(periods=periods)
            
            # Get index of shock variable
            shock_idx = variables.index(shock_var)
            
            # Extract responses
            responses = {}
            for i, var in enumerate(variables):
                responses[var] = irf.irfs[:, i, shock_idx].tolist()
            
            return responses
            
        except Exception as e:
            logger.error(f"IRF computation failed: {e}")
            return {v: [0.0] * periods for v in variables}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of causal analysis"""
        if not self.results:
            return {"status": "no_analysis_run"}
        
        significant_results = [r for r in self.results if r.is_significant]
        
        return {
            "total_tests": len(self.results),
            "significant_relationships": len(significant_results),
            "max_lag_tested": self.max_lag,
            "significance_level": self.significance_level,
            "root_causes": self.causal_network.root_causes if self.causal_network else [],
            "strongest_effects": sorted(
                [
                    {
                        "cause": r.cause_var,
                        "effect": r.effect_var,
                        "strength": r.strength,
                        "optimal_lag": r.optimal_lag
                    }
                    for r in significant_results
                ],
                key=lambda x: x["strength"],
                reverse=True
            )[:10]
        }
