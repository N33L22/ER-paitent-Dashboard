"""
Survival Analysis Module
Kaplan-Meier and Cox Proportional Hazards for LOS analysis
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    logger.warning("Lifelines not available")


class SurvivalAnalyzer:
    """
    Survival Analysis for ED Length of Stay
    
    Treats LOS as a time-to-event outcome:
    - Event = discharge/admission (patient leaves ED)
    - Censoring = patient still in ED
    
    Methods:
    - Kaplan-Meier: Non-parametric survival curves
    - Cox PH: Semi-parametric regression with hazard ratios
    """
    
    def __init__(self):
        """Initialize survival analyzer"""
        if not LIFELINES_AVAILABLE:
            raise ImportError("Lifelines is required for survival analysis")
        
        self.km_fitters: Dict[str, KaplanMeierFitter] = {}
        self.cox_model: Optional[CoxPHFitter] = None
        self.is_fitted: bool = False
    
    def fit_kaplan_meier(
        self,
        df: pd.DataFrame,
        duration_col: str = 'total_los_minutes',
        event_col: str = 'event_observed',
        stratify_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit Kaplan-Meier survival curves
        
        Args:
            df: DataFrame with duration and event columns
            duration_col: Name of duration column
            event_col: Name of event indicator column
            stratify_by: Optional column to stratify curves
            
        Returns:
            Dictionary with survival curve data
        """
        logger.info("Fitting Kaplan-Meier survival curves...")
        
        # Ensure event column exists
        if event_col not in df.columns:
            df[event_col] = 1  # Assume all events observed
        
        results = {}
        
        if stratify_by and stratify_by in df.columns:
            # Stratified analysis
            groups = df[stratify_by].unique()
            
            for group in groups:
                mask = df[stratify_by] == group
                group_df = df[mask]
                
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=group_df[duration_col],
                    event_observed=group_df[event_col],
                    label=str(group)
                )
                
                self.km_fitters[str(group)] = kmf
                
                # Extract survival curve data
                timeline = kmf.survival_function_.index.tolist()
                survival_prob = kmf.survival_function_.iloc[:, 0].tolist()
                ci_lower = kmf.confidence_interval_.iloc[:, 0].tolist()
                ci_upper = kmf.confidence_interval_.iloc[:, 1].tolist()
                
                results[str(group)] = {
                    'timeline': timeline,
                    'survival_probability': survival_prob,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'median_survival': float(kmf.median_survival_time_) if kmf.median_survival_time_ < np.inf else None,
                    'n_observations': len(group_df),
                    'n_events': int(group_df[event_col].sum())
                }
            
            # Log-rank test for difference between groups
            if len(groups) == 2:
                g1, g2 = groups[0], groups[1]
                test_result = logrank_test(
                    df[df[stratify_by] == g1][duration_col],
                    df[df[stratify_by] == g2][duration_col],
                    df[df[stratify_by] == g1][event_col],
                    df[df[stratify_by] == g2][event_col]
                )
                results['log_rank_test'] = {
                    'test_statistic': float(test_result.test_statistic),
                    'p_value': float(test_result.p_value),
                    'significant': test_result.p_value < 0.05
                }
            elif len(groups) > 2:
                test_result = multivariate_logrank_test(
                    df[duration_col],
                    df[stratify_by],
                    df[event_col]
                )
                results['log_rank_test'] = {
                    'test_statistic': float(test_result.test_statistic),
                    'p_value': float(test_result.p_value),
                    'significant': test_result.p_value < 0.05
                }
        else:
            # Overall survival curve
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df[duration_col],
                event_observed=df[event_col],
                label='Overall'
            )
            
            self.km_fitters['overall'] = kmf
            
            results['overall'] = {
                'timeline': kmf.survival_function_.index.tolist(),
                'survival_probability': kmf.survival_function_.iloc[:, 0].tolist(),
                'ci_lower': kmf.confidence_interval_.iloc[:, 0].tolist(),
                'ci_upper': kmf.confidence_interval_.iloc[:, 1].tolist(),
                'median_survival': float(kmf.median_survival_time_) if kmf.median_survival_time_ < np.inf else None,
                'n_observations': len(df),
                'n_events': int(df[event_col].sum())
            }
        
        self.is_fitted = True
        logger.info(f"Fitted {len(self.km_fitters)} Kaplan-Meier curves")
        
        return results
    
    def fit_cox_ph(
        self,
        df: pd.DataFrame,
        duration_col: str = 'total_los_minutes',
        event_col: str = 'event_observed',
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fit Cox Proportional Hazards model
        
        Args:
            df: DataFrame with covariates
            duration_col: Name of duration column
            event_col: Name of event indicator column
            covariates: List of covariate columns to include
            
        Returns:
            Model summary and hazard ratios
        """
        logger.info("Fitting Cox Proportional Hazards model...")
        
        # Ensure event column
        if event_col not in df.columns:
            df[event_col] = 1
        
        # Select covariates
        if covariates is None:
            # Default covariates
            covariates = [
                'acuity', 'age', 'is_weekend', 'is_peak_hours'
            ]
        
        # Filter to available columns
        available_covariates = [c for c in covariates if c in df.columns]
        
        # Prepare data
        cox_df = df[[duration_col, event_col] + available_covariates].copy()
        cox_df = cox_df.dropna()
        
        # Fit model
        self.cox_model = CoxPHFitter()
        self.cox_model.fit(
            cox_df,
            duration_col=duration_col,
            event_col=event_col
        )
        
        # Extract results
        summary = self.cox_model.summary
        
        hazard_ratios = {}
        for covariate in available_covariates:
            if covariate in summary.index:
                row = summary.loc[covariate]
                hazard_ratios[covariate] = {
                    'hazard_ratio': float(np.exp(row['coef'])),
                    'coefficient': float(row['coef']),
                    'se': float(row['se(coef)']),
                    'z': float(row['z']),
                    'p_value': float(row['p']),
                    'ci_lower': float(np.exp(row['coef lower 95%'])),
                    'ci_upper': float(np.exp(row['coef upper 95%'])),
                    'significant': row['p'] < 0.05
                }
        
        results = {
            'hazard_ratios': hazard_ratios,
            'concordance_index': float(self.cox_model.concordance_index_),
            'log_likelihood': float(self.cox_model.log_likelihood_),
            'n_observations': len(cox_df),
            'n_events': int(cox_df[event_col].sum()),
            'covariates_used': available_covariates
        }
        
        logger.info(f"Cox model fitted. C-index: {results['concordance_index']:.3f}")
        
        return results
    
    def predict_survival(
        self,
        times: List[float],
        patient_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Predict survival probability at specific times
        
        Args:
            times: List of time points (minutes)
            patient_data: Optional patient covariates for Cox prediction
            
        Returns:
            Survival probabilities at each time point
        """
        results = {}
        
        # Kaplan-Meier predictions (overall or stratified)
        for group_name, kmf in self.km_fitters.items():
            probs = []
            for t in times:
                if t <= kmf.survival_function_.index.max():
                    prob = kmf.predict(t)
                else:
                    prob = kmf.survival_function_.iloc[-1, 0]
                probs.append(float(prob))
            
            results[f'km_{group_name}'] = {
                'times': times,
                'survival_probabilities': probs
            }
        
        # Cox predictions if model fitted and patient data provided
        if self.cox_model is not None and patient_data is not None:
            patient_df = pd.DataFrame([patient_data])
            
            # Ensure all covariates present
            for cov in self.cox_model.params_.index:
                if cov not in patient_df.columns:
                    patient_df[cov] = 0
            
            survival_func = self.cox_model.predict_survival_function(patient_df)
            
            probs = []
            for t in times:
                if t <= survival_func.index.max():
                    prob = float(survival_func.loc[t].values[0]) if t in survival_func.index else float(survival_func.iloc[(survival_func.index <= t).sum() - 1].values[0])
                else:
                    prob = float(survival_func.iloc[-1].values[0])
                probs.append(prob)
            
            results['cox_prediction'] = {
                'times': times,
                'survival_probabilities': probs,
                'patient_risk_score': float(self.cox_model.predict_partial_hazard(patient_df).values[0])
            }
        
        return results
    
    def get_percentile_times(
        self,
        percentiles: List[float] = [25, 50, 75, 90]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get LOS times at various percentiles from survival curves
        
        Args:
            percentiles: List of percentiles (0-100)
            
        Returns:
            Percentile times for each fitted group
        """
        results = {}
        
        for group_name, kmf in self.km_fitters.items():
            group_results = {}
            
            for pct in percentiles:
                target_prob = 1 - (pct / 100)  # Convert to survival probability
                
                # Find time where survival crosses target
                sf = kmf.survival_function_.iloc[:, 0]
                times_below = sf[sf <= target_prob].index
                
                if len(times_below) > 0:
                    group_results[f'p{pct}'] = float(times_below.min())
                else:
                    group_results[f'p{pct}'] = None
            
            results[group_name] = group_results
        
        return results
    
    def get_visualization_data(
        self,
        max_time: float = 480  # 8 hours
    ) -> Dict[str, Any]:
        """
        Get data formatted for visualization
        
        Args:
            max_time: Maximum time to include (minutes)
            
        Returns:
            Visualization-ready data
        """
        viz_data = {
            'curves': [],
            'time_points': list(range(0, int(max_time) + 1, 15))  # 15-min intervals
        }
        
        for group_name, kmf in self.km_fitters.items():
            curve_data = {
                'name': group_name,
                'time': [],
                'survival': [],
                'ci_lower': [],
                'ci_upper': []
            }
            
            for t in viz_data['time_points']:
                if t <= kmf.survival_function_.index.max():
                    # Interpolate to exact time
                    sf = kmf.survival_function_
                    ci = kmf.confidence_interval_
                    
                    idx = (sf.index <= t).sum() - 1
                    idx = max(0, idx)
                    
                    curve_data['time'].append(t)
                    curve_data['survival'].append(float(sf.iloc[idx, 0]))
                    curve_data['ci_lower'].append(float(ci.iloc[idx, 0]))
                    curve_data['ci_upper'].append(float(ci.iloc[idx, 1]))
            
            viz_data['curves'].append(curve_data)
        
        return viz_data
