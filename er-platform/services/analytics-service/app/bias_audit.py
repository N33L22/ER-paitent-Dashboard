"""
Bias Auditing
Fairness analysis for healthcare AI models
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from loguru import logger

try:
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score,
        recall_score, f1_score, mean_absolute_error
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FairnessMetric(str, Enum):
    """Fairness metrics"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class BiasType(str, Enum):
    """Types of bias"""
    SELECTION = "selection"
    MEASUREMENT = "measurement"
    ALGORITHMIC = "algorithmic"
    LABEL = "label"
    REPRESENTATION = "representation"


@dataclass
class GroupMetrics:
    """Performance metrics for a demographic group"""
    group_name: str
    group_value: Any
    sample_size: int
    positive_rate: float
    true_positive_rate: float  # Recall/Sensitivity
    false_positive_rate: float
    true_negative_rate: float  # Specificity
    false_negative_rate: float
    precision: float
    f1_score: float
    mean_prediction: float
    mean_error: float


@dataclass
class FairnessResult:
    """Result of fairness analysis"""
    metric: FairnessMetric
    protected_attribute: str
    reference_group: str
    compared_group: str
    reference_value: float
    compared_value: float
    ratio: float  # compared / reference
    difference: float  # compared - reference
    is_fair: bool
    threshold: float
    description: str


@dataclass
class BiasReport:
    """Comprehensive bias audit report"""
    model_name: str
    audit_timestamp: str
    protected_attributes: List[str]
    total_samples: int
    group_metrics: Dict[str, List[GroupMetrics]]
    fairness_results: List[FairnessResult]
    overall_fairness_score: float  # 0-100
    bias_alerts: List[str]
    recommendations: List[str]


class BiasAuditor:
    """
    Bias Auditor for Healthcare AI Models
    
    Analyzes predictions for fairness across protected attributes:
    - Demographics (age, sex, race, ethnicity)
    - Socioeconomic (insurance type)
    - Clinical (triage level)
    
    Implements multiple fairness metrics from ML fairness literature.
    """
    
    # Standard fairness thresholds (80% rule of thumb)
    DEFAULT_RATIO_THRESHOLD = 0.8
    DEFAULT_DIFF_THRESHOLD = 0.1
    
    def __init__(
        self,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
        diff_threshold: float = DEFAULT_DIFF_THRESHOLD
    ):
        self.ratio_threshold = ratio_threshold
        self.diff_threshold = diff_threshold
        self.group_metrics: Dict[str, List[GroupMetrics]] = {}
        self.fairness_results: List[FairnessResult] = []
        
    def compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        groups: np.ndarray,
        attribute_name: str,
        is_regression: bool = False
    ) -> List[GroupMetrics]:
        """
        Compute performance metrics for each group.
        
        Parameters
        ----------
        y_true : array
            True labels/values
        y_pred : array
            Predicted labels/values
        y_prob : array, optional
            Predicted probabilities (for classification)
        groups : array
            Group assignments for each sample
        attribute_name : str
            Name of protected attribute
        is_regression : bool
            Whether this is a regression task
        
        Returns
        -------
        list
            GroupMetrics for each unique group
        """
        unique_groups = np.unique(groups)
        metrics_list = []
        
        for group_val in unique_groups:
            mask = groups == group_val
            n_samples = mask.sum()
            
            if n_samples == 0:
                continue
            
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
            y_prob_g = y_prob[mask] if y_prob is not None else None
            
            if is_regression:
                # Regression metrics (for LOS prediction)
                # Binarize at median for fairness analysis
                threshold = np.median(y_true)
                y_true_bin = (y_true_g > threshold).astype(int)
                y_pred_bin = (y_pred_g > threshold).astype(int)
                
                positive_rate = y_pred_bin.mean()
                mean_prediction = y_pred_g.mean()
                mean_error = np.abs(y_true_g - y_pred_g).mean()
            else:
                y_true_bin = y_true_g.astype(int)
                y_pred_bin = y_pred_g.astype(int)
                positive_rate = y_pred_bin.mean()
                mean_prediction = y_prob_g.mean() if y_prob_g is not None else y_pred_bin.mean()
                mean_error = (y_true_bin != y_pred_bin).mean()
            
            # Confusion matrix values
            if SKLEARN_AVAILABLE and len(np.unique(y_true_bin)) > 1:
                cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            else:
                tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
                fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum()
                tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
                fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum()
            
            total_pos = tp + fn + 1e-10
            total_neg = tn + fp + 1e-10
            total_pred_pos = tp + fp + 1e-10
            
            metrics = GroupMetrics(
                group_name=attribute_name,
                group_value=group_val,
                sample_size=int(n_samples),
                positive_rate=positive_rate,
                true_positive_rate=tp / total_pos,
                false_positive_rate=fp / total_neg,
                true_negative_rate=tn / total_neg,
                false_negative_rate=fn / total_pos,
                precision=tp / total_pred_pos,
                f1_score=2 * tp / (2 * tp + fp + fn + 1e-10),
                mean_prediction=mean_prediction,
                mean_error=mean_error
            )
            metrics_list.append(metrics)
        
        self.group_metrics[attribute_name] = metrics_list
        return metrics_list
    
    def check_demographic_parity(
        self,
        metrics: List[GroupMetrics],
        attribute_name: str,
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Check demographic parity (equal positive rates).
        
        A model satisfies demographic parity if the probability of
        receiving a positive prediction is the same across groups.
        """
        results = []
        
        if not metrics or len(metrics) < 2:
            return results
        
        # Use first group or specified as reference
        ref_idx = 0
        if reference_group:
            for i, m in enumerate(metrics):
                if str(m.group_value) == str(reference_group):
                    ref_idx = i
                    break
        
        ref = metrics[ref_idx]
        
        for m in metrics:
            if m.group_value == ref.group_value:
                continue
            
            ratio = m.positive_rate / (ref.positive_rate + 1e-10)
            diff = m.positive_rate - ref.positive_rate
            
            is_fair = (
                ratio >= self.ratio_threshold and 
                ratio <= 1 / self.ratio_threshold and
                abs(diff) <= self.diff_threshold
            )
            
            results.append(FairnessResult(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                protected_attribute=attribute_name,
                reference_group=str(ref.group_value),
                compared_group=str(m.group_value),
                reference_value=ref.positive_rate,
                compared_value=m.positive_rate,
                ratio=ratio,
                difference=diff,
                is_fair=is_fair,
                threshold=self.ratio_threshold,
                description=f"Positive rate ratio: {ratio:.2f} (threshold: {self.ratio_threshold})"
            ))
        
        return results
    
    def check_equalized_odds(
        self,
        metrics: List[GroupMetrics],
        attribute_name: str,
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Check equalized odds (equal TPR and FPR).
        
        A model satisfies equalized odds if TPR and FPR are equal
        across groups.
        """
        results = []
        
        if not metrics or len(metrics) < 2:
            return results
        
        ref_idx = 0
        if reference_group:
            for i, m in enumerate(metrics):
                if str(m.group_value) == str(reference_group):
                    ref_idx = i
                    break
        
        ref = metrics[ref_idx]
        
        for m in metrics:
            if m.group_value == ref.group_value:
                continue
            
            # TPR difference
            tpr_diff = abs(m.true_positive_rate - ref.true_positive_rate)
            fpr_diff = abs(m.false_positive_rate - ref.false_positive_rate)
            
            is_fair = tpr_diff <= self.diff_threshold and fpr_diff <= self.diff_threshold
            
            results.append(FairnessResult(
                metric=FairnessMetric.EQUALIZED_ODDS,
                protected_attribute=attribute_name,
                reference_group=str(ref.group_value),
                compared_group=str(m.group_value),
                reference_value=ref.true_positive_rate,
                compared_value=m.true_positive_rate,
                ratio=m.true_positive_rate / (ref.true_positive_rate + 1e-10),
                difference=tpr_diff + fpr_diff,  # Combined difference
                is_fair=is_fair,
                threshold=self.diff_threshold,
                description=f"TPR diff: {tpr_diff:.3f}, FPR diff: {fpr_diff:.3f}"
            ))
        
        return results
    
    def check_equal_opportunity(
        self,
        metrics: List[GroupMetrics],
        attribute_name: str,
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Check equal opportunity (equal TPR only).
        
        Equal opportunity requires equal TPR across groups
        (among those who should receive positive outcome).
        """
        results = []
        
        if not metrics or len(metrics) < 2:
            return results
        
        ref_idx = 0
        if reference_group:
            for i, m in enumerate(metrics):
                if str(m.group_value) == str(reference_group):
                    ref_idx = i
                    break
        
        ref = metrics[ref_idx]
        
        for m in metrics:
            if m.group_value == ref.group_value:
                continue
            
            ratio = m.true_positive_rate / (ref.true_positive_rate + 1e-10)
            diff = abs(m.true_positive_rate - ref.true_positive_rate)
            
            is_fair = ratio >= self.ratio_threshold and diff <= self.diff_threshold
            
            results.append(FairnessResult(
                metric=FairnessMetric.EQUAL_OPPORTUNITY,
                protected_attribute=attribute_name,
                reference_group=str(ref.group_value),
                compared_group=str(m.group_value),
                reference_value=ref.true_positive_rate,
                compared_value=m.true_positive_rate,
                ratio=ratio,
                difference=diff,
                is_fair=is_fair,
                threshold=self.ratio_threshold,
                description=f"TPR ratio: {ratio:.2f} (threshold: {self.ratio_threshold})"
            ))
        
        return results
    
    def check_predictive_parity(
        self,
        metrics: List[GroupMetrics],
        attribute_name: str,
        reference_group: Optional[str] = None
    ) -> List[FairnessResult]:
        """
        Check predictive parity (equal precision).
        
        Predictive parity requires equal precision (PPV) across groups.
        """
        results = []
        
        if not metrics or len(metrics) < 2:
            return results
        
        ref_idx = 0
        if reference_group:
            for i, m in enumerate(metrics):
                if str(m.group_value) == str(reference_group):
                    ref_idx = i
                    break
        
        ref = metrics[ref_idx]
        
        for m in metrics:
            if m.group_value == ref.group_value:
                continue
            
            ratio = m.precision / (ref.precision + 1e-10)
            diff = abs(m.precision - ref.precision)
            
            is_fair = ratio >= self.ratio_threshold and diff <= self.diff_threshold
            
            results.append(FairnessResult(
                metric=FairnessMetric.PREDICTIVE_PARITY,
                protected_attribute=attribute_name,
                reference_group=str(ref.group_value),
                compared_group=str(m.group_value),
                reference_value=ref.precision,
                compared_value=m.precision,
                ratio=ratio,
                difference=diff,
                is_fair=is_fair,
                threshold=self.ratio_threshold,
                description=f"Precision ratio: {ratio:.2f}"
            ))
        
        return results
    
    def run_full_audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_df: pd.DataFrame,
        is_regression: bool = False,
        model_name: str = "model"
    ) -> BiasReport:
        """
        Run comprehensive bias audit.
        
        Parameters
        ----------
        y_true : array
            True labels/values
        y_pred : array
            Predicted labels/values
        y_prob : array, optional
            Predicted probabilities
        protected_df : pd.DataFrame
            DataFrame with protected attributes as columns
        is_regression : bool
            Whether this is regression (e.g., LOS prediction)
        model_name : str
            Name of model being audited
        
        Returns
        -------
        BiasReport
            Comprehensive bias report
        """
        from datetime import datetime
        
        protected_attributes = list(protected_df.columns)
        all_fairness_results = []
        bias_alerts = []
        
        # Analyze each protected attribute
        for attr in protected_attributes:
            groups = protected_df[attr].values
            
            # Compute group metrics
            metrics = self.compute_group_metrics(
                y_true, y_pred, y_prob, groups, attr, is_regression
            )
            
            # Run fairness checks
            dp_results = self.check_demographic_parity(metrics, attr)
            eo_results = self.check_equalized_odds(metrics, attr)
            eop_results = self.check_equal_opportunity(metrics, attr)
            pp_results = self.check_predictive_parity(metrics, attr)
            
            all_fairness_results.extend(dp_results)
            all_fairness_results.extend(eo_results)
            all_fairness_results.extend(eop_results)
            all_fairness_results.extend(pp_results)
            
            # Generate alerts for unfair results
            for result in dp_results + eo_results + eop_results + pp_results:
                if not result.is_fair:
                    alert = (
                        f"ALERT: {result.metric.value} violation for {attr}: "
                        f"{result.compared_group} vs {result.reference_group} "
                        f"(ratio: {result.ratio:.2f})"
                    )
                    bias_alerts.append(alert)
        
        # Compute overall fairness score
        if all_fairness_results:
            fair_count = sum(1 for r in all_fairness_results if r.is_fair)
            overall_score = fair_count / len(all_fairness_results) * 100
        else:
            overall_score = 100.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_fairness_results, self.group_metrics
        )
        
        self.fairness_results = all_fairness_results
        
        return BiasReport(
            model_name=model_name,
            audit_timestamp=datetime.now().isoformat(),
            protected_attributes=protected_attributes,
            total_samples=len(y_true),
            group_metrics=self.group_metrics,
            fairness_results=all_fairness_results,
            overall_fairness_score=overall_score,
            bias_alerts=bias_alerts,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        results: List[FairnessResult],
        group_metrics: Dict[str, List[GroupMetrics]]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze patterns in unfair results
        unfair_attrs = {}
        for r in results:
            if not r.is_fair:
                attr = r.protected_attribute
                if attr not in unfair_attrs:
                    unfair_attrs[attr] = []
                unfair_attrs[attr].append(r)
        
        for attr, unfair_results in unfair_attrs.items():
            # Check if disparity is consistent
            metrics = unfair_results[0].metric
            groups = set(r.compared_group for r in unfair_results)
            
            recommendations.append(
                f"Review model performance for {attr}: disparities detected for {groups}"
            )
            
            # Check for underrepresentation
            if attr in group_metrics:
                sizes = [m.sample_size for m in group_metrics[attr]]
                if sizes and max(sizes) > 3 * min(sizes):
                    recommendations.append(
                        f"Consider collecting more data for underrepresented {attr} groups"
                    )
        
        if not recommendations:
            recommendations.append("Model passes basic fairness checks")
        
        return recommendations
    
    def get_disparity_summary(self) -> Dict[str, Any]:
        """Get summary of disparities for visualization"""
        summary = {
            'by_attribute': {},
            'by_metric': {},
            'violations': []
        }
        
        for result in self.fairness_results:
            attr = result.protected_attribute
            metric = result.metric.value
            
            # By attribute
            if attr not in summary['by_attribute']:
                summary['by_attribute'][attr] = {'total': 0, 'violations': 0}
            summary['by_attribute'][attr]['total'] += 1
            if not result.is_fair:
                summary['by_attribute'][attr]['violations'] += 1
            
            # By metric
            if metric not in summary['by_metric']:
                summary['by_metric'][metric] = {'total': 0, 'violations': 0}
            summary['by_metric'][metric]['total'] += 1
            if not result.is_fair:
                summary['by_metric'][metric]['violations'] += 1
            
            # Violations detail
            if not result.is_fair:
                summary['violations'].append({
                    'attribute': attr,
                    'metric': metric,
                    'reference_group': result.reference_group,
                    'compared_group': result.compared_group,
                    'ratio': round(result.ratio, 3),
                    'difference': round(result.difference, 3)
                })
        
        return summary
