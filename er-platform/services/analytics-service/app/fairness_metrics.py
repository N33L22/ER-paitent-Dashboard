"""
Enhanced Fairness and Bias Metrics for Healthcare AI Models
Comprehensive fairness analysis with visualization support

Authors: Neel, Harsh, Tanishk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from loguru import logger

try:
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class HealthcareFairnessMetric(str, Enum):
    """Healthcare-specific fairness metrics"""
    # Standard ML fairness
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    
    # Healthcare-specific
    TREATMENT_EQUALITY = "treatment_equality"  # Equal error rates
    TRIAGE_EQUITY = "triage_equity"  # Equal priority across demographics
    WAIT_TIME_PARITY = "wait_time_parity"  # Equal predicted wait times
    LOS_EQUITY = "los_equity"  # Equal LOS prediction accuracy


@dataclass
class DisparityMetrics:
    """Metrics capturing disparity between groups"""
    attribute: str
    reference_group: str
    comparison_group: str
    metric_name: str
    reference_value: float
    comparison_value: float
    absolute_difference: float
    relative_ratio: float
    disparity_index: float  # 0 = perfect parity, 1 = maximum disparity
    is_significant: bool
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'attribute': self.attribute,
            'reference_group': self.reference_group,
            'comparison_group': self.comparison_group,
            'metric_name': self.metric_name,
            'reference_value': round(self.reference_value, 4),
            'comparison_value': round(self.comparison_value, 4),
            'absolute_difference': round(self.absolute_difference, 4),
            'relative_ratio': round(self.relative_ratio, 4),
            'disparity_index': round(self.disparity_index, 4),
            'is_significant': self.is_significant
        }


@dataclass
class FairnessScorecard:
    """Comprehensive fairness scorecard"""
    model_name: str
    timestamp: str
    overall_fairness_score: float  # 0-100
    grade: str  # A, B, C, D, F
    disparities: List[DisparityMetrics]
    group_performance: Dict[str, Dict]
    recommendations: List[str]
    risk_areas: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'overall_fairness_score': round(self.overall_fairness_score, 2),
            'grade': self.grade,
            'disparities': [d.to_dict() for d in self.disparities],
            'group_performance': self.group_performance,
            'recommendations': self.recommendations,
            'risk_areas': self.risk_areas
        }


class EnhancedFairnessAnalyzer:
    """
    Enhanced Fairness Analyzer for Healthcare ML Models
    
    Provides comprehensive fairness analysis including:
    - Multiple fairness metrics
    - Statistical significance testing
    - Visualization-ready outputs
    - Healthcare-specific equity measures
    - Actionable recommendations
    """
    
    # Thresholds based on EEOC's 4/5ths rule
    RATIO_THRESHOLD = 0.8
    DIFF_THRESHOLD = 0.1
    
    # Grade thresholds
    GRADE_THRESHOLDS = {
        'A': 90,
        'B': 80,
        'C': 70,
        'D': 60,
        'F': 0
    }
    
    def __init__(
        self,
        ratio_threshold: float = 0.8,
        diff_threshold: float = 0.1,
        significance_level: float = 0.05
    ):
        self.ratio_threshold = ratio_threshold
        self.diff_threshold = diff_threshold
        self.significance_level = significance_level
        self.last_analysis: Optional[FairnessScorecard] = None
    
    def compute_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute all confusion matrix-derived metrics"""
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        
        if len(np.unique(y_true)) < 2:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0, 'fnr': 0.0,
                'positive_rate': 0.0
            }
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            cm = np.array([[0, 0], [0, 0]])
        
        tn, fp, fn, tp = cm.ravel()
        
        total_pos = tp + fn + 1e-10
        total_neg = tn + fp + 1e-10
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn + 1e-10),
            'precision': tp / (tp + fp + 1e-10),
            'recall': tp / total_pos,
            'f1': 2 * tp / (2 * tp + fp + fn + 1e-10),
            'tpr': tp / total_pos,  # True Positive Rate (Sensitivity)
            'fpr': fp / total_neg,  # False Positive Rate
            'tnr': tn / total_neg,  # True Negative Rate (Specificity)
            'fnr': fn / total_pos,  # False Negative Rate
            'positive_rate': (tp + fp) / (tp + tn + fp + fn + 1e-10),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def analyze_group_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        groups: np.ndarray,
        attribute_name: str,
        is_regression: bool = False
    ) -> Dict[str, Dict]:
        """Analyze model performance for each group"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        groups = np.asarray(groups)
        
        unique_groups = np.unique(groups)
        group_metrics = {}
        
        for group in unique_groups:
            mask = groups == group
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
            y_prob_g = y_prob[mask] if y_prob is not None else None
            
            if is_regression:
                # Regression metrics
                errors = y_true_g - y_pred_g
                abs_errors = np.abs(errors)
                
                group_metrics[str(group)] = {
                    'count': int(mask.sum()),
                    'mae': float(np.mean(abs_errors)),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'median_ae': float(np.median(abs_errors)),
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'mean_prediction': float(np.mean(y_pred_g)),
                    'mean_actual': float(np.mean(y_true_g))
                }
            else:
                # Binarize for classification
                threshold = 0.5
                if y_pred_g.dtype in [np.float32, np.float64]:
                    y_pred_bin = (y_pred_g > np.median(y_pred_g)).astype(int)
                    y_true_bin = (y_true_g > np.median(y_true_g)).astype(int)
                else:
                    y_pred_bin = y_pred_g.astype(int)
                    y_true_bin = y_true_g.astype(int)
                
                cm_metrics = self.compute_confusion_matrix_metrics(y_true_bin, y_pred_bin)
                cm_metrics['count'] = int(mask.sum())
                
                if y_prob_g is not None and len(np.unique(y_true_bin)) > 1:
                    try:
                        cm_metrics['auc'] = float(roc_auc_score(y_true_bin, y_prob_g))
                    except:
                        cm_metrics['auc'] = None
                
                group_metrics[str(group)] = cm_metrics
        
        return group_metrics
    
    def compute_disparities(
        self,
        group_metrics: Dict[str, Dict],
        metric_name: str,
        attribute_name: str,
        reference_group: Optional[str] = None
    ) -> List[DisparityMetrics]:
        """Compute disparities for a specific metric across groups"""
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return []
        
        # Select reference group (largest or specified)
        if reference_group is None or reference_group not in groups:
            reference_group = max(groups, key=lambda g: group_metrics[g].get('count', 0))
        
        ref_value = group_metrics[reference_group].get(metric_name, 0)
        disparities = []
        
        for group in groups:
            if group == reference_group:
                continue
            
            comp_value = group_metrics[group].get(metric_name, 0)
            
            abs_diff = abs(comp_value - ref_value)
            ratio = comp_value / (ref_value + 1e-10)
            
            # Disparity index: 0 = perfect parity, 1 = maximum disparity
            if ratio >= 1:
                disparity_idx = 1 - (1 / ratio) if ratio > 0 else 1
            else:
                disparity_idx = 1 - ratio
            
            is_significant = (
                ratio < self.ratio_threshold or 
                ratio > 1 / self.ratio_threshold or
                abs_diff > self.diff_threshold
            )
            
            disparities.append(DisparityMetrics(
                attribute=attribute_name,
                reference_group=str(reference_group),
                comparison_group=str(group),
                metric_name=metric_name,
                reference_value=ref_value,
                comparison_value=comp_value,
                absolute_difference=abs_diff,
                relative_ratio=ratio,
                disparity_index=disparity_idx,
                is_significant=is_significant
            ))
        
        return disparities
    
    def generate_fairness_scorecard(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: Dict[str, np.ndarray],
        is_regression: bool = False,
        model_name: str = "model"
    ) -> FairnessScorecard:
        """
        Generate comprehensive fairness scorecard.
        
        Parameters
        ----------
        y_true : array
            Ground truth
        y_pred : array
            Predictions
        y_prob : array, optional
            Probabilities
        protected_attributes : dict
            Dict of attribute_name -> group assignments
        is_regression : bool
            Whether this is regression task
        model_name : str
            Model name
        
        Returns
        -------
        FairnessScorecard
        """
        all_disparities = []
        all_group_performance = {}
        risk_areas = []
        recommendations = []
        
        # Metrics to check
        if is_regression:
            metrics_to_check = ['mae', 'rmse', 'mean_error']
        else:
            metrics_to_check = ['accuracy', 'precision', 'recall', 'tpr', 'fpr', 'positive_rate']
        
        for attr_name, groups in protected_attributes.items():
            # Get group performance
            group_perf = self.analyze_group_performance(
                y_true, y_pred, y_prob, groups, attr_name, is_regression
            )
            all_group_performance[attr_name] = group_perf
            
            # Compute disparities for each metric
            for metric in metrics_to_check:
                disparities = self.compute_disparities(
                    group_perf, metric, attr_name
                )
                all_disparities.extend(disparities)
                
                # Check for significant disparities
                for d in disparities:
                    if d.is_significant:
                        risk_areas.append(
                            f"{attr_name}: {d.comparison_group} has {metric} "
                            f"ratio of {d.relative_ratio:.2f} vs {d.reference_group}"
                        )
        
        # Calculate overall fairness score
        if all_disparities:
            fair_count = sum(1 for d in all_disparities if not d.is_significant)
            overall_score = (fair_count / len(all_disparities)) * 100
        else:
            overall_score = 100.0
        
        # Determine grade
        grade = 'F'
        for g, threshold in self.GRADE_THRESHOLDS.items():
            if overall_score >= threshold:
                grade = g
                break
        
        # Generate recommendations
        if risk_areas:
            unique_attrs = set(d.attribute for d in all_disparities if d.is_significant)
            for attr in unique_attrs:
                attr_disparities = [d for d in all_disparities 
                                   if d.attribute == attr and d.is_significant]
                if attr_disparities:
                    worst = max(attr_disparities, key=lambda x: x.disparity_index)
                    recommendations.append(
                        f"Review model fairness for {attr}: significant disparity "
                        f"in {worst.metric_name} between {worst.comparison_group} "
                        f"and {worst.reference_group}"
                    )
        
        if overall_score >= 90:
            recommendations.append("Model shows good fairness across protected attributes")
        elif overall_score < 70:
            recommendations.append("Consider bias mitigation techniques: reweighting, adversarial debiasing")
            recommendations.append("Collect more balanced training data for underrepresented groups")
        
        scorecard = FairnessScorecard(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            overall_fairness_score=overall_score,
            grade=grade,
            disparities=all_disparities,
            group_performance=all_group_performance,
            recommendations=recommendations,
            risk_areas=risk_areas
        )
        
        self.last_analysis = scorecard
        return scorecard
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization"""
        if self.last_analysis is None:
            return {}
        
        scorecard = self.last_analysis
        
        # Prepare disparity chart data
        disparity_chart = []
        for d in scorecard.disparities:
            disparity_chart.append({
                'attribute': d.attribute,
                'group': d.comparison_group,
                'metric': d.metric_name,
                'ratio': d.relative_ratio,
                'threshold_lower': self.ratio_threshold,
                'threshold_upper': 1 / self.ratio_threshold,
                'is_fair': not d.is_significant
            })
        
        # Prepare group comparison data
        group_comparison = {}
        for attr, groups in scorecard.group_performance.items():
            group_comparison[attr] = {
                'groups': list(groups.keys()),
                'counts': [g.get('count', 0) for g in groups.values()],
                'accuracy': [g.get('accuracy', g.get('mae', 0)) for g in groups.values()]
            }
        
        return {
            'scorecard_summary': {
                'overall_score': scorecard.overall_fairness_score,
                'grade': scorecard.grade,
                'num_disparities': len([d for d in scorecard.disparities if d.is_significant]),
                'total_checks': len(scorecard.disparities)
            },
            'disparity_chart': disparity_chart,
            'group_comparison': group_comparison,
            'risk_areas': scorecard.risk_areas,
            'recommendations': scorecard.recommendations
        }
    
    def analyze_intersectional_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        attributes: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze intersectional bias (e.g., elderly + female).
        
        Intersectional bias can be missed when analyzing attributes separately.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Create intersectional groups
        attr_names = list(attributes.keys())
        
        if len(attr_names) < 2:
            return {'message': 'Need at least 2 attributes for intersectional analysis'}
        
        # Combine first two attributes
        attr1 = np.asarray(attributes[attr_names[0]])
        attr2 = np.asarray(attributes[attr_names[1]])
        
        intersectional_groups = np.array([
            f"{a1}_{a2}" for a1, a2 in zip(attr1, attr2)
        ])
        
        # Analyze performance
        group_perf = self.analyze_group_performance(
            y_true, y_pred, None, intersectional_groups,
            f"{attr_names[0]}_{attr_names[1]}", is_regression=True
        )
        
        return {
            'intersectional_attribute': f"{attr_names[0]} x {attr_names[1]}",
            'groups': list(group_perf.keys()),
            'performance': group_perf,
            'group_sizes': {k: v.get('count', 0) for k, v in group_perf.items()}
        }


# Singleton instance
fairness_analyzer = EnhancedFairnessAnalyzer()
