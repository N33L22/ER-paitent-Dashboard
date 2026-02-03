"""
Model Evaluation Module for ER Patient Flow Intelligence Platform
Comprehensive metrics including confusion matrix, accuracy, precision, recall, F1

Authors: Neel, Harsh, Tanishk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from loguru import logger

try:
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, roc_curve, precision_recall_curve,
        mean_absolute_error, mean_squared_error, r2_score,
        classification_report, average_precision_score,
        balanced_accuracy_score, matthews_corrcoef
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available for model evaluation")


class TaskType(str, Enum):
    """Type of ML task"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class ConfusionMatrixResult:
    """Confusion matrix results"""
    matrix: List[List[int]]
    labels: List[str]
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    """Classification model metrics"""
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    sensitivity: float  # Same as recall
    matthews_correlation: float
    roc_auc: Optional[float] = None
    average_precision: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrixResult] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'accuracy': round(self.accuracy, 4),
            'balanced_accuracy': round(self.balanced_accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'specificity': round(self.specificity, 4),
            'sensitivity': round(self.sensitivity, 4),
            'matthews_correlation': round(self.matthews_correlation, 4),
        }
        if self.roc_auc is not None:
            result['roc_auc'] = round(self.roc_auc, 4)
        if self.average_precision is not None:
            result['average_precision'] = round(self.average_precision, 4)
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.to_dict()
        if self.per_class_metrics is not None:
            result['per_class_metrics'] = self.per_class_metrics
        return result


@dataclass
class RegressionMetrics:
    """Regression model metrics"""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2: float   # R-squared
    explained_variance: float
    median_absolute_error: float
    
    def to_dict(self) -> Dict:
        return {
            'mae': round(self.mae, 4),
            'mse': round(self.mse, 4),
            'rmse': round(self.rmse, 4),
            'mape': round(self.mape, 4),
            'r2': round(self.r2, 4),
            'explained_variance': round(self.explained_variance, 4),
            'median_absolute_error': round(self.median_absolute_error, 4)
        }


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report"""
    model_name: str
    task_type: str
    evaluation_timestamp: str
    sample_size: int
    metrics: Union[ClassificationMetrics, RegressionMetrics]
    threshold_analysis: Optional[Dict] = None
    error_analysis: Optional[Dict] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'evaluation_timestamp': self.evaluation_timestamp,
            'sample_size': self.sample_size,
            'metrics': self.metrics.to_dict() if hasattr(self.metrics, 'to_dict') else self.metrics,
            'threshold_analysis': self.threshold_analysis,
            'error_analysis': self.error_analysis,
            'recommendations': self.recommendations
        }


class ModelEvaluator:
    """
    Comprehensive Model Evaluation for Healthcare ML Models
    
    Supports:
    - Binary and multiclass classification (LOS category, admission prediction)
    - Regression (LOS minutes, wait time prediction)
    - Threshold optimization
    - Error analysis by subgroup
    """
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for model evaluation")
        self.last_evaluation: Optional[EvaluationReport] = None
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        average: str = 'weighted',
        model_name: str = 'model'
    ) -> ClassificationMetrics:
        """
        Evaluate classification model performance.
        
        Parameters
        ----------
        y_true : array-like
            Ground truth labels
        y_pred : array-like
            Predicted labels
        y_prob : array-like, optional
            Predicted probabilities for positive class or all classes
        labels : list, optional
            Label names for confusion matrix
        average : str
            Averaging method for multiclass metrics ('weighted', 'macro', 'micro')
        model_name : str
            Name of the model being evaluated
        
        Returns
        -------
        ClassificationMetrics
            Comprehensive classification metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Determine if binary or multiclass
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_classes) <= 2
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if is_binary:
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tp = fp = tn = fn = 0
                if cm.shape[0] == 1:
                    if unique_classes[0] == 1:
                        tp = cm[0, 0]
                    else:
                        tn = cm[0, 0]
        else:
            # For multiclass, aggregate
            tp = np.sum(np.diag(cm))
            fp = fn = tn = 0
            for i in range(len(cm)):
                fp += np.sum(cm[:, i]) - cm[i, i]
                fn += np.sum(cm[i, :]) - cm[i, i]
            tn = np.sum(cm) - tp - fp - fn
        
        # Labels for confusion matrix display
        if labels is None:
            labels = [str(c) for c in unique_classes]
        
        cm_result = ConfusionMatrixResult(
            matrix=cm.tolist(),
            labels=labels,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn)
        )
        
        # Core metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Specificity and sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall/TPR
        
        # ROC AUC (if probabilities provided)
        roc_auc = None
        avg_precision = None
        
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            try:
                if is_binary:
                    # For binary, use positive class probability
                    if y_prob.ndim == 2:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_auc = roc_auc_score(y_true, y_prob_pos)
                    avg_precision = average_precision_score(y_true, y_prob_pos)
                else:
                    # Multiclass AUC
                    roc_auc = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average=average
                    )
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
        
        # Per-class metrics for multiclass
        per_class = None
        if not is_binary:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            per_class = {
                str(k): v for k, v in report.items()
                if k not in ['accuracy', 'macro avg', 'weighted avg']
            }
        
        return ClassificationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            sensitivity=sensitivity,
            matthews_correlation=mcc,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            confusion_matrix=cm_result,
            per_class_metrics=per_class
        )
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = 'model'
    ) -> RegressionMetrics:
        """
        Evaluate regression model performance.
        
        Parameters
        ----------
        y_true : array-like
            Ground truth values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        
        Returns
        -------
        RegressionMetrics
            Comprehensive regression metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zeros)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        # Median absolute error
        median_ae = np.median(np.abs(y_true - y_pred))
        
        return RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2=r2,
            explained_variance=explained_var,
            median_absolute_error=median_ae
        )
    
    def analyze_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance at different classification thresholds.
        
        Useful for optimizing precision-recall tradeoff.
        
        Returns
        -------
        Dict with threshold analysis including optimal thresholds
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05).tolist()
        
        results = []
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'threshold': round(thresh, 2),
                'accuracy': round(acc, 4),
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1_score': round(f1, 4)
            })
        
        # Find optimal thresholds
        best_f1_idx = np.argmax([r['f1_score'] for r in results])
        best_acc_idx = np.argmax([r['accuracy'] for r in results])
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        
        # Precision-recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        return {
            'threshold_analysis': results,
            'optimal_f1_threshold': results[best_f1_idx]['threshold'],
            'optimal_accuracy_threshold': results[best_acc_idx]['threshold'],
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            }
        }
    
    def analyze_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: Optional[Dict[str, np.ndarray]] = None,
        features: Optional[pd.DataFrame] = None,
        task_type: TaskType = TaskType.REGRESSION
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors by subgroups.
        
        Parameters
        ----------
        y_true : array
            Ground truth
        y_pred : array
            Predictions
        groups : dict, optional
            Dictionary of group arrays for subgroup analysis
        features : DataFrame, optional
            Feature matrix for error correlation
        task_type : TaskType
            Type of task (regression/classification)
        
        Returns
        -------
        Dict with error analysis
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if task_type == TaskType.REGRESSION:
            errors = y_true - y_pred
            abs_errors = np.abs(errors)
            
            analysis = {
                'overall': {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'mean_abs_error': float(np.mean(abs_errors)),
                    'median_abs_error': float(np.median(abs_errors)),
                    'error_percentiles': {
                        'p10': float(np.percentile(abs_errors, 10)),
                        'p25': float(np.percentile(abs_errors, 25)),
                        'p50': float(np.percentile(abs_errors, 50)),
                        'p75': float(np.percentile(abs_errors, 75)),
                        'p90': float(np.percentile(abs_errors, 90)),
                        'p99': float(np.percentile(abs_errors, 99))
                    }
                },
                'by_group': {}
            }
            
            # Analyze by groups
            if groups:
                for group_name, group_values in groups.items():
                    group_analysis = {}
                    for val in np.unique(group_values):
                        mask = group_values == val
                        group_errors = abs_errors[mask]
                        group_analysis[str(val)] = {
                            'count': int(mask.sum()),
                            'mae': float(np.mean(group_errors)),
                            'median_ae': float(np.median(group_errors))
                        }
                    analysis['by_group'][group_name] = group_analysis
            
            # High error cases
            high_error_threshold = np.percentile(abs_errors, 90)
            analysis['high_error_cases'] = {
                'threshold': float(high_error_threshold),
                'count': int((abs_errors >= high_error_threshold).sum()),
                'percentage': float((abs_errors >= high_error_threshold).mean() * 100)
            }
        
        else:
            # Classification error analysis
            incorrect = y_true != y_pred
            
            analysis = {
                'overall': {
                    'error_rate': float(incorrect.mean()),
                    'error_count': int(incorrect.sum()),
                    'total_samples': len(y_true)
                },
                'by_group': {},
                'misclassification_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            
            if groups:
                for group_name, group_values in groups.items():
                    group_analysis = {}
                    for val in np.unique(group_values):
                        mask = group_values == val
                        group_errors = incorrect[mask]
                        group_analysis[str(val)] = {
                            'count': int(mask.sum()),
                            'error_rate': float(group_errors.mean()),
                            'error_count': int(group_errors.sum())
                        }
                    analysis['by_group'][group_name] = group_analysis
        
        return analysis
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        model_name: str = 'model',
        groups: Optional[Dict[str, np.ndarray]] = None,
        labels: Optional[List[str]] = None
    ) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.
        
        Parameters
        ----------
        y_true : array
            Ground truth
        y_pred : array
            Predictions
        y_prob : array, optional
            Predicted probabilities
        task_type : TaskType
            Type of ML task
        model_name : str
            Model name
        groups : dict, optional
            Groups for subgroup analysis
        labels : list, optional
            Class labels
        
        Returns
        -------
        EvaluationReport
            Comprehensive report
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        recommendations = []
        
        if task_type == TaskType.REGRESSION:
            metrics = self.evaluate_regression(y_true, y_pred, model_name)
            threshold_analysis = None
            
            # Generate recommendations
            if metrics.r2 < 0.5:
                recommendations.append("Low R² - consider adding more features or using ensemble methods")
            if metrics.mape > 30:
                recommendations.append("High MAPE - model has significant prediction errors for some samples")
            
        else:
            metrics = self.evaluate_classification(
                y_true, y_pred, y_prob, labels, model_name=model_name
            )
            
            # Threshold analysis if probabilities available
            threshold_analysis = None
            if y_prob is not None:
                if y_prob.ndim == 2:
                    y_prob_binary = y_prob[:, 1]
                else:
                    y_prob_binary = y_prob
                threshold_analysis = self.analyze_threshold(y_true, y_prob_binary)
            
            # Generate recommendations
            if metrics.accuracy < 0.7:
                recommendations.append("Low accuracy - consider feature engineering or model tuning")
            if metrics.precision < 0.6:
                recommendations.append("Low precision - many false positives, consider raising threshold")
            if metrics.recall < 0.6:
                recommendations.append("Low recall - missing many positive cases, consider lowering threshold")
            if metrics.f1_score < 0.7:
                recommendations.append("Low F1 - poor balance between precision and recall")
        
        # Error analysis
        error_analysis = self.analyze_errors(y_true, y_pred, groups, task_type=task_type)
        
        report = EvaluationReport(
            model_name=model_name,
            task_type=task_type.value,
            evaluation_timestamp=datetime.now().isoformat(),
            sample_size=len(y_true),
            metrics=metrics,
            threshold_analysis=threshold_analysis,
            error_analysis=error_analysis,
            recommendations=recommendations
        )
        
        self.last_evaluation = report
        return report
    
    def evaluate_los_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        acuity: Optional[np.ndarray] = None,
        age_group: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Specialized evaluation for LOS prediction with clinically relevant metrics.
        
        Parameters
        ----------
        y_true : array
            True LOS in minutes
        y_pred : array
            Predicted LOS in minutes
        acuity : array, optional
            Patient acuity levels for stratification
        age_group : array, optional
            Age groups for stratification
        
        Returns
        -------
        Dict with LOS-specific evaluation
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Basic regression metrics
        reg_metrics = self.evaluate_regression(y_true, y_pred)
        
        # Clinically relevant thresholds (within X minutes)
        errors = np.abs(y_true - y_pred)
        within_30 = (errors <= 30).mean() * 100
        within_60 = (errors <= 60).mean() * 100
        within_120 = (errors <= 120).mean() * 100
        
        # Categorical accuracy (short/medium/long stay)
        def categorize_los(los):
            if los < 120:
                return 'short'
            elif los < 360:
                return 'medium'
            else:
                return 'long'
        
        true_cat = np.array([categorize_los(x) for x in y_true])
        pred_cat = np.array([categorize_los(x) for x in y_pred])
        category_accuracy = (true_cat == pred_cat).mean() * 100
        
        result = {
            'regression_metrics': reg_metrics.to_dict(),
            'clinical_metrics': {
                'within_30_min_pct': round(within_30, 2),
                'within_60_min_pct': round(within_60, 2),
                'within_120_min_pct': round(within_120, 2),
                'category_accuracy_pct': round(category_accuracy, 2)
            }
        }
        
        # Stratified metrics
        groups = {}
        if acuity is not None:
            groups['acuity'] = np.asarray(acuity)
        if age_group is not None:
            groups['age_group'] = np.asarray(age_group)
        
        if groups:
            error_analysis = self.analyze_errors(
                y_true, y_pred, groups, task_type=TaskType.REGRESSION
            )
            result['stratified_analysis'] = error_analysis['by_group']
        
        return result
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        probabilities: Optional[Dict[str, np.ndarray]] = None,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test set.
        
        Parameters
        ----------
        y_true : array
            Ground truth
        predictions : dict
            Dict of model_name -> predictions
        probabilities : dict, optional
            Dict of model_name -> probabilities
        task_type : TaskType
            Type of task
        
        Returns
        -------
        Dict with comparison results
        """
        results = {}
        
        for model_name, y_pred in predictions.items():
            y_prob = probabilities.get(model_name) if probabilities else None
            
            if task_type == TaskType.REGRESSION:
                metrics = self.evaluate_regression(y_true, y_pred)
            else:
                metrics = self.evaluate_classification(y_true, y_pred, y_prob)
            
            results[model_name] = metrics.to_dict()
        
        # Determine best model
        if task_type == TaskType.REGRESSION:
            best_model = min(results.keys(), key=lambda x: results[x]['mae'])
            ranking_metric = 'mae'
        else:
            best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
            ranking_metric = 'f1_score'
        
        return {
            'model_results': results,
            'best_model': best_model,
            'ranking_metric': ranking_metric,
            'comparison_timestamp': datetime.now().isoformat()
        }


# Singleton instance
evaluator = ModelEvaluator()
