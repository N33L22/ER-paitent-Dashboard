"""
Anomaly Detection
Multi-method anomaly detection for ED operations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
from loguru import logger

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using statistical fallback")


class AnomalyType(str, Enum):
    """Types of anomalies"""
    STATISTICAL = "statistical"  # Z-score based
    ISOLATION = "isolation_forest"  # Isolation Forest
    LOCAL_OUTLIER = "local_outlier"  # LOF
    TEMPORAL = "temporal"  # Time-series based
    MULTIVARIATE = "multivariate"  # Multi-dimensional


class AnomalySeverity(str, Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly"""
    timestamp: datetime
    metric: str
    value: float
    expected_value: float
    deviation: float
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class AnomalyReport:
    """Report of anomaly detection"""
    detection_time: datetime
    period_start: datetime
    period_end: datetime
    total_anomalies: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    by_metric: Dict[str, int]
    anomalies: List[Anomaly]
    health_score: float  # 0-100, lower = more anomalous


class AnomalyDetector:
    """
    Multi-method Anomaly Detection for ED Operations
    
    Combines multiple detection methods:
    1. Statistical (Z-score, IQR)
    2. Isolation Forest
    3. Local Outlier Factor
    4. Temporal (change detection)
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        contamination: float = 0.05,
        window_size: int = 24
    ):
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.window_size = window_size
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
    def fit(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        fit_isolation_forest: bool = True,
        fit_lof: bool = True
    ) -> None:
        """
        Fit anomaly detection models on historical data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Historical time series data
        metrics : list
            Column names of metrics to monitor
        fit_isolation_forest : bool
            Whether to fit Isolation Forest
        fit_lof : bool
            Whether to fit Local Outlier Factor
        """
        # Compute baseline statistics
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                self.baseline_stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std() + 1e-6,
                    'median': values.median(),
                    'q1': values.quantile(0.25),
                    'q3': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max()
                }
        
        # Fit scalers
        X = df[metrics].dropna()
        if len(X) == 0:
            return
            
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
            
            # Fit Isolation Forest
            if fit_isolation_forest:
                iso_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                iso_forest.fit(X_scaled)
                self.models['isolation_forest'] = iso_forest
            
            # Fit LOF
            if fit_lof:
                lof = LocalOutlierFactor(
                    n_neighbors=min(20, len(X) - 1),
                    contamination=self.contamination,
                    novelty=True
                )
                lof.fit(X_scaled)
                self.models['lof'] = lof
    
    def detect_statistical(
        self,
        values: np.ndarray,
        metric: str
    ) -> List[Tuple[int, float, float]]:
        """
        Detect anomalies using statistical methods.
        
        Returns list of (index, z_score, severity_score)
        """
        anomalies = []
        
        if metric not in self.baseline_stats:
            # Use current data statistics
            mean = np.nanmean(values)
            std = np.nanstd(values) + 1e-6
        else:
            mean = self.baseline_stats[metric]['mean']
            std = self.baseline_stats[metric]['std']
        
        z_scores = (values - mean) / std
        
        for i, z in enumerate(z_scores):
            if abs(z) > self.z_threshold:
                severity = min(abs(z) / self.z_threshold, 4.0) / 4.0  # 0-1
                anomalies.append((i, z, severity))
        
        return anomalies
    
    def detect_isolation_forest(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using Isolation Forest.
        
        Returns list of (index, anomaly_score)
        """
        if not SKLEARN_AVAILABLE or 'isolation_forest' not in self.models:
            return []
        
        X = df[metrics].dropna()
        if len(X) == 0:
            return []
            
        X_scaled = self.scalers['main'].transform(X)
        
        # Predict (-1 = anomaly, 1 = normal)
        predictions = self.models['isolation_forest'].predict(X_scaled)
        scores = self.models['isolation_forest'].decision_function(X_scaled)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                # Convert score to 0-1 (more negative = more anomalous)
                severity = max(0, min(1, -score))
                anomalies.append((X.index[i], severity))
        
        return anomalies
    
    def detect_lof(
        self,
        df: pd.DataFrame,
        metrics: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Detect anomalies using Local Outlier Factor.
        
        Returns list of (index, anomaly_score)
        """
        if not SKLEARN_AVAILABLE or 'lof' not in self.models:
            return []
        
        X = df[metrics].dropna()
        if len(X) == 0:
            return []
            
        X_scaled = self.scalers['main'].transform(X)
        
        predictions = self.models['lof'].predict(X_scaled)
        scores = self.models['lof'].decision_function(X_scaled)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                severity = max(0, min(1, -score))
                anomalies.append((X.index[i], severity))
        
        return anomalies
    
    def detect_temporal(
        self,
        values: np.ndarray,
        metric: str
    ) -> List[Tuple[int, float, str]]:
        """
        Detect temporal anomalies (sudden changes).
        
        Returns list of (index, change_magnitude, change_type)
        """
        if len(values) < self.window_size:
            return []
        
        anomalies = []
        
        # Rolling statistics
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size:i]
            current = values[i]
            
            window_mean = np.nanmean(window)
            window_std = np.nanstd(window) + 1e-6
            
            # Point change
            z = (current - window_mean) / window_std
            if abs(z) > self.z_threshold:
                change_type = "spike" if z > 0 else "drop"
                anomalies.append((i, abs(z), change_type))
            
            # Trend change (compare recent to historical)
            if i >= self.window_size * 2:
                recent = values[i - self.window_size // 2:i]
                historical = values[i - self.window_size:i - self.window_size // 2]
                
                recent_mean = np.nanmean(recent)
                hist_mean = np.nanmean(historical)
                
                trend_change = (recent_mean - hist_mean) / (np.nanstd(historical) + 1e-6)
                if abs(trend_change) > self.z_threshold:
                    change_type = "upward_trend" if trend_change > 0 else "downward_trend"
                    anomalies.append((i, abs(trend_change), change_type))
        
        return anomalies
    
    def _severity_from_score(self, score: float) -> AnomalySeverity:
        """Convert numeric score to severity level"""
        if score >= 0.75:
            return AnomalySeverity.CRITICAL
        elif score >= 0.5:
            return AnomalySeverity.HIGH
        elif score >= 0.25:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def detect_all(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        timestamp_col: str = 'timestamp'
    ) -> AnomalyReport:
        """
        Run all detection methods and compile report.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to analyze
        metrics : list
            Columns to check
        timestamp_col : str
            Name of timestamp column
        
        Returns
        -------
        AnomalyReport
            Comprehensive anomaly report
        """
        anomalies = []
        
        # Get timestamps
        if timestamp_col in df.columns:
            timestamps = pd.to_datetime(df[timestamp_col])
        else:
            timestamps = pd.date_range(
                start=datetime.now() - timedelta(hours=len(df)),
                periods=len(df),
                freq='H'
            )
        
        # Statistical detection per metric
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            values = df[metric].values
            stat_anomalies = self.detect_statistical(values, metric)
            
            for idx, z_score, severity in stat_anomalies:
                expected = self.baseline_stats.get(metric, {}).get('mean', np.nanmean(values))
                anomalies.append(Anomaly(
                    timestamp=timestamps.iloc[idx] if idx < len(timestamps) else datetime.now(),
                    metric=metric,
                    value=values[idx],
                    expected_value=expected,
                    deviation=z_score,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=self._severity_from_score(severity),
                    score=severity,
                    description=f"{metric} is {abs(z_score):.1f} std from expected"
                ))
        
        # Temporal detection
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            values = df[metric].values
            temporal_anomalies = self.detect_temporal(values, metric)
            
            for idx, magnitude, change_type in temporal_anomalies:
                anomalies.append(Anomaly(
                    timestamp=timestamps.iloc[idx] if idx < len(timestamps) else datetime.now(),
                    metric=metric,
                    value=values[idx],
                    expected_value=np.nanmean(values[max(0, idx-self.window_size):idx]),
                    deviation=magnitude,
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=self._severity_from_score(magnitude / 5),
                    score=magnitude / 5,
                    context={'change_type': change_type},
                    description=f"Detected {change_type} in {metric}"
                ))
        
        # Multivariate detection (Isolation Forest)
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            iso_anomalies = self.detect_isolation_forest(df, available_metrics)
            for idx, score in iso_anomalies:
                anomalies.append(Anomaly(
                    timestamp=timestamps.iloc[idx] if idx < len(timestamps) else datetime.now(),
                    metric='multivariate',
                    value=0,
                    expected_value=0,
                    deviation=score,
                    anomaly_type=AnomalyType.ISOLATION,
                    severity=self._severity_from_score(score),
                    score=score,
                    description=f"Multivariate anomaly detected (score: {score:.2f})"
                ))
            
            # LOF detection
            lof_anomalies = self.detect_lof(df, available_metrics)
            for idx, score in lof_anomalies:
                anomalies.append(Anomaly(
                    timestamp=timestamps.iloc[idx] if idx < len(timestamps) else datetime.now(),
                    metric='multivariate',
                    value=0,
                    expected_value=0,
                    deviation=score,
                    anomaly_type=AnomalyType.LOCAL_OUTLIER,
                    severity=self._severity_from_score(score),
                    score=score,
                    description=f"Local outlier detected (score: {score:.2f})"
                ))
        
        # Compile report
        by_severity = {}
        by_type = {}
        by_metric = {}
        
        for a in anomalies:
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
            by_type[a.anomaly_type.value] = by_type.get(a.anomaly_type.value, 0) + 1
            by_metric[a.metric] = by_metric.get(a.metric, 0) + 1
        
        # Health score: fewer anomalies = higher score
        anomaly_rate = len(anomalies) / max(1, len(df) * len(metrics))
        health_score = max(0, 100 - anomaly_rate * 1000)
        
        return AnomalyReport(
            detection_time=datetime.now(),
            period_start=timestamps.iloc[0] if len(timestamps) > 0 else datetime.now(),
            period_end=timestamps.iloc[-1] if len(timestamps) > 0 else datetime.now(),
            total_anomalies=len(anomalies),
            by_severity=by_severity,
            by_type=by_type,
            by_metric=by_metric,
            anomalies=sorted(anomalies, key=lambda x: x.score, reverse=True),
            health_score=health_score
        )
    
    def get_anomaly_timeline(
        self,
        report: AnomalyReport,
        bucket_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get anomalies bucketed by time for timeline visualization"""
        if not report.anomalies:
            return []
        
        # Bucket anomalies
        buckets: Dict[datetime, List[Anomaly]] = {}
        
        for a in report.anomalies:
            bucket_time = a.timestamp.replace(
                minute=0, second=0, microsecond=0
            )
            if bucket_time not in buckets:
                buckets[bucket_time] = []
            buckets[bucket_time].append(a)
        
        # Build timeline
        timeline = []
        for time, anomaly_list in sorted(buckets.items()):
            max_severity = max(
                a.severity.value for a in anomaly_list
            )
            timeline.append({
                'timestamp': time.isoformat(),
                'count': len(anomaly_list),
                'max_severity': max_severity,
                'metrics': list(set(a.metric for a in anomaly_list)),
                'types': list(set(a.anomaly_type.value for a in anomaly_list))
            })
        
        return timeline
