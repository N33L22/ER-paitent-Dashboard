"""
Feature Engineering Pipeline
Creates ML-ready features from patient journey data
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class FeatureEngineer:
    """
    Feature engineering pipeline for ED patient flow data
    
    Creates features for:
    - XGBoost LOS prediction
    - LSTM arrival forecasting
    - Survival analysis
    """
    
    # Holiday dates (US Federal Holidays - extend as needed)
    HOLIDAYS = [
        # 2024
        (1, 1), (1, 15), (2, 19), (5, 27), (6, 19), (7, 4),
        (9, 2), (10, 14), (11, 11), (11, 28), (12, 25),
        # 2025
        (1, 1), (1, 20), (2, 17), (5, 26), (6, 19), (7, 4),
        (9, 1), (10, 13), (11, 11), (11, 27), (12, 25),
    ]
    
    def __init__(
        self,
        include_temporal: bool = True,
        include_lags: bool = True,
        include_rolling: bool = True,
        lag_hours: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        """
        Initialize feature engineer
        
        Args:
            include_temporal: Include time-based features
            include_lags: Include lag features
            include_rolling: Include rolling statistics
            lag_hours: Hours for lag features
            rolling_windows: Window sizes for rolling statistics
        """
        self.include_temporal = include_temporal
        self.include_lags = include_lags
        self.include_rolling = include_rolling
        self.lag_hours = lag_hours or [1, 2, 4, 8, 12, 24, 168]
        self.rolling_windows = rolling_windows or [6, 12, 24, 168]
    
    def engineer_los_features(
        self,
        df: pd.DataFrame,
        current_state: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Engineer features for LOS prediction
        
        Args:
            df: DataFrame with patient data
            current_state: Optional current ED state metrics
            
        Returns:
            DataFrame with engineered features
        """
        result = df.copy()
        
        # Ensure datetime
        if "arrival_time" in result.columns:
            result["arrival_time"] = pd.to_datetime(result["arrival_time"])
        
        # =========================================================
        # TEMPORAL FEATURES
        # =========================================================
        if self.include_temporal and "arrival_time" in result.columns:
            result = self._add_temporal_features(result, "arrival_time")
        
        # =========================================================
        # ACUITY FEATURES
        # =========================================================
        if "acuity" in result.columns:
            # One-hot encode acuity
            for level in [1, 2, 3, 4, 5]:
                result[f"acuity_{level}"] = (result["acuity"] == level).astype(int)
            
            # Is critical (ESI 1-2)
            result["is_critical"] = (result["acuity"] <= 2).astype(int)
            
            # Is low acuity (ESI 4-5)
            result["is_low_acuity"] = (result["acuity"] >= 4).astype(int)
        
        # =========================================================
        # CHIEF COMPLAINT FEATURES
        # =========================================================
        if "chief_complaint_category" in result.columns:
            # One-hot encode top categories
            top_categories = [
                "chest_pain", "abdominal_pain", "shortness_of_breath",
                "headache", "back_pain", "fever", "fall_injury",
                "laceration", "nausea_vomiting", "dizziness"
            ]
            for cat in top_categories:
                result[f"complaint_{cat}"] = (
                    result["chief_complaint_category"] == cat
                ).astype(int)
            
            # Is pain-related
            pain_categories = ["chest_pain", "abdominal_pain", "back_pain", "headache", "general_pain"]
            result["is_pain_complaint"] = result["chief_complaint_category"].isin(pain_categories).astype(int)
            
            # Is respiratory
            resp_categories = ["shortness_of_breath", "cough"]
            result["is_respiratory"] = result["chief_complaint_category"].isin(resp_categories).astype(int)
        
        # =========================================================
        # AGE FEATURES
        # =========================================================
        if "age" in result.columns:
            # Age groups
            result["age_group_pediatric"] = (result["age"] < 18).astype(int)
            result["age_group_young_adult"] = ((result["age"] >= 18) & (result["age"] < 40)).astype(int)
            result["age_group_middle_age"] = ((result["age"] >= 40) & (result["age"] < 65)).astype(int)
            result["age_group_senior"] = (result["age"] >= 65).astype(int)
            result["age_group_elderly"] = (result["age"] >= 80).astype(int)
            
            # Age normalized
            result["age_normalized"] = (result["age"] - 50) / 30
        
        # =========================================================
        # SYSTEM STATE FEATURES (if provided)
        # =========================================================
        if current_state:
            result["concurrent_patients"] = current_state.get("total_patients", 20)
            result["bed_utilization"] = current_state.get("bed_utilization", 0.7)
            result["current_wait_time"] = current_state.get("current_wait_time", 30)
            result["arrivals_last_hour"] = current_state.get("arrivals_last_hour", 20)
        
        # =========================================================
        # INTERACTION FEATURES
        # =========================================================
        if "acuity" in result.columns and "age" in result.columns:
            result["acuity_age_interaction"] = result["acuity"] * result["age"] / 100
        
        if self.include_temporal and "hour" in result.columns and "acuity" in result.columns:
            # Peak hours with high acuity
            result["peak_hour_critical"] = (
                result["is_peak_hours"] * result["is_critical"]
            )
        
        logger.info(f"Engineered {len(result.columns)} features for LOS prediction")
        return result
    
    def engineer_arrival_features(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 168
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features for LSTM arrival forecasting
        
        Args:
            df: DataFrame with hourly arrivals
            lookback_hours: Number of hours for lookback window
            
        Returns:
            Tuple of (X, y) arrays ready for LSTM
        """
        result = df.copy()
        
        # Ensure sorted by time
        result = result.sort_values("timestamp")
        
        # Add temporal features
        result = self._add_temporal_features(result, "timestamp")
        
        # Add lag features
        if self.include_lags:
            for lag in self.lag_hours:
                result[f"arrivals_lag_{lag}h"] = result["arrivals"].shift(lag)
        
        # Add rolling statistics
        if self.include_rolling:
            for window in self.rolling_windows:
                result[f"arrivals_rolling_mean_{window}h"] = (
                    result["arrivals"].rolling(window=window).mean()
                )
                result[f"arrivals_rolling_std_{window}h"] = (
                    result["arrivals"].rolling(window=window).std()
                )
                result[f"arrivals_rolling_max_{window}h"] = (
                    result["arrivals"].rolling(window=window).max()
                )
                result[f"arrivals_rolling_min_{window}h"] = (
                    result["arrivals"].rolling(window=window).min()
                )
        
        # Difference features
        result["arrivals_diff_1h"] = result["arrivals"].diff(1)
        result["arrivals_diff_24h"] = result["arrivals"].diff(24)
        
        # Drop NaN rows from lags/rolling
        result = result.dropna()
        
        # Prepare sequences for LSTM
        feature_cols = [c for c in result.columns if c not in ["timestamp", "arrivals"]]
        
        X_list = []
        y_list = []
        
        for i in range(lookback_hours, len(result) - 24):
            # Input sequence
            X_seq = result.iloc[i-lookback_hours:i][["arrivals"] + feature_cols].values
            X_list.append(X_seq)
            
            # Target: next 24 hours of arrivals
            y_seq = result.iloc[i:i+24]["arrivals"].values
            y_list.append(y_seq)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created LSTM sequences: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def engineer_survival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for survival analysis (LOS as time-to-event)
        
        Args:
            df: DataFrame with patient data
            
        Returns:
            DataFrame with survival analysis features
        """
        result = df.copy()
        
        # Duration (LOS in minutes)
        if "total_los_minutes" in result.columns:
            result["duration"] = result["total_los_minutes"]
        
        # Event observed (1 = discharged/completed, 0 = censored)
        if "disposition" in result.columns:
            result["event_observed"] = result["disposition"].apply(
                lambda x: 1 if x in ["discharged", "admitted", "transferred"] else 0
            )
        else:
            result["event_observed"] = 1
        
        # Add temporal features
        if "arrival_time" in result.columns:
            result = self._add_temporal_features(result, "arrival_time")
        
        # Stratification variables
        if "acuity" in result.columns:
            result["acuity_group"] = result["acuity"].apply(
                lambda x: "high" if x <= 2 else ("medium" if x == 3 else "low")
            )
        
        if "age" in result.columns:
            result["age_group"] = pd.cut(
                result["age"],
                bins=[0, 18, 40, 65, 80, 120],
                labels=["pediatric", "young_adult", "middle_age", "senior", "elderly"]
            )
        
        return result
    
    def _add_temporal_features(
        self,
        df: pd.DataFrame,
        time_column: str
    ) -> pd.DataFrame:
        """Add temporal features based on a timestamp column"""
        result = df.copy()
        
        # Basic temporal features
        result["hour"] = result[time_column].dt.hour
        result["day_of_week"] = result[time_column].dt.dayofweek
        result["day_of_month"] = result[time_column].dt.day
        result["month"] = result[time_column].dt.month
        result["week_of_year"] = result[time_column].dt.isocalendar().week
        
        # Binary features
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        result["is_night"] = result["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
        result["is_morning"] = result["hour"].apply(lambda h: 1 if 6 <= h < 12 else 0)
        result["is_afternoon"] = result["hour"].apply(lambda h: 1 if 12 <= h < 18 else 0)
        result["is_evening"] = result["hour"].apply(lambda h: 1 if 18 <= h < 22 else 0)
        result["is_peak_hours"] = result["hour"].apply(lambda h: 1 if 10 <= h <= 22 else 0)
        
        # Holiday indicator
        result["is_holiday"] = result[time_column].apply(
            lambda dt: 1 if (dt.month, dt.day) in self.HOLIDAYS else 0
        )
        
        # Cyclic encoding (for neural networks)
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
        result["day_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["day_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
        result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
        result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
        
        return result
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groupings for analysis
        
        Returns:
            Dictionary mapping group names to feature names
        """
        return {
            "temporal": [
                "hour", "day_of_week", "day_of_month", "month", "week_of_year",
                "is_weekend", "is_night", "is_morning", "is_afternoon", "is_evening",
                "is_peak_hours", "is_holiday", "hour_sin", "hour_cos",
                "day_sin", "day_cos", "month_sin", "month_cos"
            ],
            "patient_demographics": [
                "age", "age_normalized", "age_group_pediatric", "age_group_young_adult",
                "age_group_middle_age", "age_group_senior", "age_group_elderly"
            ],
            "clinical": [
                "acuity", "acuity_1", "acuity_2", "acuity_3", "acuity_4", "acuity_5",
                "is_critical", "is_low_acuity"
            ],
            "chief_complaint": [
                "complaint_chest_pain", "complaint_abdominal_pain",
                "complaint_shortness_of_breath", "complaint_headache",
                "complaint_back_pain", "complaint_fever", "complaint_fall_injury",
                "complaint_laceration", "complaint_nausea_vomiting", "complaint_dizziness",
                "is_pain_complaint", "is_respiratory"
            ],
            "system_state": [
                "concurrent_patients", "bed_utilization", "current_wait_time",
                "arrivals_last_hour"
            ],
            "interactions": [
                "acuity_age_interaction", "peak_hour_critical"
            ]
        }
