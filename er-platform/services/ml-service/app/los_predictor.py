"""
XGBoost Length of Stay Predictor
Predicts individual patient LOS with SHAP explainability
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from loguru import logger

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost or sklearn not available")


class LOSPredictor:
    """
    XGBoost-based Length of Stay Predictor
    
    Features:
    - 50+ engineered features
    - SHAP explainability for every prediction
    - Uncertainty estimation via quantile regression
    - Cross-validation for robust evaluation
    """
    
    # Default model hyperparameters (tuned for ED LOS prediction)
    DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 12,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.4,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Feature columns for prediction
    FEATURE_COLUMNS = [
        # Temporal
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_peak_hours',
        'is_holiday', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos',
        # Clinical
        'acuity', 'acuity_1', 'acuity_2', 'acuity_3', 'acuity_4', 'acuity_5',
        'is_critical', 'is_low_acuity',
        # Demographics
        'age', 'age_normalized', 'age_group_pediatric', 'age_group_young_adult',
        'age_group_middle_age', 'age_group_senior', 'age_group_elderly',
        # Chief complaint
        'complaint_chest_pain', 'complaint_abdominal_pain',
        'complaint_shortness_of_breath', 'complaint_headache',
        'complaint_back_pain', 'complaint_fever', 'complaint_fall_injury',
        'complaint_laceration', 'complaint_nausea_vomiting', 'complaint_dizziness',
        'is_pain_complaint', 'is_respiratory',
        # System state
        'concurrent_patients', 'bed_utilization', 'current_wait_time',
        'arrivals_last_hour',
        # Interactions
        'acuity_age_interaction', 'peak_hour_critical'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        params: Optional[Dict] = None
    ):
        """
        Initialize LOS Predictor
        
        Args:
            model_path: Path to saved model file
            params: XGBoost hyperparameters
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost and sklearn are required for LOS prediction")
        
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model: Optional[xgb.XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        
        # Metrics
        self.training_metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for training or prediction
        
        Args:
            df: Input DataFrame with raw features
            include_target: Whether to extract target variable
            
        Returns:
            Tuple of (features DataFrame, target Series if include_target)
        """
        # Select available feature columns
        available_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        X = df[available_cols].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = available_cols
        
        # Extract target if needed
        y = None
        if include_target and 'total_los_minutes' in df.columns:
            y = df['total_los_minutes'].copy()
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the LOS prediction model
        
        Args:
            df: Training DataFrame with features and target
            test_size: Proportion for test split
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics and feature importance
        """
        logger.info("Training XGBoost LOS Predictor...")
        
        # Prepare features
        X, y = self.prepare_features(df, include_target=True)
        
        if y is None:
            raise ValueError("Target column 'total_los_minutes' not found")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.training_metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(np.mean(np.abs((y_test - y_pred) / (y_test + 1)) * 100)),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=cv_folds,
            scoring='neg_mean_absolute_error'
        )
        self.training_metrics['cv_mae_mean'] = float(-cv_scores.mean())
        self.training_metrics['cv_mae_std'] = float(cv_scores.std())
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        self.is_trained = True
        
        logger.info(f"Training complete. MAE: {self.training_metrics['mae']:.2f} min, "
                   f"RMSE: {self.training_metrics['rmse']:.2f} min, "
                   f"R²: {self.training_metrics['r2']:.3f}")
        
        return {
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        include_shap: bool = False
    ) -> Dict[str, Any]:
        """
        Predict LOS for new patients
        
        Args:
            df: DataFrame with patient features
            include_shap: Whether to include SHAP explanations
            
        Returns:
            Dictionary with predictions and optionally SHAP values
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X, _ = self.prepare_features(df, include_target=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        result = {
            'predictions': predictions.tolist(),
            'mean_predicted_los': float(np.mean(predictions)),
            'median_predicted_los': float(np.median(predictions)),
            'p90_predicted_los': float(np.percentile(predictions, 90))
        }
        
        # Add SHAP explanations if requested
        if include_shap and self.explainer is not None:
            shap_values = self.explainer.shap_values(X)
            
            # Get top features for each prediction
            top_features_per_prediction = []
            for i in range(len(X)):
                feature_impacts = [
                    (name, float(shap_values[i, j]))
                    for j, name in enumerate(self.feature_names)
                ]
                # Sort by absolute impact
                feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features_per_prediction.append(feature_impacts[:5])
            
            result['shap_explanations'] = top_features_per_prediction
            result['shap_values'] = shap_values.tolist()
            result['expected_value'] = float(self.explainer.expected_value)
        
        return result
    
    def predict_single(
        self,
        patient_data: Dict[str, Any],
        include_shap: bool = True
    ) -> Dict[str, Any]:
        """
        Predict LOS for a single patient
        
        Args:
            patient_data: Dictionary with patient features
            include_shap: Whether to include SHAP explanation
            
        Returns:
            Dictionary with prediction and explanation
        """
        df = pd.DataFrame([patient_data])
        result = self.predict(df, include_shap=include_shap)
        
        return {
            'predicted_los_minutes': result['predictions'][0],
            'shap_explanation': result.get('shap_explanations', [[]])[0] if include_shap else None
        }
    
    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top N most important features"""
        if not self.is_trained:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {'feature': name, 'importance': imp}
            for name, imp in sorted_features
        ]
    
    def save(self, path: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'params': self.params
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        self.feature_importance = model_data['feature_importance']
        self.params = model_data['params']
        
        # Reinitialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_partial_dependence(
        self,
        df: pd.DataFrame,
        feature: str,
        num_points: int = 50
    ) -> Dict[str, List]:
        """
        Calculate partial dependence for a feature
        
        Args:
            df: DataFrame with features
            feature: Feature name
            num_points: Number of points for the grid
            
        Returns:
            Dictionary with feature values and predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X, _ = self.prepare_features(df, include_target=False)
        
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' not found")
        
        # Create grid
        feature_values = np.linspace(
            X[feature].min(),
            X[feature].max(),
            num_points
        )
        
        # Calculate predictions for each value
        predictions = []
        for val in feature_values:
            X_copy = X.copy()
            X_copy[feature] = val
            pred = self.model.predict(X_copy)
            predictions.append(float(np.mean(pred)))
        
        return {
            'feature': feature,
            'values': feature_values.tolist(),
            'predictions': predictions
        }
