"""
LSTM Arrival Forecaster
Predicts patient arrivals 1-24 hours ahead with uncertainty quantification
"""

import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")


class ArrivalForecaster:
    """
    LSTM-based Arrival Forecaster
    
    Features:
    - 168-hour (1 week) lookback window
    - 24-hour forecast horizon
    - Monte Carlo dropout for uncertainty quantification
    - Cyclic time encoding
    """
    
    # Default architecture parameters
    DEFAULT_CONFIG = {
        'lookback_hours': 168,
        'forecast_horizon': 24,
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'dense_units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 10,
        'mc_samples': 100  # Monte Carlo samples for uncertainty
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Arrival Forecaster
        
        Args:
            model_path: Path to saved model
            config: Model configuration
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for arrival forecasting")
        
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.model: Optional[keras.Model] = None
        self.scaler_mean: float = 0.0
        self.scaler_std: float = 1.0
        self.is_trained: bool = False
        self.training_history: Dict[str, List] = {}
        self.n_features: int = 0
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def build_model(self, n_features: int) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            n_features: Number of input features per timestep
            
        Returns:
            Compiled Keras model
        """
        self.n_features = n_features
        
        model = models.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.config['lstm_units_1'],
                return_sequences=True,
                input_shape=(self.config['lookback_hours'], n_features)
            ),
            layers.Dropout(self.config['dropout_rate']),
            
            # Second LSTM layer
            layers.LSTM(
                self.config['lstm_units_2'],
                return_sequences=False
            ),
            layers.Dropout(self.config['dropout_rate']),
            
            # Dense layers
            layers.Dense(self.config['dense_units'], activation='relu'),
            layers.Dropout(self.config['dropout_rate']),
            
            # Output layer (24 hours)
            layers.Dense(self.config['forecast_horizon'])
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def prepare_sequences(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training
        
        Args:
            df: DataFrame with hourly arrivals and features
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Ensure sorted by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Feature columns
        feature_cols = ['arrivals']
        
        # Add temporal features
        if 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Cyclic encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        feature_cols.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        
        # Add lag features
        for lag in [1, 24, 168]:
            col_name = f'arrivals_lag_{lag}'
            df[col_name] = df['arrivals'].shift(lag)
            feature_cols.append(col_name)
        
        # Add rolling statistics
        for window in [24, 168]:
            df[f'arrivals_rolling_mean_{window}'] = df['arrivals'].rolling(window=window).mean()
            df[f'arrivals_rolling_std_{window}'] = df['arrivals'].rolling(window=window).std()
            feature_cols.extend([
                f'arrivals_rolling_mean_{window}',
                f'arrivals_rolling_std_{window}'
            ])
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        # Normalize arrivals
        self.scaler_mean = df['arrivals'].mean()
        self.scaler_std = df['arrivals'].std()
        df['arrivals_scaled'] = (df['arrivals'] - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        X_list = []
        y_list = []
        
        lookback = self.config['lookback_hours']
        horizon = self.config['forecast_horizon']
        
        for i in range(lookback, len(df) - horizon):
            # Input sequence (use scaled arrivals for input)
            X_seq = df.iloc[i-lookback:i][feature_cols].values.copy()
            X_seq[:, 0] = (X_seq[:, 0] - self.scaler_mean) / self.scaler_std  # Scale arrivals
            X_list.append(X_seq)
            
            # Target: next 24 hours (scaled)
            y_seq = (df.iloc[i:i+horizon]['arrivals'].values - self.scaler_mean) / self.scaler_std
            y_list.append(y_seq)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X)} sequences with {len(feature_cols)} features")
        
        return X, y
    
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            df: DataFrame with hourly arrival data
            validation_split: Proportion for validation
            
        Returns:
            Training metrics
        """
        logger.info("Training LSTM Arrival Forecaster...")
        
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        # Split data (time-series aware - no shuffling)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training sequences: {len(X_train)}, Validation: {len(X_val)}")
        
        # Build model
        self.model = self.build_model(n_features=X.shape[2])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Store history
        self.training_history = {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
        
        # Evaluate
        val_predictions = self.model.predict(X_val, verbose=0)
        
        # Unscale for metrics
        y_val_unscaled = y_val * self.scaler_std + self.scaler_mean
        val_pred_unscaled = val_predictions * self.scaler_std + self.scaler_mean
        
        mae = np.mean(np.abs(y_val_unscaled - val_pred_unscaled))
        rmse = np.sqrt(np.mean((y_val_unscaled - val_pred_unscaled) ** 2))
        mape = np.mean(np.abs((y_val_unscaled - val_pred_unscaled) / (y_val_unscaled + 1)) * 100)
        
        self.is_trained = True
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'train_sequences': len(X_train),
            'val_sequences': len(X_val),
            'epochs_trained': len(history.history['loss']),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        logger.info(f"Training complete. MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.1f}%")
        
        return metrics
    
    def predict(
        self,
        recent_data: pd.DataFrame,
        with_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Predict arrivals for next 24 hours
        
        Args:
            recent_data: DataFrame with at least 168 hours of data
            with_uncertainty: Whether to compute uncertainty via MC dropout
            
        Returns:
            Predictions with optional uncertainty intervals
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Prepare sequence
        X, _ = self.prepare_sequences(recent_data)
        
        if len(X) < 1:
            raise ValueError(f"Need at least {self.config['lookback_hours']} hours of data")
        
        # Use last sequence
        X_input = X[-1:].copy()
        
        if with_uncertainty:
            # Monte Carlo dropout for uncertainty
            predictions = []
            
            # Enable dropout during inference
            for _ in range(self.config['mc_samples']):
                pred = self.model(X_input, training=True)  # training=True enables dropout
                predictions.append(pred.numpy()[0])
            
            predictions = np.array(predictions)
            
            # Unscale
            predictions_unscaled = predictions * self.scaler_std + self.scaler_mean
            
            # Statistics
            mean_pred = np.mean(predictions_unscaled, axis=0)
            std_pred = np.std(predictions_unscaled, axis=0)
            p5 = np.percentile(predictions_unscaled, 5, axis=0)
            p95 = np.percentile(predictions_unscaled, 95, axis=0)
            
            result = {
                'mean_forecast': mean_pred.tolist(),
                'std': std_pred.tolist(),
                'lower_bound_5': p5.tolist(),
                'upper_bound_95': p95.tolist(),
                'forecast_horizon_hours': self.config['forecast_horizon'],
                'uncertainty_method': 'mc_dropout',
                'mc_samples': self.config['mc_samples']
            }
        else:
            # Single point prediction
            pred = self.model.predict(X_input, verbose=0)[0]
            pred_unscaled = pred * self.scaler_std + self.scaler_mean
            
            result = {
                'mean_forecast': pred_unscaled.tolist(),
                'forecast_horizon_hours': self.config['forecast_horizon']
            }
        
        # Summary statistics
        result['total_predicted_arrivals'] = float(np.sum(result['mean_forecast']))
        result['avg_hourly_arrivals'] = float(np.mean(result['mean_forecast']))
        result['peak_hour'] = int(np.argmax(result['mean_forecast']))
        result['peak_arrivals'] = float(np.max(result['mean_forecast']))
        
        return result
    
    def predict_next_n_hours(
        self,
        recent_data: pd.DataFrame,
        hours: int = 4
    ) -> Dict[str, float]:
        """
        Get summary prediction for next N hours
        
        Args:
            recent_data: Historical data
            hours: Hours to forecast
            
        Returns:
            Summary statistics
        """
        full_forecast = self.predict(recent_data, with_uncertainty=True)
        
        # Slice to requested hours
        hours = min(hours, self.config['forecast_horizon'])
        
        return {
            'predicted_arrivals': float(np.sum(full_forecast['mean_forecast'][:hours])),
            'avg_hourly': float(np.mean(full_forecast['mean_forecast'][:hours])),
            'lower_bound': float(np.sum(full_forecast.get('lower_bound_5', full_forecast['mean_forecast'])[:hours])),
            'upper_bound': float(np.sum(full_forecast.get('upper_bound_95', full_forecast['mean_forecast'])[:hours])),
            'hours': hours
        }
    
    def save(self, path: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save Keras model
        self.model.save(f"{path}.keras")
        
        # Save metadata
        import json
        metadata = {
            'config': self.config,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'n_features': self.n_features,
            'training_history': self.training_history
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        import json
        
        self.model = keras.models.load_model(f"{path}.keras")
        
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.config = metadata['config']
        self.scaler_mean = metadata['scaler_mean']
        self.scaler_std = metadata['scaler_std']
        self.n_features = metadata['n_features']
        self.training_history = metadata.get('training_history', {})
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
