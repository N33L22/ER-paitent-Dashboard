"""
ML Service - FastAPI Application
XGBoost LOS prediction, LSTM arrival forecasting, survival analysis
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import numpy as np
import pandas as pd
import httpx

from .los_predictor import LOSPredictor
from .arrival_forecaster import ArrivalForecaster
from .survival_analysis import SurvivalAnalyzer


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """Single patient prediction request"""
    acuity: int = Field(ge=1, le=5)
    age: int = Field(ge=0, le=120)
    chief_complaint_category: str = "other"
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    is_weekend: Optional[int] = None
    concurrent_patients: int = 20
    bed_utilization: float = 0.7
    include_shap: bool = True


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[Dict[str, Any]]
    include_shap: bool = False


class ForecastRequest(BaseModel):
    """Arrival forecast request"""
    hours_ahead: int = Field(default=24, ge=1, le=24)
    with_uncertainty: bool = True


class TrainingRequest(BaseModel):
    """Model training request"""
    model_type: str = Field(..., pattern="^(los|arrival|survival)$")
    config: Optional[Dict[str, Any]] = None


class SurvivalRequest(BaseModel):
    """Survival analysis request"""
    stratify_by: Optional[str] = "acuity"
    include_cox: bool = True


class HealthCheck(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime


# =============================================================================
# GLOBAL STATE
# =============================================================================

class ModelStore:
    """Model storage and management"""
    def __init__(self):
        self.los_predictor: Optional[LOSPredictor] = None
        self.arrival_forecaster: Optional[ArrivalForecaster] = None
        self.survival_analyzer: Optional[SurvivalAnalyzer] = None
        self.data_service_url: str = os.getenv("DATA_SERVICE_URL", "http://localhost:8001")
        self.is_initialized: bool = False
        self.training_data: Optional[pd.DataFrame] = None
        self.hourly_data: Optional[pd.DataFrame] = None


model_store = ModelStore()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def fetch_training_data():
    """Fetch training data from data service"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Get patient data
        response = await client.get(
            f"{model_store.data_service_url}/features/los",
            params={"limit": 50000}
        )
        if response.status_code == 200:
            data = response.json()
            model_store.training_data = pd.DataFrame(data)
            logger.info(f"Fetched {len(model_store.training_data)} training records")
        
        # Get hourly data for arrival forecasting
        response = await client.get(
            f"{model_store.data_service_url}/features/arrivals"
        )
        if response.status_code == 200:
            data = response.json()
            model_store.hourly_data = pd.DataFrame(data)
            model_store.hourly_data['timestamp'] = pd.to_datetime(model_store.hourly_data['timestamp'])
            logger.info(f"Fetched {len(model_store.hourly_data)} hourly records")


def prepare_patient_features(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare patient features for prediction"""
    now = datetime.now()
    
    features = {
        'acuity': patient.get('acuity', 3),
        'age': patient.get('age', 50),
        'hour': patient.get('hour', now.hour),
        'day_of_week': patient.get('day_of_week', now.weekday()),
        'is_weekend': patient.get('is_weekend', 1 if now.weekday() >= 5 else 0),
        'is_night': 1 if now.hour < 6 or now.hour >= 22 else 0,
        'is_peak_hours': 1 if 10 <= now.hour <= 22 else 0,
        'is_holiday': 0,
        'concurrent_patients': patient.get('concurrent_patients', 20),
        'bed_utilization': patient.get('bed_utilization', 0.7),
        'current_wait_time': patient.get('current_wait_time', 30),
        'arrivals_last_hour': patient.get('arrivals_last_hour', 20),
    }
    
    # Cyclic encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * now.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * now.month / 12)
    
    # Acuity features
    for level in [1, 2, 3, 4, 5]:
        features[f'acuity_{level}'] = 1 if features['acuity'] == level else 0
    features['is_critical'] = 1 if features['acuity'] <= 2 else 0
    features['is_low_acuity'] = 1 if features['acuity'] >= 4 else 0
    
    # Age features
    age = features['age']
    features['age_normalized'] = (age - 50) / 30
    features['age_group_pediatric'] = 1 if age < 18 else 0
    features['age_group_young_adult'] = 1 if 18 <= age < 40 else 0
    features['age_group_middle_age'] = 1 if 40 <= age < 65 else 0
    features['age_group_senior'] = 1 if age >= 65 else 0
    features['age_group_elderly'] = 1 if age >= 80 else 0
    
    # Chief complaint features
    cc = patient.get('chief_complaint_category', 'other')
    complaint_categories = [
        'chest_pain', 'abdominal_pain', 'shortness_of_breath', 'headache',
        'back_pain', 'fever', 'fall_injury', 'laceration', 'nausea_vomiting', 'dizziness'
    ]
    for cat in complaint_categories:
        features[f'complaint_{cat}'] = 1 if cc == cat else 0
    
    pain_categories = ['chest_pain', 'abdominal_pain', 'back_pain', 'headache', 'general_pain']
    features['is_pain_complaint'] = 1 if cc in pain_categories else 0
    features['is_respiratory'] = 1 if cc in ['shortness_of_breath', 'cough'] else 0
    
    # Interactions
    features['acuity_age_interaction'] = features['acuity'] * features['age'] / 100
    features['peak_hour_critical'] = features['is_peak_hours'] * features['is_critical']
    
    return features


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup"""
    logger.info("Starting ML Service...")
    
    # Initialize model objects
    try:
        model_store.los_predictor = LOSPredictor()
        logger.info("LOS Predictor initialized")
    except Exception as e:
        logger.warning(f"Could not initialize LOS Predictor: {e}")
    
    try:
        model_store.arrival_forecaster = ArrivalForecaster()
        logger.info("Arrival Forecaster initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Arrival Forecaster: {e}")
    
    try:
        model_store.survival_analyzer = SurvivalAnalyzer()
        logger.info("Survival Analyzer initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Survival Analyzer: {e}")
    
    # Try to fetch training data
    try:
        await fetch_training_data()
    except Exception as e:
        logger.warning(f"Could not fetch training data: {e}")
    
    # Auto-train models if data available
    if model_store.training_data is not None and len(model_store.training_data) > 0:
        try:
            if model_store.los_predictor:
                model_store.los_predictor.train(model_store.training_data)
                logger.info("LOS Predictor trained")
        except Exception as e:
            logger.warning(f"Could not train LOS Predictor: {e}")
    
    model_store.is_initialized = True
    logger.info("ML Service initialized")
    
    yield
    
    logger.info("Shutting down ML Service...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ER Patient Flow - ML Service",
    description="XGBoost LOS prediction, LSTM arrival forecasting, survival analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if model_store.is_initialized else "initializing",
        service="ml-service",
        version="1.0.0",
        models_loaded={
            "los_predictor": model_store.los_predictor is not None and model_store.los_predictor.is_trained,
            "arrival_forecaster": model_store.arrival_forecaster is not None and model_store.arrival_forecaster.is_trained,
            "survival_analyzer": model_store.survival_analyzer is not None and model_store.survival_analyzer.is_fitted
        },
        timestamp=datetime.now()
    )


# =============================================================================
# LOS PREDICTION ENDPOINTS
# =============================================================================

@app.post("/predict/los")
async def predict_los(request: PredictionRequest):
    """Predict length of stay for a single patient"""
    if model_store.los_predictor is None or not model_store.los_predictor.is_trained:
        raise HTTPException(status_code=503, detail="LOS model not trained")
    
    # Prepare features
    features = prepare_patient_features(request.model_dump())
    
    # Make prediction
    result = model_store.los_predictor.predict_single(
        features,
        include_shap=request.include_shap
    )
    
    return {
        "predicted_los_minutes": result['predicted_los_minutes'],
        "predicted_los_hours": result['predicted_los_minutes'] / 60,
        "shap_explanation": result.get('shap_explanation'),
        "input_features": {
            "acuity": request.acuity,
            "age": request.age,
            "chief_complaint": request.chief_complaint_category
        }
    }


@app.post("/predict/los/batch")
async def predict_los_batch(request: BatchPredictionRequest):
    """Predict LOS for multiple patients"""
    if model_store.los_predictor is None or not model_store.los_predictor.is_trained:
        raise HTTPException(status_code=503, detail="LOS model not trained")
    
    # Prepare features for all patients
    all_features = [prepare_patient_features(p) for p in request.patients]
    df = pd.DataFrame(all_features)
    
    # Make predictions
    result = model_store.los_predictor.predict(df, include_shap=request.include_shap)
    
    return {
        "predictions": result['predictions'],
        "summary": {
            "mean_los_minutes": result['mean_predicted_los'],
            "median_los_minutes": result['median_predicted_los'],
            "p90_los_minutes": result['p90_predicted_los']
        },
        "count": len(request.patients)
    }


@app.get("/los/feature-importance")
async def get_feature_importance(top_n: int = 20):
    """Get feature importance from LOS model"""
    if model_store.los_predictor is None or not model_store.los_predictor.is_trained:
        raise HTTPException(status_code=503, detail="LOS model not trained")
    
    return {
        "feature_importance": model_store.los_predictor.get_feature_importance(top_n),
        "training_metrics": model_store.los_predictor.training_metrics
    }


# =============================================================================
# ARRIVAL FORECASTING ENDPOINTS
# =============================================================================

@app.post("/predict/arrivals")
async def predict_arrivals(request: ForecastRequest):
    """Forecast patient arrivals"""
    if model_store.arrival_forecaster is None or not model_store.arrival_forecaster.is_trained:
        # Return simulated forecast if model not trained
        now = datetime.now()
        hours = list(range(request.hours_ahead))
        
        # Simulate realistic forecast
        base = 18
        forecasts = []
        for h in hours:
            hour = (now.hour + h) % 24
            # Peak pattern
            multiplier = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.5
            arrival = base * multiplier + np.random.normal(0, 2)
            forecasts.append(max(5, arrival))
        
        return {
            "mean_forecast": forecasts,
            "forecast_horizon_hours": request.hours_ahead,
            "total_predicted_arrivals": sum(forecasts),
            "avg_hourly_arrivals": np.mean(forecasts),
            "note": "Using simulated forecast (model not trained)"
        }
    
    result = model_store.arrival_forecaster.predict(
        model_store.hourly_data,
        with_uncertainty=request.with_uncertainty
    )
    
    # Slice to requested hours
    for key in ['mean_forecast', 'std', 'lower_bound_5', 'upper_bound_95']:
        if key in result:
            result[key] = result[key][:request.hours_ahead]
    
    return result


@app.get("/predict/arrivals/summary")
async def get_arrival_summary(hours: int = 4):
    """Get summary arrival forecast for next N hours"""
    if model_store.arrival_forecaster is None or not model_store.arrival_forecaster.is_trained:
        # Simulated summary
        base_rate = 18
        return {
            "predicted_arrivals": base_rate * hours,
            "avg_hourly": base_rate,
            "hours": hours,
            "note": "Simulated (model not trained)"
        }
    
    return model_store.arrival_forecaster.predict_next_n_hours(
        model_store.hourly_data,
        hours=hours
    )


# =============================================================================
# SURVIVAL ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/survival/fit")
async def fit_survival_models(request: SurvivalRequest):
    """Fit survival analysis models"""
    if model_store.survival_analyzer is None:
        raise HTTPException(status_code=503, detail="Survival analyzer not available")
    
    if model_store.training_data is None:
        raise HTTPException(status_code=503, detail="Training data not available")
    
    df = model_store.training_data.copy()
    
    # Ensure required columns
    if 'event_observed' not in df.columns:
        df['event_observed'] = 1
    
    # Fit Kaplan-Meier
    km_results = model_store.survival_analyzer.fit_kaplan_meier(
        df,
        duration_col='total_los_minutes',
        event_col='event_observed',
        stratify_by=request.stratify_by
    )
    
    # Fit Cox PH if requested
    cox_results = None
    if request.include_cox:
        cox_results = model_store.survival_analyzer.fit_cox_ph(
            df,
            duration_col='total_los_minutes',
            event_col='event_observed'
        )
    
    return {
        "kaplan_meier": km_results,
        "cox_ph": cox_results
    }


@app.get("/survival/curves")
async def get_survival_curves():
    """Get survival curve data for visualization"""
    if model_store.survival_analyzer is None or not model_store.survival_analyzer.is_fitted:
        raise HTTPException(status_code=503, detail="Survival models not fitted")
    
    return model_store.survival_analyzer.get_visualization_data()


@app.get("/survival/percentiles")
async def get_survival_percentiles():
    """Get LOS percentiles from survival analysis"""
    if model_store.survival_analyzer is None or not model_store.survival_analyzer.is_fitted:
        raise HTTPException(status_code=503, detail="Survival models not fitted")
    
    return model_store.survival_analyzer.get_percentile_times()


# =============================================================================
# TRAINING ENDPOINTS
# =============================================================================

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain a model"""
    
    # Fetch fresh data
    await fetch_training_data()
    
    if model_store.training_data is None or len(model_store.training_data) == 0:
        raise HTTPException(status_code=503, detail="No training data available")
    
    if request.model_type == "los":
        if model_store.los_predictor is None:
            raise HTTPException(status_code=503, detail="LOS predictor not initialized")
        
        result = model_store.los_predictor.train(model_store.training_data)
        return {
            "status": "success",
            "model": "los_predictor",
            "metrics": result['metrics'],
            "top_features": result['feature_importance']
        }
    
    elif request.model_type == "arrival":
        if model_store.arrival_forecaster is None:
            raise HTTPException(status_code=503, detail="Arrival forecaster not initialized")
        
        if model_store.hourly_data is None:
            raise HTTPException(status_code=503, detail="Hourly data not available")
        
        result = model_store.arrival_forecaster.train(model_store.hourly_data)
        return {
            "status": "success",
            "model": "arrival_forecaster",
            "metrics": result
        }
    
    elif request.model_type == "survival":
        if model_store.survival_analyzer is None:
            raise HTTPException(status_code=503, detail="Survival analyzer not initialized")
        
        km_result = model_store.survival_analyzer.fit_kaplan_meier(
            model_store.training_data,
            stratify_by='acuity'
        )
        cox_result = model_store.survival_analyzer.fit_cox_ph(model_store.training_data)
        
        return {
            "status": "success",
            "model": "survival_analyzer",
            "kaplan_meier": km_result,
            "cox_ph": cox_result
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")


@app.get("/models/status")
async def get_model_status():
    """Get status of all models"""
    return {
        "los_predictor": {
            "initialized": model_store.los_predictor is not None,
            "trained": model_store.los_predictor.is_trained if model_store.los_predictor else False,
            "metrics": model_store.los_predictor.training_metrics if model_store.los_predictor and model_store.los_predictor.is_trained else None
        },
        "arrival_forecaster": {
            "initialized": model_store.arrival_forecaster is not None,
            "trained": model_store.arrival_forecaster.is_trained if model_store.arrival_forecaster else False
        },
        "survival_analyzer": {
            "initialized": model_store.survival_analyzer is not None,
            "fitted": model_store.survival_analyzer.is_fitted if model_store.survival_analyzer else False
        },
        "training_data_available": model_store.training_data is not None,
        "training_data_size": len(model_store.training_data) if model_store.training_data is not None else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
