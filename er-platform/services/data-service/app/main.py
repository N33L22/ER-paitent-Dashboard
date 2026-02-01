"""
Data Service - FastAPI Application
Main entry point for the data service
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import pandas as pd

from .schemas import (
    PatientJourney, PatientEvent, SystemState, HourlyMetrics,
    DataSummary, HealthCheck, SyntheticDataConfig, PatientDataRequest,
    QueueDataRequest, FeatureEngineeringConfig
)
from .data_loader import MIMICDataLoader
from .synthetic_generator import SyntheticDataGenerator
from .feature_engineer import FeatureEngineer


# =============================================================================
# GLOBAL STATE
# =============================================================================

class DataStore:
    """In-memory data store for the service"""
    def __init__(self):
        self.journeys: List[PatientJourney] = []
        self.journeys_df: Optional[pd.DataFrame] = None
        self.hourly_df: Optional[pd.DataFrame] = None
        self.queue_df: Optional[pd.DataFrame] = None
        self.generator: Optional[SyntheticDataGenerator] = None
        self.mimic_loader: Optional[MIMICDataLoader] = None
        self.feature_engineer: FeatureEngineer = FeatureEngineer()
        self.is_initialized: bool = False


data_store = DataStore()


# =============================================================================
# LIFESPAN / STARTUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize data on startup"""
    logger.info("Starting Data Service...")
    
    # Check for MIMIC data
    mimic_path = os.getenv("MIMIC_DATA_PATH", "./data/raw/mimic-iv-ed")
    data_store.mimic_loader = MIMICDataLoader(mimic_path)
    
    use_synthetic = os.getenv("USE_SYNTHETIC_DATA", "true").lower() == "true"
    
    if data_store.mimic_loader.is_available() and not use_synthetic:
        logger.info("MIMIC-IV-ED data found, loading...")
        try:
            data_store.journeys = data_store.mimic_loader.build_patient_journeys(limit=50000)
        except Exception as e:
            logger.error(f"Error loading MIMIC data: {e}")
            use_synthetic = True
    else:
        use_synthetic = True
    
    if use_synthetic:
        logger.info("Generating synthetic data...")
        num_patients = int(os.getenv("SYNTHETIC_NUM_PATIENTS", "50000"))
        
        config = SyntheticDataConfig(
            num_patients=num_patients,
            start_date=datetime.now() - timedelta(days=365),
            simulation_days=365
        )
        
        data_store.generator = SyntheticDataGenerator(config)
        data_store.journeys = data_store.generator.generate_all_journeys()
        data_store.journeys_df = data_store.generator.to_dataframe()
        data_store.hourly_df = data_store.generator.get_hourly_metrics()
        data_store.queue_df = data_store.generator.get_queue_evolution_data()
    
    data_store.is_initialized = True
    logger.info(f"Data Service initialized with {len(data_store.journeys)} patient journeys")
    
    yield
    
    logger.info("Shutting down Data Service...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ER Patient Flow - Data Service",
    description="Data ingestion, feature engineering, and synthetic data generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
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
        status="healthy" if data_store.is_initialized else "initializing",
        service="data-service",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for container orchestration"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready", "patients_loaded": len(data_store.journeys)}


# =============================================================================
# DATA ENDPOINTS
# =============================================================================

@app.get("/summary", response_model=DataSummary)
async def get_data_summary():
    """Get summary statistics of available data"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df
    
    # Acuity distribution
    acuity_dist = df["acuity"].value_counts().to_dict()
    
    # Disposition distribution
    disp_dist = df["disposition"].value_counts().to_dict()
    
    return DataSummary(
        total_patients=len(df["patient_id"].unique()),
        total_stays=len(df),
        date_range_start=df["arrival_time"].min(),
        date_range_end=df["arrival_time"].max(),
        acuity_distribution={int(k): int(v) for k, v in acuity_dist.items()},
        disposition_distribution={str(k): int(v) for k, v in disp_dist.items()},
        mean_los_minutes=float(df["total_los_minutes"].mean()),
        median_los_minutes=float(df["total_los_minutes"].median())
    )


@app.get("/patients", response_model=List[Dict])
async def get_patients(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    acuity: Optional[List[int]] = Query(None),
    disposition: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0
):
    """Get patient journey data with optional filters"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df.copy()
    
    # Apply filters
    if start_date:
        df = df[df["arrival_time"] >= start_date]
    if end_date:
        df = df[df["arrival_time"] <= end_date]
    if acuity:
        df = df[df["acuity"].isin(acuity)]
    if disposition:
        df = df[df["disposition"] == disposition]
    
    # Paginate
    df = df.iloc[offset:offset + limit]
    
    return df.to_dict(orient="records")


@app.get("/patients/{stay_id}")
async def get_patient_journey(stay_id: str):
    """Get a specific patient journey by stay ID"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    for journey in data_store.journeys:
        if journey.stay_id == stay_id:
            return journey.model_dump()
    
    raise HTTPException(status_code=404, detail="Patient journey not found")


@app.get("/hourly-metrics")
async def get_hourly_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get hourly aggregated metrics"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.hourly_df.copy()
    
    if start_date:
        df = df[df["timestamp"] >= start_date]
    if end_date:
        df = df[df["timestamp"] <= end_date]
    
    # Convert timestamps to ISO format strings for JSON
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    return df.to_dict(orient="records")


@app.get("/daily-metrics")
async def get_daily_metrics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get daily aggregated metrics"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df.copy()
    
    if start_date:
        df = df[df["arrival_time"] >= start_date]
    if end_date:
        df = df[df["arrival_time"] <= end_date]
    
    # Daily aggregation
    df["date"] = df["arrival_time"].dt.date
    
    daily = df.groupby("date").agg(
        total_arrivals=("stay_id", "count"),
        mean_los_minutes=("total_los_minutes", "mean"),
        median_los_minutes=("total_los_minutes", "median"),
        p90_los_minutes=("total_los_minutes", lambda x: x.quantile(0.9)),
        mean_wait_minutes=("wait_to_bed_minutes", "mean"),
        lwbs_count=("disposition", lambda x: (x == "left_without_being_seen").sum())
    ).reset_index()
    
    daily["lwbs_rate"] = daily["lwbs_count"] / daily["total_arrivals"]
    daily["date"] = daily["date"].astype(str)
    
    return daily.to_dict(orient="records")


# =============================================================================
# QUEUE & SYSTEM STATE ENDPOINTS
# =============================================================================

@app.get("/queue-evolution")
async def get_queue_evolution():
    """Get queue evolution data for 3D visualization"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return data_store.queue_df.to_dict(orient="records")


@app.get("/current-state", response_model=SystemState)
async def get_current_state():
    """Get current ED system state (simulated real-time)"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    import numpy as np
    
    # Simulate current state based on time of day
    now = datetime.now()
    hour = now.hour
    
    # Base values with hourly variation
    hour_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.5
    
    total_beds = 20
    patients_in_beds = int(15 + np.random.normal(3 * hour_factor, 2))
    patients_in_beds = max(0, min(total_beds, patients_in_beds))
    
    patients_waiting = int(5 + np.random.normal(5 * hour_factor, 2))
    patients_waiting = max(0, patients_waiting)
    
    return SystemState(
        timestamp=now,
        total_patients=patients_in_beds + patients_waiting,
        patients_waiting=patients_waiting,
        patients_in_beds=patients_in_beds,
        available_beds=total_beds - patients_in_beds,
        total_beds=total_beds,
        bed_utilization=patients_in_beds / total_beds,
        physicians_on_duty=4,
        nurses_on_duty=8,
        current_wait_time_minutes=max(5, 30 + np.random.normal(10 * hour_factor, 10)),
        average_wait_time_minutes=45,
        triage_queue=max(0, int(np.random.normal(3, 1))),
        waiting_room_queue=patients_waiting,
        arrivals_last_hour=int(15 + np.random.normal(5 * hour_factor, 3)),
        discharges_last_hour=int(12 + np.random.normal(3 * hour_factor, 2)),
        lwbs_last_hour=max(0, int(np.random.normal(0.5, 0.5))),
        predicted_arrivals_next_hour=int(18 + np.random.normal(4 * hour_factor, 2)),
        predicted_arrivals_next_4_hours=int(70 + np.random.normal(15 * hour_factor, 5))
    )


# =============================================================================
# FEATURE ENGINEERING ENDPOINTS
# =============================================================================

@app.get("/features/los")
async def get_los_features(
    limit: int = 10000
):
    """Get feature-engineered data for LOS prediction"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df.head(limit).copy()
    
    # Engineer features
    featured_df = data_store.feature_engineer.engineer_los_features(df)
    
    # Select relevant columns
    feature_cols = [c for c in featured_df.columns if c not in [
        "stay_id", "patient_id", "departure_time",
        "chief_complaint", "gender"
    ]]
    
    result = featured_df[feature_cols].copy()
    result["arrival_time"] = result["arrival_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    return result.to_dict(orient="records")


@app.get("/features/arrivals")
async def get_arrival_features():
    """Get feature-engineered hourly arrival data for LSTM"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.hourly_df.copy()
    
    # Add lag and rolling features
    for lag in [1, 2, 4, 8, 12, 24, 168]:
        df[f"arrivals_lag_{lag}h"] = df["arrivals"].shift(lag)
    
    for window in [6, 12, 24, 168]:
        df[f"arrivals_rolling_mean_{window}h"] = df["arrivals"].rolling(window=window).mean()
        df[f"arrivals_rolling_std_{window}h"] = df["arrivals"].rolling(window=window).std()
    
    # Drop NaN rows
    df = df.dropna()
    
    # Convert timestamp
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    return df.to_dict(orient="records")


# =============================================================================
# SYNTHETIC DATA ENDPOINTS
# =============================================================================

@app.post("/synthetic/generate")
async def generate_synthetic_data(config: SyntheticDataConfig):
    """Generate new synthetic dataset with custom configuration"""
    generator = SyntheticDataGenerator(config)
    journeys = generator.generate_all_journeys()
    
    # Update data store
    data_store.journeys = journeys
    data_store.journeys_df = generator.to_dataframe()
    data_store.hourly_df = generator.get_hourly_metrics()
    data_store.queue_df = generator.get_queue_evolution_data()
    data_store.generator = generator
    
    return {
        "status": "success",
        "patients_generated": len(journeys),
        "date_range_start": config.start_date.isoformat(),
        "date_range_end": (config.start_date + timedelta(days=config.simulation_days)).isoformat()
    }


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================

@app.get("/statistics/los-by-acuity")
async def get_los_by_acuity():
    """Get LOS statistics grouped by acuity level"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df
    
    stats = df.groupby("acuity")["total_los_minutes"].agg([
        "count", "mean", "median", "std",
        lambda x: x.quantile(0.1),
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
        lambda x: x.quantile(0.9)
    ]).reset_index()
    
    stats.columns = [
        "acuity", "count", "mean", "median", "std",
        "p10", "p25", "p75", "p90"
    ]
    
    return stats.to_dict(orient="records")


@app.get("/statistics/arrivals-by-hour")
async def get_arrivals_by_hour():
    """Get arrival patterns by hour of day"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.hourly_df
    
    stats = df.groupby("hour")["arrivals"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    return stats.to_dict(orient="records")


@app.get("/statistics/arrivals-by-day")
async def get_arrivals_by_day():
    """Get arrival patterns by day of week"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.hourly_df
    
    stats = df.groupby("day_of_week")["arrivals"].agg([
        "count", "mean", "median", "std", "min", "max"
    ]).reset_index()
    
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    stats["day_name"] = stats["day_of_week"].apply(lambda x: day_names[x])
    
    return stats.to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
