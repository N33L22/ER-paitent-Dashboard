"""
Data Service - FastAPI Application
Main entry point for the data service

Enhanced with:
- Real-time SSE streaming
- CSV upload and validation
- Synthetic data CSV generation
- Model evaluation data endpoints
"""

import os
import io
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from loguru import logger
import pandas as pd
import numpy as np

from .schemas import (
    PatientJourney, PatientEvent, SystemState, HourlyMetrics,
    DataSummary, HealthCheck, SyntheticDataConfig, PatientDataRequest,
    QueueDataRequest, FeatureEngineeringConfig
)
from .data_loader import MIMICDataLoader
from .synthetic_generator import SyntheticDataGenerator
from .feature_engineer import FeatureEngineer
from .csv_generator import SyntheticCSVGenerator
from .enhanced_streaming import EnhancedRealtimeStreamer, StreamConfig
from .universal_uploader import UniversalDataUploader


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
        self.csv_generator: Optional[SyntheticCSVGenerator] = None
        self.streamer: Optional[EnhancedRealtimeStreamer] = None
        self.uploader: Optional[UniversalDataUploader] = None
        self.uploaded_data: Optional[pd.DataFrame] = None
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
    
    # Initialize additional components
    data_store.csv_generator = SyntheticCSVGenerator(output_dir="./data/synthetic")
    data_store.streamer = EnhancedRealtimeStreamer()
    data_store.uploader = UniversalDataUploader()
    
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
    
    # Replace NaN and infinity values with None for JSON compatibility
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(pd.notnull(result), None)
    
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


# =============================================================================
# REAL-TIME STREAMING ENDPOINTS (SSE)
# =============================================================================

@app.get("/stream/arrivals")
async def stream_arrivals(speed_factor: float = 1.0):
    """
    Stream patient arrivals via Server-Sent Events (SSE)
    
    Parameters:
    - speed_factor: 1.0 = real-time, 10.0 = 10x faster for demos
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    data_store.streamer.config.speed_factor = speed_factor
    
    async def event_generator():
        async for event in data_store.streamer.stream_arrivals():
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/stream/state")
async def stream_system_state(interval: int = 30, speed_factor: float = 1.0):
    """
    Stream ED system state via SSE
    
    Parameters:
    - interval: seconds between updates
    - speed_factor: simulation speed multiplier
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    data_store.streamer.config.speed_factor = speed_factor
    
    async def event_generator():
        async for event in data_store.streamer.stream_system_state(interval):
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.get("/stream/metrics")
async def stream_metrics(interval: int = 60, speed_factor: float = 1.0):
    """Stream real-time metrics via SSE"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    data_store.streamer.config.speed_factor = speed_factor
    
    async def event_generator():
        async for event in data_store.streamer.stream_metrics(interval):
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream/combined")
async def stream_combined(interval: int = 10, speed_factor: float = 1.0):
    """Stream all events (state, arrivals, alerts, metrics) via SSE"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    data_store.streamer.config.speed_factor = speed_factor
    
    async def event_generator():
        async for event in data_store.streamer.stream_combined(interval):
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream/snapshot")
async def get_stream_snapshot():
    """Get current state snapshot without streaming"""
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return data_store.streamer.get_snapshot()


@app.post("/stream/stop")
async def stop_streaming():
    """Stop all active streams"""
    if data_store.streamer:
        data_store.streamer.stop()
    return {"status": "stopped"}


@app.post("/stream/reset")
async def reset_streaming():
    """Reset streaming state"""
    if data_store.streamer:
        data_store.streamer.reset()
    return {"status": "reset"}


# =============================================================================
# CSV UPLOAD ENDPOINTS
# =============================================================================

@app.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    sheet_name: Optional[str] = Form(None)
):
    """
    Upload CSV/Excel/JSON/Parquet file for analysis
    
    Supports:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)
    
    Returns detected schema and data preview
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Read file content
    content = await file.read()
    file_obj = io.BytesIO(content)
    
    # Process upload
    result = data_store.uploader.upload_and_process(
        file_obj, file.filename, sheet_name
    )
    
    if result['success']:
        # Store uploaded data
        data_store.uploaded_data = pd.DataFrame(result['original_data'])
        logger.info(f"Uploaded {len(data_store.uploaded_data)} rows from {file.filename}")
    
    return result


@app.get("/upload/last")
async def get_last_upload():
    """Get the last uploaded dataset"""
    if data_store.uploaded_data is None:
        raise HTTPException(status_code=404, detail="No data uploaded yet")
    
    return {
        "row_count": len(data_store.uploaded_data),
        "columns": data_store.uploaded_data.columns.tolist(),
        "preview": data_store.uploaded_data.head(100).to_dict(orient="records")
    }


@app.get("/upload/data")
async def get_uploaded_data(
    limit: int = 1000,
    offset: int = 0
):
    """Get uploaded data with pagination"""
    if data_store.uploaded_data is None:
        raise HTTPException(status_code=404, detail="No data uploaded yet")
    
    df = data_store.uploaded_data.iloc[offset:offset + limit]
    return df.to_dict(orient="records")


# =============================================================================
# SYNTHETIC CSV GENERATION ENDPOINTS
# =============================================================================

@app.post("/synthetic/generate-csv")
async def generate_synthetic_csv(
    num_patients: int = 10000,
    days: int = 90,
    dataset_type: str = "all"
):
    """
    Generate synthetic CSV datasets
    
    Parameters:
    - num_patients: Number of patients to generate
    - days: Number of days to span
    - dataset_type: 'all', 'arrivals', 'hourly', 'events', 'evaluation'
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        if dataset_type == "all":
            datasets = data_store.csv_generator.generate_all_datasets(
                num_patients=num_patients, days=days
            )
            return {
                "status": "success",
                "datasets_generated": list(datasets.keys()),
                "output_directory": data_store.csv_generator.output_dir,
                "total_rows": sum(len(df) for df in datasets.values())
            }
        
        elif dataset_type == "arrivals":
            df = data_store.csv_generator.generate_patient_arrivals(
                num_patients=num_patients, days=days
            )
            return {
                "status": "success",
                "rows": len(df),
                "preview": df.head(10).to_dict(orient="records")
            }
        
        elif dataset_type == "hourly":
            df = data_store.csv_generator.generate_hourly_metrics(days=days)
            return {
                "status": "success",
                "rows": len(df),
                "preview": df.head(10).to_dict(orient="records")
            }
        
        elif dataset_type == "events":
            df = data_store.csv_generator.generate_patient_events(
                num_patients=min(num_patients, 2000)
            )
            return {
                "status": "success",
                "rows": len(df),
                "preview": df.head(10).to_dict(orient="records")
            }
        
        elif dataset_type == "evaluation":
            df_los = data_store.csv_generator.generate_ml_evaluation_data(
                num_samples=num_patients, task='los_prediction'
            )
            df_adm = data_store.csv_generator.generate_ml_evaluation_data(
                num_samples=num_patients, task='admission_prediction',
                filename='admission_evaluation_data.csv'
            )
            return {
                "status": "success",
                "los_rows": len(df_los),
                "admission_rows": len(df_adm),
                "preview_los": df_los.head(5).to_dict(orient="records"),
                "preview_admission": df_adm.head(5).to_dict(orient="records")
            }
        
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown dataset_type: {dataset_type}"
            )
    
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/synthetic/download/{filename}")
async def download_synthetic_csv(filename: str):
    """Download a generated synthetic CSV file"""
    filepath = os.path.join(data_store.csv_generator.output_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        filepath,
        media_type="text/csv",
        filename=filename
    )


@app.get("/synthetic/list")
async def list_synthetic_files():
    """List available synthetic CSV files"""
    output_dir = data_store.csv_generator.output_dir
    
    if not os.path.exists(output_dir):
        return {"files": []}
    
    files = []
    for f in os.listdir(output_dir):
        if f.endswith('.csv'):
            filepath = os.path.join(output_dir, f)
            files.append({
                "filename": f,
                "size_bytes": os.path.getsize(filepath),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).isoformat()
            })
    
    return {"files": files, "directory": output_dir}


# =============================================================================
# EVALUATION DATA ENDPOINTS
# =============================================================================

@app.get("/evaluation/los-data")
async def get_los_evaluation_data(limit: int = 5000):
    """
    Get data formatted for LOS model evaluation
    
    Returns true values, predictions, and protected attributes
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df.head(limit).copy()
    
    # Simulate predictions (in real scenario, call ML service)
    np.random.seed(42)
    df['predicted_los_minutes'] = df['total_los_minutes'].apply(
        lambda x: max(15, x + np.random.normal(0, x * 0.25))
    )
    df['error'] = df['predicted_los_minutes'] - df['total_los_minutes']
    df['abs_error'] = np.abs(df['error'])
    
    # Add age groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 18, 40, 65, 100],
        labels=['pediatric', 'young_adult', 'middle_age', 'senior']
    )
    
    return {
        "sample_size": len(df),
        "columns": df.columns.tolist(),
        "data": df[[
            'patient_id', 'age', 'age_group', 'gender', 'acuity',
            'total_los_minutes', 'predicted_los_minutes', 
            'error', 'abs_error', 'disposition'
        ]].to_dict(orient="records")
    }


@app.get("/evaluation/admission-data")
async def get_admission_evaluation_data(limit: int = 5000):
    """
    Get data formatted for admission prediction evaluation
    """
    if not data_store.is_initialized:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = data_store.journeys_df.head(limit).copy()
    
    # True admission label
    df['true_admitted'] = (df['disposition'] == 'admitted').astype(int)
    
    # Simulate predictions
    np.random.seed(42)
    base_prob = df['acuity'].map({1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.02})
    df['predicted_probability'] = (base_prob + np.random.normal(0, 0.15, len(df))).clip(0.01, 0.99)
    df['predicted_admitted'] = (df['predicted_probability'] >= 0.5).astype(int)
    
    # Add age groups
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 18, 40, 65, 100],
        labels=['pediatric', 'young_adult', 'middle_age', 'senior']
    )
    
    return {
        "sample_size": len(df),
        "data": df[[
            'patient_id', 'age', 'age_group', 'gender', 'acuity',
            'true_admitted', 'predicted_admitted', 'predicted_probability'
        ]].to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
