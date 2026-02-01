"""
Event-Driven Data Schemas for ER Patient Flow Platform
Pydantic models for patient events, journeys, and system state
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class EventType(str, Enum):
    """Types of events in a patient's ED journey"""
    ARRIVAL = "arrival"
    TRIAGE = "triage"
    REGISTRATION = "registration"
    BED_ASSIGNMENT = "bed_assignment"
    PHYSICIAN_EVAL = "physician_evaluation"
    LAB_ORDER = "lab_order"
    LAB_RESULT = "lab_result"
    IMAGING_ORDER = "imaging_order"
    IMAGING_RESULT = "imaging_result"
    TREATMENT = "treatment"
    REASSESSMENT = "reassessment"
    DISPOSITION = "disposition"
    DISCHARGE = "discharge"
    ADMISSION = "admission"
    TRANSFER = "transfer"
    LWBS = "left_without_being_seen"


class Disposition(str, Enum):
    """Patient disposition outcomes"""
    DISCHARGED = "discharged"
    ADMITTED = "admitted"
    TRANSFERRED = "transferred"
    LWBS = "left_without_being_seen"
    AMA = "against_medical_advice"
    EXPIRED = "expired"
    OBSERVATION = "observation"


class AcuityLevel(int, Enum):
    """Emergency Severity Index (ESI) levels"""
    RESUSCITATION = 1  # Immediate life-saving intervention
    EMERGENT = 2       # High risk, confused/lethargic, severe pain
    URGENT = 3         # Stable, multiple resources needed
    LESS_URGENT = 4    # One resource needed
    NON_URGENT = 5     # No resources needed


class Location(str, Enum):
    """ED locations/zones"""
    WAITING_ROOM = "waiting_room"
    TRIAGE = "triage"
    REGISTRATION = "registration"
    MAIN_ED = "main_ed"
    RESUSCITATION = "resuscitation"
    FAST_TRACK = "fast_track"
    OBSERVATION = "observation"
    IMAGING = "imaging"
    LAB = "lab"


class PatientEvent(BaseModel):
    """Individual event in a patient's ED journey"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    stay_id: str
    event_type: EventType
    timestamp: datetime
    location: Location
    resource_id: Optional[str] = None
    duration_minutes: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VitalSigns(BaseModel):
    """Patient vital signs at a point in time"""
    timestamp: datetime
    temperature: Optional[float] = None  # Celsius
    heart_rate: Optional[int] = None     # bpm
    respiratory_rate: Optional[int] = None  # breaths/min
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    oxygen_saturation: Optional[float] = None  # percentage
    pain_score: Optional[int] = None  # 0-10


class TriageAssessment(BaseModel):
    """Triage assessment data"""
    acuity: AcuityLevel
    chief_complaint: str
    chief_complaint_category: str
    vital_signs: VitalSigns
    arrival_mode: str = "ambulatory"  # ambulatory, ambulance, helicopter
    notes: Optional[str] = None


class PatientJourney(BaseModel):
    """Complete patient journey through ED"""
    stay_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    arrival_time: datetime
    departure_time: Optional[datetime] = None
    acuity: int = Field(ge=1, le=5)
    age: int = Field(ge=0, le=120)
    gender: Optional[str] = None
    chief_complaint: str
    chief_complaint_category: str
    triage_assessment: Optional[TriageAssessment] = None
    disposition: Optional[Disposition] = None
    events: List[PatientEvent] = Field(default_factory=list)
    total_los_minutes: Optional[float] = None
    
    # Computed timestamps
    triage_time: Optional[datetime] = None
    bed_assignment_time: Optional[datetime] = None
    physician_eval_time: Optional[datetime] = None
    
    # Wait times
    wait_to_triage_minutes: Optional[float] = None
    wait_to_bed_minutes: Optional[float] = None
    wait_to_physician_minutes: Optional[float] = None
    
    def compute_metrics(self):
        """Compute derived metrics from events"""
        if self.arrival_time and self.departure_time:
            delta = self.departure_time - self.arrival_time
            self.total_los_minutes = delta.total_seconds() / 60
        
        for event in self.events:
            if event.event_type == EventType.TRIAGE:
                self.triage_time = event.timestamp
                delta = event.timestamp - self.arrival_time
                self.wait_to_triage_minutes = delta.total_seconds() / 60
            elif event.event_type == EventType.BED_ASSIGNMENT:
                self.bed_assignment_time = event.timestamp
                delta = event.timestamp - self.arrival_time
                self.wait_to_bed_minutes = delta.total_seconds() / 60
            elif event.event_type == EventType.PHYSICIAN_EVAL:
                self.physician_eval_time = event.timestamp
                delta = event.timestamp - self.arrival_time
                self.wait_to_physician_minutes = delta.total_seconds() / 60


class SystemState(BaseModel):
    """Current ED system state snapshot"""
    timestamp: datetime
    total_patients: int
    patients_waiting: int
    patients_in_beds: int
    available_beds: int
    total_beds: int
    bed_utilization: float
    
    # Staff on duty
    physicians_on_duty: int
    nurses_on_duty: int
    
    # Wait times
    current_wait_time_minutes: float
    average_wait_time_minutes: float
    
    # Queue lengths by location
    triage_queue: int
    waiting_room_queue: int
    
    # Recent metrics
    arrivals_last_hour: int
    discharges_last_hour: int
    lwbs_last_hour: int
    
    # Predictions
    predicted_arrivals_next_hour: Optional[float] = None
    predicted_arrivals_next_4_hours: Optional[float] = None


class HourlyMetrics(BaseModel):
    """Hourly aggregated metrics"""
    timestamp: datetime
    hour: int
    day_of_week: int
    arrivals: int
    departures: int
    mean_los_minutes: float
    median_los_minutes: float
    p90_los_minutes: float
    mean_wait_minutes: float
    median_wait_minutes: float
    bed_utilization: float
    lwbs_count: int
    lwbs_rate: float
    
    # By acuity
    arrivals_by_acuity: Dict[int, int] = Field(default_factory=dict)
    los_by_acuity: Dict[int, float] = Field(default_factory=dict)


class DailyMetrics(BaseModel):
    """Daily aggregated metrics"""
    date: datetime
    total_arrivals: int
    total_departures: int
    mean_los_minutes: float
    median_los_minutes: float
    p90_los_minutes: float
    mean_wait_minutes: float
    peak_census: int
    peak_wait_minutes: float
    lwbs_count: int
    lwbs_rate: float
    admission_rate: float


# =============================================================================
# Request/Response Models
# =============================================================================

class PatientDataRequest(BaseModel):
    """Request for patient data with filters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    acuity_levels: Optional[List[int]] = None
    dispositions: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0


class QueueDataRequest(BaseModel):
    """Request for queue evolution data"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    resolution: str = "hourly"  # hourly, daily


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation"""
    num_patients: int = 10000
    start_date: datetime = Field(default_factory=datetime.now)
    simulation_days: int = 30
    base_arrival_rate: float = 20.0  # patients per hour
    seed: int = 42
    
    # Acuity distribution (must sum to 1.0)
    acuity_distribution: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.01, 2: 0.15, 3: 0.50, 4: 0.30, 5: 0.04}
    )
    
    # LOS parameters by acuity (gamma distribution: shape, scale)
    los_params_by_acuity: Dict[int, Dict[str, float]] = Field(
        default_factory=lambda: {
            1: {"shape": 3.0, "scale": 120.0},   # Mean ~360 min
            2: {"shape": 2.5, "scale": 100.0},   # Mean ~250 min
            3: {"shape": 2.0, "scale": 80.0},    # Mean ~160 min
            4: {"shape": 1.5, "scale": 50.0},    # Mean ~75 min
            5: {"shape": 1.2, "scale": 30.0},    # Mean ~36 min
        }
    )


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering"""
    include_temporal_features: bool = True
    include_lag_features: bool = True
    include_rolling_features: bool = True
    lag_hours: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 12, 24, 168])
    rolling_windows: List[int] = Field(default_factory=lambda: [6, 12, 24, 168])


# =============================================================================
# Response Models
# =============================================================================

class DataSummary(BaseModel):
    """Summary of available data"""
    total_patients: int
    total_stays: int
    date_range_start: datetime
    date_range_end: datetime
    acuity_distribution: Dict[int, int]
    disposition_distribution: Dict[str, int]
    mean_los_minutes: float
    median_los_minutes: float


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "data-service"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
