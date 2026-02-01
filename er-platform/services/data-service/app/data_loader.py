"""
MIMIC-IV-ED Data Loader
Handles loading and processing of MIMIC-IV Emergency Department data
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from .schemas import (
    PatientJourney, PatientEvent, EventType, Location,
    Disposition, TriageAssessment, VitalSigns, AcuityLevel
)


class MIMICDataLoader:
    """
    Loader for MIMIC-IV-ED dataset
    
    Expected files:
    - edstays.csv: ED stay information
    - triage.csv: Triage assessments
    - vitalsign.csv: Vital signs
    - diagnosis.csv: Diagnoses
    - medrecon.csv: Medication reconciliation
    """
    
    # Chief complaint categories for normalization
    CHIEF_COMPLAINT_CATEGORIES = {
        "chest": "chest_pain",
        "abdominal": "abdominal_pain",
        "head": "headache",
        "back": "back_pain",
        "breath": "shortness_of_breath",
        "fever": "fever",
        "fall": "fall_injury",
        "laceration": "laceration",
        "nausea": "nausea_vomiting",
        "dizziness": "dizziness",
        "weakness": "weakness",
        "mental": "altered_mental_status",
        "cough": "cough",
        "pain": "general_pain",
        "injury": "injury",
    }
    
    def __init__(self, data_path: str = "./data/raw/mimic-iv-ed"):
        """
        Initialize MIMIC data loader
        
        Args:
            data_path: Path to MIMIC-IV-ED data directory
        """
        self.data_path = Path(data_path)
        self._edstays_df: Optional[pd.DataFrame] = None
        self._triage_df: Optional[pd.DataFrame] = None
        self._vitalsign_df: Optional[pd.DataFrame] = None
        self._diagnosis_df: Optional[pd.DataFrame] = None
        
    def is_available(self) -> bool:
        """Check if MIMIC data is available"""
        required_files = ["edstays.csv", "triage.csv"]
        return all((self.data_path / f).exists() for f in required_files)
    
    def load_edstays(self) -> pd.DataFrame:
        """Load ED stays data"""
        if self._edstays_df is None:
            file_path = self.data_path / "edstays.csv"
            if not file_path.exists():
                # Try .csv.gz
                file_path = self.data_path / "edstays.csv.gz"
            
            logger.info(f"Loading edstays from {file_path}")
            self._edstays_df = pd.read_csv(
                file_path,
                parse_dates=["intime", "outtime"]
            )
            
            # Calculate LOS
            self._edstays_df["los_minutes"] = (
                (self._edstays_df["outtime"] - self._edstays_df["intime"])
                .dt.total_seconds() / 60
            )
            
            logger.info(f"Loaded {len(self._edstays_df)} ED stays")
            
        return self._edstays_df
    
    def load_triage(self) -> pd.DataFrame:
        """Load triage assessment data"""
        if self._triage_df is None:
            file_path = self.data_path / "triage.csv"
            if not file_path.exists():
                file_path = self.data_path / "triage.csv.gz"
            
            logger.info(f"Loading triage from {file_path}")
            self._triage_df = pd.read_csv(file_path)
            
            # Normalize chief complaints
            self._triage_df["chiefcomplaint_category"] = (
                self._triage_df["chiefcomplaint"]
                .fillna("unknown")
                .apply(self._categorize_chief_complaint)
            )
            
            logger.info(f"Loaded {len(self._triage_df)} triage records")
            
        return self._triage_df
    
    def load_vitalsigns(self) -> pd.DataFrame:
        """Load vital signs data"""
        if self._vitalsign_df is None:
            file_path = self.data_path / "vitalsign.csv"
            if not file_path.exists():
                file_path = self.data_path / "vitalsign.csv.gz"
            
            logger.info(f"Loading vitalsigns from {file_path}")
            self._vitalsign_df = pd.read_csv(
                file_path,
                parse_dates=["charttime"]
            )
            
            logger.info(f"Loaded {len(self._vitalsign_df)} vital sign records")
            
        return self._vitalsign_df
    
    def _categorize_chief_complaint(self, complaint: str) -> str:
        """Categorize chief complaint into standard categories"""
        if pd.isna(complaint):
            return "other"
        
        complaint_lower = str(complaint).lower()
        
        for keyword, category in self.CHIEF_COMPLAINT_CATEGORIES.items():
            if keyword in complaint_lower:
                return category
        
        return "other"
    
    def build_patient_journeys(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PatientJourney]:
        """
        Build patient journey objects from MIMIC data
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of journeys to return
            
        Returns:
            List of PatientJourney objects
        """
        edstays = self.load_edstays()
        triage = self.load_triage()
        
        # Merge data
        df = edstays.merge(triage, on="stay_id", how="left")
        
        # Apply date filters
        if start_date:
            df = df[df["intime"] >= start_date]
        if end_date:
            df = df[df["intime"] <= end_date]
        
        # Apply limit
        if limit:
            df = df.head(limit)
        
        journeys = []
        for _, row in df.iterrows():
            journey = self._row_to_journey(row)
            if journey:
                journeys.append(journey)
        
        logger.info(f"Built {len(journeys)} patient journeys")
        return journeys
    
    def _row_to_journey(self, row: pd.Series) -> Optional[PatientJourney]:
        """Convert a DataFrame row to PatientJourney object"""
        try:
            # Extract acuity (ESI level)
            acuity = row.get("acuity", 3)
            if pd.isna(acuity):
                acuity = 3  # Default to ESI 3
            acuity = int(acuity)
            
            # Determine disposition
            disposition = self._parse_disposition(row.get("disposition", ""))
            
            # Create journey
            journey = PatientJourney(
                stay_id=str(row["stay_id"]),
                patient_id=str(row.get("subject_id", row["stay_id"])),
                arrival_time=row["intime"],
                departure_time=row.get("outtime"),
                acuity=acuity,
                age=int(row.get("anchor_age", 50)) if not pd.isna(row.get("anchor_age")) else 50,
                gender=row.get("gender"),
                chief_complaint=str(row.get("chiefcomplaint", "unknown")),
                chief_complaint_category=str(row.get("chiefcomplaint_category", "other")),
                disposition=disposition,
                total_los_minutes=row.get("los_minutes")
            )
            
            # Build events timeline
            events = self._build_events(row, journey)
            journey.events = events
            journey.compute_metrics()
            
            return journey
            
        except Exception as e:
            logger.warning(f"Error processing row: {e}")
            return None
    
    def _parse_disposition(self, disposition_str: str) -> Disposition:
        """Parse disposition string to enum"""
        if pd.isna(disposition_str):
            return Disposition.DISCHARGED
        
        disp_lower = str(disposition_str).lower()
        
        if "admit" in disp_lower:
            return Disposition.ADMITTED
        elif "transfer" in disp_lower:
            return Disposition.TRANSFERRED
        elif "left" in disp_lower or "lwbs" in disp_lower:
            return Disposition.LWBS
        elif "ama" in disp_lower or "against" in disp_lower:
            return Disposition.AMA
        elif "expire" in disp_lower or "died" in disp_lower:
            return Disposition.EXPIRED
        else:
            return Disposition.DISCHARGED
    
    def _build_events(
        self,
        row: pd.Series,
        journey: PatientJourney
    ) -> List[PatientEvent]:
        """Build event timeline for a patient journey"""
        events = []
        arrival_time = journey.arrival_time
        
        # Arrival event
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.ARRIVAL,
            timestamp=arrival_time,
            location=Location.WAITING_ROOM
        ))
        
        # Triage event (typically 5-15 min after arrival)
        triage_delay = np.random.uniform(5, 15)
        triage_time = arrival_time + timedelta(minutes=triage_delay)
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.TRIAGE,
            timestamp=triage_time,
            location=Location.TRIAGE,
            duration_minutes=np.random.uniform(5, 10)
        ))
        
        # Bed assignment (varies by acuity)
        acuity_wait_times = {1: 0, 2: 10, 3: 30, 4: 60, 5: 90}
        base_wait = acuity_wait_times.get(journey.acuity, 30)
        bed_wait = base_wait + np.random.exponential(base_wait * 0.5)
        bed_time = triage_time + timedelta(minutes=bed_wait)
        
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.BED_ASSIGNMENT,
            timestamp=bed_time,
            location=Location.MAIN_ED
        ))
        
        # Physician evaluation
        physician_delay = np.random.uniform(10, 30)
        physician_time = bed_time + timedelta(minutes=physician_delay)
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.PHYSICIAN_EVAL,
            timestamp=physician_time,
            location=Location.MAIN_ED,
            duration_minutes=np.random.uniform(15, 45)
        ))
        
        # Disposition event
        if journey.departure_time:
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.DISPOSITION,
                timestamp=journey.departure_time - timedelta(minutes=10),
                location=Location.MAIN_ED
            ))
            
            # Discharge/Admission event
            final_event_type = (
                EventType.ADMISSION if journey.disposition == Disposition.ADMITTED
                else EventType.DISCHARGE
            )
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=final_event_type,
                timestamp=journey.departure_time,
                location=Location.MAIN_ED
            ))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def get_hourly_arrivals(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get hourly arrival counts
        
        Returns DataFrame with columns: timestamp, arrivals, hour, day_of_week
        """
        edstays = self.load_edstays()
        
        df = edstays.copy()
        
        if start_date:
            df = df[df["intime"] >= start_date]
        if end_date:
            df = df[df["intime"] <= end_date]
        
        # Floor to hour
        df["hour_floor"] = df["intime"].dt.floor("H")
        
        # Aggregate
        hourly = df.groupby("hour_floor").agg(
            arrivals=("stay_id", "count")
        ).reset_index()
        
        hourly.columns = ["timestamp", "arrivals"]
        hourly["hour"] = hourly["timestamp"].dt.hour
        hourly["day_of_week"] = hourly["timestamp"].dt.dayofweek
        hourly["is_weekend"] = hourly["day_of_week"].isin([5, 6]).astype(int)
        
        return hourly
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        edstays = self.load_edstays()
        triage = self.load_triage()
        
        # Merge
        df = edstays.merge(triage[["stay_id", "acuity"]], on="stay_id", how="left")
        
        # Acuity distribution
        acuity_dist = df["acuity"].value_counts().to_dict()
        
        # Disposition distribution
        disp_dist = df["disposition"].value_counts().to_dict()
        
        return {
            "total_stays": len(df),
            "total_unique_patients": df["subject_id"].nunique(),
            "date_range_start": df["intime"].min().isoformat(),
            "date_range_end": df["intime"].max().isoformat(),
            "mean_los_minutes": float(df["los_minutes"].mean()),
            "median_los_minutes": float(df["los_minutes"].median()),
            "p90_los_minutes": float(df["los_minutes"].quantile(0.9)),
            "acuity_distribution": {int(k): int(v) for k, v in acuity_dist.items() if not pd.isna(k)},
            "disposition_distribution": disp_dist
        }
