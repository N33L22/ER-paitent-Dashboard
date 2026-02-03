"""
Synthetic Data Generator
Generates realistic ED patient data using statistical models
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from .schemas import (
    PatientJourney, PatientEvent, EventType, Location,
    Disposition, TriageAssessment, VitalSigns, AcuityLevel,
    SyntheticDataConfig, HourlyMetrics
)


class SyntheticDataGenerator:
    """
    Generates realistic synthetic ED patient data
    
    Uses statistical models based on published ED research:
    - Non-homogeneous Poisson process for arrivals
    - Gamma distributions for LOS by acuity
    - Empirical distributions for service times
    """
    
    # Hourly arrival rate multipliers (relative to base rate)
    # Peak hours: 10 AM - 10 PM
    HOURLY_PATTERNS = {
        0: 0.5, 1: 0.4, 2: 0.35, 3: 0.3, 4: 0.3, 5: 0.35,
        6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.15,
        12: 1.2, 13: 1.15, 14: 1.1, 15: 1.1, 16: 1.15, 17: 1.2,
        18: 1.25, 19: 1.2, 20: 1.1, 21: 1.0, 22: 0.8, 23: 0.65
    }
    
    # Day of week multipliers (0=Monday, 6=Sunday)
    DAY_OF_WEEK_PATTERNS = {
        0: 1.0,   # Monday
        1: 0.95,  # Tuesday
        2: 0.95,  # Wednesday
        3: 0.95,  # Thursday
        4: 1.0,   # Friday
        5: 1.1,   # Saturday
        6: 1.05   # Sunday
    }
    
    # Chief complaint categories with probabilities
    CHIEF_COMPLAINTS = {
        "chest_pain": 0.10,
        "abdominal_pain": 0.12,
        "shortness_of_breath": 0.08,
        "headache": 0.06,
        "back_pain": 0.07,
        "fever": 0.08,
        "fall_injury": 0.06,
        "laceration": 0.05,
        "nausea_vomiting": 0.06,
        "dizziness": 0.04,
        "weakness": 0.05,
        "altered_mental_status": 0.03,
        "cough": 0.05,
        "general_pain": 0.08,
        "injury": 0.07
    }
    
    # Age distribution by acuity (mean, std)
    AGE_BY_ACUITY = {
        1: (65, 15),
        2: (58, 18),
        3: (45, 20),
        4: (35, 18),
        5: (30, 15)
    }
    
    # Disposition probabilities by acuity
    DISPOSITION_BY_ACUITY = {
        1: {"admitted": 0.70, "discharged": 0.15, "transferred": 0.10, "expired": 0.05},
        2: {"admitted": 0.45, "discharged": 0.45, "transferred": 0.08, "expired": 0.02},
        3: {"admitted": 0.25, "discharged": 0.70, "transferred": 0.04, "lwbs": 0.01},
        4: {"admitted": 0.08, "discharged": 0.85, "transferred": 0.02, "lwbs": 0.05},
        5: {"admitted": 0.02, "discharged": 0.88, "transferred": 0.01, "lwbs": 0.09}
    }
    
    def __init__(self, config: Optional[SyntheticDataConfig] = None):
        """
        Initialize synthetic data generator
        
        Args:
            config: Configuration for data generation
        """
        self.config = config or SyntheticDataConfig()
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
        self._journeys: List[PatientJourney] = []
        self._hourly_df: Optional[pd.DataFrame] = None
        
    def generate_arrivals(self) -> List[datetime]:
        """
        Generate arrival times using non-homogeneous Poisson process
        
        Returns:
            List of arrival timestamps
        """
        arrivals = []
        current_time = self.config.start_date
        end_time = current_time + timedelta(days=self.config.simulation_days)
        
        while current_time < end_time:
            # Get rate multipliers
            hour_mult = self.HOURLY_PATTERNS[current_time.hour]
            dow_mult = self.DAY_OF_WEEK_PATTERNS[current_time.weekday()]
            
            # Effective arrival rate
            effective_rate = self.config.base_arrival_rate * hour_mult * dow_mult
            
            # Generate inter-arrival time (exponential distribution)
            inter_arrival = np.random.exponential(60 / effective_rate)  # in minutes
            current_time += timedelta(minutes=inter_arrival)
            
            if current_time < end_time:
                arrivals.append(current_time)
        
        logger.info(f"Generated {len(arrivals)} arrivals over {self.config.simulation_days} days")
        return arrivals
    
    def generate_acuity(self) -> int:
        """Generate ESI acuity level based on configured distribution"""
        levels = list(self.config.acuity_distribution.keys())
        probs = list(self.config.acuity_distribution.values())
        return np.random.choice(levels, p=probs)
    
    def generate_los(self, acuity: int) -> float:
        """
        Generate length of stay based on acuity using gamma distribution
        
        Args:
            acuity: ESI acuity level (1-5)
            
        Returns:
            LOS in minutes
        """
        params = self.config.los_params_by_acuity.get(
            acuity,
            {"shape": 2.0, "scale": 60.0}
        )
        los = np.random.gamma(params["shape"], params["scale"])
        
        # Add some randomness and ensure minimum
        los = max(15, los + np.random.normal(0, 10))
        
        # Cap at reasonable maximum (24 hours)
        return min(los, 1440)
    
    def generate_chief_complaint(self, acuity: int) -> Tuple[str, str]:
        """
        Generate chief complaint based on acuity
        
        Returns:
            Tuple of (complaint_text, complaint_category)
        """
        # Higher acuity more likely to have serious complaints
        complaints = list(self.CHIEF_COMPLAINTS.keys())
        probs = list(self.CHIEF_COMPLAINTS.values())
        
        # Adjust probabilities based on acuity
        if acuity <= 2:
            # Increase probability of serious complaints
            serious = ["chest_pain", "shortness_of_breath", "altered_mental_status"]
            probs = [p * 2 if c in serious else p * 0.7 for c, p in zip(complaints, probs)]
        
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        category = np.random.choice(complaints, p=probs)
        
        # Generate realistic complaint text
        complaint_texts = {
            "chest_pain": ["Chest pain", "Chest tightness", "Chest pressure"],
            "abdominal_pain": ["Abdominal pain", "Stomach pain", "Belly pain"],
            "shortness_of_breath": ["Shortness of breath", "Difficulty breathing", "SOB"],
            "headache": ["Headache", "Head pain", "Migraine"],
            "back_pain": ["Back pain", "Lower back pain", "Spine pain"],
            "fever": ["Fever", "High temperature", "Feeling hot"],
            "fall_injury": ["Fall", "Fell down", "Injury from fall"],
            "laceration": ["Laceration", "Cut", "Wound"],
            "nausea_vomiting": ["Nausea", "Vomiting", "N/V"],
            "dizziness": ["Dizziness", "Lightheaded", "Vertigo"],
            "weakness": ["Weakness", "Fatigue", "Feeling weak"],
            "altered_mental_status": ["Confusion", "Altered mental status", "Disoriented"],
            "cough": ["Cough", "Coughing", "Productive cough"],
            "general_pain": ["Pain", "Hurting", "Discomfort"],
            "injury": ["Injury", "Hurt", "Accident"]
        }
        
        text = random.choice(complaint_texts.get(category, ["Unknown"]))
        return text, category
    
    def generate_vital_signs(self, acuity: int, age: int) -> VitalSigns:
        """Generate realistic vital signs based on acuity and age"""
        # Base values
        base_hr = 75
        base_rr = 16
        base_sbp = 120
        base_dbp = 80
        base_temp = 37.0
        base_o2 = 98
        
        # Adjust for age
        age_factor = (age - 50) / 50  # Normalized age factor
        base_hr += int(age_factor * 5)
        base_sbp += int(age_factor * 10)
        
        # Adjust for acuity (more abnormal for higher acuity)
        acuity_noise = (6 - acuity) * 5  # More noise for higher acuity
        
        return VitalSigns(
            timestamp=datetime.now(),
            temperature=round(base_temp + np.random.normal(0, 0.3 * (6 - acuity) / 5), 1),
            heart_rate=int(base_hr + np.random.normal(0, acuity_noise)),
            respiratory_rate=int(base_rr + np.random.normal(0, 2 * (6 - acuity) / 5)),
            blood_pressure_systolic=int(base_sbp + np.random.normal(0, acuity_noise)),
            blood_pressure_diastolic=int(base_dbp + np.random.normal(0, acuity_noise / 2)),
            oxygen_saturation=round(min(100, base_o2 - abs(np.random.normal(0, (6 - acuity)))), 1),
            pain_score=min(10, max(0, int(np.random.normal(5, 2))))
        )
    
    def generate_disposition(self, acuity: int) -> Disposition:
        """Generate disposition based on acuity"""
        dist = self.DISPOSITION_BY_ACUITY.get(acuity, self.DISPOSITION_BY_ACUITY[3])
        
        dispositions = list(dist.keys())
        probs = list(dist.values())
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p / total for p in probs]
        
        selected = np.random.choice(dispositions, p=probs)
        
        disposition_map = {
            "admitted": Disposition.ADMITTED,
            "discharged": Disposition.DISCHARGED,
            "transferred": Disposition.TRANSFERRED,
            "expired": Disposition.EXPIRED,
            "lwbs": Disposition.LWBS,
            "ama": Disposition.AMA
        }
        
        return disposition_map.get(selected, Disposition.DISCHARGED)
    
    def generate_patient_journey(self, arrival_time: datetime) -> PatientJourney:
        """
        Generate a complete patient journey
        
        Args:
            arrival_time: Patient arrival timestamp
            
        Returns:
            PatientJourney object with all events
        """
        # Generate patient characteristics
        acuity = self.generate_acuity()
        age_mean, age_std = self.AGE_BY_ACUITY[acuity]
        age = int(np.clip(np.random.normal(age_mean, age_std), 18, 100))
        
        chief_complaint, complaint_category = self.generate_chief_complaint(acuity)
        disposition = self.generate_disposition(acuity)
        los_minutes = self.generate_los(acuity)
        
        # Handle LWBS - shorter LOS
        if disposition == Disposition.LWBS:
            los_minutes = min(los_minutes, np.random.uniform(30, 120))
        
        departure_time = arrival_time + timedelta(minutes=los_minutes)
        
        # Create journey
        journey = PatientJourney(
            arrival_time=arrival_time,
            departure_time=departure_time,
            acuity=acuity,
            age=age,
            gender=random.choice(["M", "F"]),
            chief_complaint=chief_complaint,
            chief_complaint_category=complaint_category,
            disposition=disposition,
            total_los_minutes=los_minutes
        )
        
        # Generate triage assessment
        journey.triage_assessment = TriageAssessment(
            acuity=AcuityLevel(acuity),
            chief_complaint=chief_complaint,
            chief_complaint_category=complaint_category,
            vital_signs=self.generate_vital_signs(acuity, age),
            arrival_mode=np.random.choice(
                ["ambulatory", "ambulance", "helicopter"],
                p=[0.75, 0.23, 0.02] if acuity > 2 else [0.3, 0.65, 0.05]
            )
        )
        
        # Generate events
        events = self._generate_events(journey)
        journey.events = events
        journey.compute_metrics()
        
        return journey
    
    def _generate_events(self, journey: PatientJourney) -> List[PatientEvent]:
        """Generate event timeline for a patient journey"""
        events = []
        current_time = journey.arrival_time
        
        # 1. Arrival
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.ARRIVAL,
            timestamp=current_time,
            location=Location.WAITING_ROOM
        ))
        
        # 2. Triage (5-15 minutes after arrival)
        triage_delay = np.random.uniform(5, 15)
        current_time += timedelta(minutes=triage_delay)
        triage_duration = np.random.uniform(5, 10)
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.TRIAGE,
            timestamp=current_time,
            location=Location.TRIAGE,
            duration_minutes=triage_duration
        ))
        current_time += timedelta(minutes=triage_duration)
        
        # Handle LWBS - patient leaves without being seen
        if journey.disposition == Disposition.LWBS:
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.LWBS,
                timestamp=journey.departure_time,
                location=Location.WAITING_ROOM
            ))
            return events
        
        # 3. Bed assignment (depends on acuity)
        bed_wait_by_acuity = {1: (0, 5), 2: (5, 20), 3: (15, 60), 4: (30, 90), 5: (45, 120)}
        wait_range = bed_wait_by_acuity[journey.acuity]
        bed_wait = np.random.uniform(wait_range[0], wait_range[1])
        current_time += timedelta(minutes=bed_wait)
        
        # Determine location based on acuity
        if journey.acuity == 1:
            location = Location.RESUSCITATION
        elif journey.acuity >= 4:
            location = Location.FAST_TRACK
        else:
            location = Location.MAIN_ED
        
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.BED_ASSIGNMENT,
            timestamp=current_time,
            location=location
        ))
        
        # 4. Physician evaluation (10-30 min after bed)
        physician_delay = np.random.uniform(10, 30)
        current_time += timedelta(minutes=physician_delay)
        physician_duration = np.random.uniform(15, 45)
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=EventType.PHYSICIAN_EVAL,
            timestamp=current_time,
            location=location,
            duration_minutes=physician_duration
        ))
        current_time += timedelta(minutes=physician_duration)
        
        # 5. Labs (50% probability for ESI 1-3)
        needs_labs = (journey.acuity <= 3 and random.random() < 0.7) or random.random() < 0.3
        if needs_labs and current_time < journey.departure_time:
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.LAB_ORDER,
                timestamp=current_time,
                location=location
            ))
            
            lab_turnaround = np.random.uniform(30, 90)
            lab_result_time = current_time + timedelta(minutes=lab_turnaround)
            if lab_result_time < journey.departure_time:
                events.append(PatientEvent(
                    patient_id=journey.patient_id,
                    stay_id=journey.stay_id,
                    event_type=EventType.LAB_RESULT,
                    timestamp=lab_result_time,
                    location=Location.LAB,
                    duration_minutes=lab_turnaround
                ))
        
        # 6. Imaging (40% probability for ESI 1-3)
        needs_imaging = (journey.acuity <= 3 and random.random() < 0.5) or random.random() < 0.2
        if needs_imaging and current_time < journey.departure_time:
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.IMAGING_ORDER,
                timestamp=current_time,
                location=location
            ))
            
            imaging_turnaround = np.random.uniform(45, 120)
            imaging_result_time = current_time + timedelta(minutes=imaging_turnaround)
            if imaging_result_time < journey.departure_time:
                events.append(PatientEvent(
                    patient_id=journey.patient_id,
                    stay_id=journey.stay_id,
                    event_type=EventType.IMAGING_RESULT,
                    timestamp=imaging_result_time,
                    location=Location.IMAGING,
                    duration_minutes=imaging_turnaround
                ))
        
        # 7. Treatment
        if current_time + timedelta(minutes=30) < journey.departure_time:
            treatment_start = current_time + timedelta(minutes=np.random.uniform(10, 30))
            treatment_duration = np.random.uniform(20, 60)
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.TREATMENT,
                timestamp=treatment_start,
                location=location,
                duration_minutes=treatment_duration
            ))
        
        # 8. Disposition decision
        disposition_time = journey.departure_time - timedelta(minutes=np.random.uniform(15, 30))
        if disposition_time > current_time:
            events.append(PatientEvent(
                patient_id=journey.patient_id,
                stay_id=journey.stay_id,
                event_type=EventType.DISPOSITION,
                timestamp=disposition_time,
                location=location
            ))
        
        # 9. Final event (discharge or admission)
        final_event_type = (
            EventType.ADMISSION if journey.disposition == Disposition.ADMITTED
            else EventType.DISCHARGE
        )
        events.append(PatientEvent(
            patient_id=journey.patient_id,
            stay_id=journey.stay_id,
            event_type=final_event_type,
            timestamp=journey.departure_time,
            location=location
        ))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def generate_all_journeys(self) -> List[PatientJourney]:
        """Generate all patient journeys based on configuration"""
        arrivals = self.generate_arrivals()
        
        # Limit to configured number of patients
        arrivals = arrivals[:self.config.num_patients]
        
        self._journeys = []
        for i, arrival in enumerate(arrivals):
            if (i + 1) % 5000 == 0:
                logger.info(f"Generated {i + 1}/{len(arrivals)} journeys")
            
            journey = self.generate_patient_journey(arrival)
            self._journeys.append(journey)
        
        logger.info(f"Generated {len(self._journeys)} complete patient journeys")
        return self._journeys
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert journeys to pandas DataFrame"""
        if not self._journeys:
            self.generate_all_journeys()
        
        records = []
        for j in self._journeys:
            records.append({
                "stay_id": j.stay_id,
                "patient_id": j.patient_id,
                "arrival_time": j.arrival_time,
                "departure_time": j.departure_time,
                "acuity": j.acuity,
                "age": j.age,
                "gender": j.gender,
                "chief_complaint": j.chief_complaint,
                "chief_complaint_category": j.chief_complaint_category,
                "disposition": j.disposition.value if j.disposition else None,
                "total_los_minutes": j.total_los_minutes,
                "wait_to_triage_minutes": j.wait_to_triage_minutes,
                "wait_to_bed_minutes": j.wait_to_bed_minutes,
                "wait_to_physician_minutes": j.wait_to_physician_minutes,
                "num_events": len(j.events)
            })
        
        return pd.DataFrame(records)
    
    def get_hourly_metrics(self) -> pd.DataFrame:
        """Calculate hourly aggregated metrics"""
        if self._hourly_df is not None:
            return self._hourly_df
        
        df = self.to_dataframe()
        
        # Create hourly buckets
        df["arrival_hour"] = df["arrival_time"].dt.floor("H")
        df["departure_hour"] = df["departure_time"].dt.floor("H")
        
        # Arrivals per hour
        arrivals = df.groupby("arrival_hour").agg(
            arrivals=("stay_id", "count"),
            mean_los=("total_los_minutes", "mean"),
            median_los=("total_los_minutes", "median"),
            p90_los=("total_los_minutes", lambda x: x.quantile(0.9)),
            mean_wait=("wait_to_bed_minutes", "mean"),
            median_wait=("wait_to_bed_minutes", "median")
        ).reset_index()
        
        arrivals.columns = [
            "timestamp", "arrivals", "mean_los_minutes", "median_los_minutes",
            "p90_los_minutes", "mean_wait_minutes", "median_wait_minutes"
        ]
        
        # Add temporal features
        arrivals["hour"] = arrivals["timestamp"].dt.hour
        arrivals["day_of_week"] = arrivals["timestamp"].dt.dayofweek
        arrivals["is_weekend"] = arrivals["day_of_week"].isin([5, 6]).astype(int)
        arrivals["month"] = arrivals["timestamp"].dt.month
        
        # Calculate LWBS rate
        lwbs_hourly = df[df["disposition"] == "left_without_being_seen"].groupby(
            "arrival_hour"
        ).size().reset_index(name="lwbs_count")
        
        arrivals = arrivals.merge(
            lwbs_hourly, left_on="timestamp", right_on="arrival_hour", how="left"
        ).drop(columns=["arrival_hour"], errors="ignore")
        
        arrivals["lwbs_count"] = arrivals["lwbs_count"].fillna(0)
        arrivals["lwbs_rate"] = arrivals["lwbs_count"] / arrivals["arrivals"]
        
        self._hourly_df = arrivals
        return arrivals
    
    def get_queue_evolution_data(self) -> pd.DataFrame:
        """
        Generate queue evolution data for 3D visualization
        
        Returns:
            DataFrame with hour, queue_length, mean_wait columns
        """
        df = self.to_dataframe()
        
        # Simulate queue states
        records = []
        
        for hour in range(24):
            hour_data = df[df["arrival_time"].dt.hour == hour]
            
            for queue_length in range(0, 51, 5):
                # Simulate wait time based on queue length
                base_wait = 10 + queue_length * 3
                actual_wait = hour_data["wait_to_bed_minutes"].mean() if len(hour_data) > 0 else base_wait
                
                # Adjust for queue length
                wait_multiplier = 1 + (queue_length / 50) * 0.5
                mean_wait = actual_wait * wait_multiplier + np.random.normal(0, 5)
                
                records.append({
                    "hour": hour,
                    "queue_length": queue_length,
                    "mean_wait": max(5, mean_wait)
                })
        
        return pd.DataFrame(records)
