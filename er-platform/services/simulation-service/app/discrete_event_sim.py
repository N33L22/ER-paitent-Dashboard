"""
SimPy Discrete Event Simulation for Emergency Department
Agent-based modeling of patient flow through ED stages
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator, Any
from enum import Enum
import numpy as np
from loguru import logger

try:
    import simpy
    SIMPY_AVAILABLE = True
except ImportError:
    SIMPY_AVAILABLE = False
    logger.warning("SimPy not available")


class PatientState(str, Enum):
    """Patient states in the ED journey"""
    ARRIVED = "arrived"
    IN_TRIAGE = "in_triage"
    WAITING_FOR_BED = "waiting_for_bed"
    IN_BED = "in_bed"
    WITH_PHYSICIAN = "with_physician"
    AWAITING_LABS = "awaiting_labs"
    AWAITING_IMAGING = "awaiting_imaging"
    IN_TREATMENT = "in_treatment"
    AWAITING_DISPOSITION = "awaiting_disposition"
    DISCHARGED = "discharged"
    ADMITTED = "admitted"
    LWBS = "lwbs"


@dataclass
class SimulationConfig:
    """Configuration for ED simulation"""
    # Resources
    num_triage_nurses: int = 2
    num_physicians: int = 4
    num_nurses: int = 8
    num_beds: int = 20
    num_lab_techs: int = 2
    num_imaging_techs: int = 2
    
    # Simulation parameters
    simulation_hours: int = 168  # 1 week
    arrival_rate_per_hour: float = 20.0
    
    # Arrival multiplier (for scenarios)
    arrival_multiplier: float = 1.0
    
    # Random seed
    seed: int = 42
    
    # LWBS thresholds (minutes)
    lwbs_threshold_esi4: int = 120
    lwbs_threshold_esi5: int = 90
    
    # Acuity distribution
    acuity_distribution: Dict[int, float] = field(default_factory=lambda: {
        1: 0.01, 2: 0.15, 3: 0.50, 4: 0.30, 5: 0.04
    })


@dataclass
class SimulatedPatient:
    """A patient in the simulation"""
    patient_id: int
    acuity: int
    arrival_time: float
    departure_time: Optional[float] = None
    state: PatientState = PatientState.ARRIVED
    
    # Timestamps
    triage_start: Optional[float] = None
    triage_end: Optional[float] = None
    bed_assignment_time: Optional[float] = None
    physician_start: Optional[float] = None
    physician_end: Optional[float] = None
    discharge_time: Optional[float] = None
    
    # Wait times
    wait_for_triage: float = 0
    wait_for_bed: float = 0
    wait_for_physician: float = 0
    
    # Outcome
    disposition: str = "discharged"
    los_minutes: float = 0
    lwbs: bool = False
    
    # Tracking
    events: List[Dict] = field(default_factory=list)
    
    def log_event(self, time: float, event: str, details: str = ""):
        """Log an event in the patient's journey"""
        self.events.append({
            "time": time,
            "event": event,
            "details": details
        })


@dataclass
class SimulationResults:
    """Results from a simulation run"""
    # Configuration
    config: SimulationConfig
    
    # Patients
    total_patients: int = 0
    completed_patients: int = 0
    lwbs_count: int = 0
    
    # LOS statistics
    mean_los_minutes: float = 0
    median_los_minutes: float = 0
    p90_los_minutes: float = 0
    std_los_minutes: float = 0
    
    # Wait times
    mean_wait_to_triage: float = 0
    mean_wait_to_bed: float = 0
    mean_wait_to_physician: float = 0
    
    # Utilization
    bed_utilization: float = 0
    physician_utilization: float = 0
    nurse_utilization: float = 0
    
    # By acuity
    los_by_acuity: Dict[int, float] = field(default_factory=dict)
    count_by_acuity: Dict[int, int] = field(default_factory=dict)
    
    # Time series
    hourly_census: List[int] = field(default_factory=list)
    hourly_arrivals: List[int] = field(default_factory=list)
    hourly_discharges: List[int] = field(default_factory=list)
    
    # All patients for analysis
    patients: List[SimulatedPatient] = field(default_factory=list)


class EDSimulation:
    """
    Discrete Event Simulation of Emergency Department
    
    Models:
    - Patient arrivals (non-homogeneous Poisson)
    - Triage process
    - Bed assignment (priority queue by acuity)
    - Physician evaluation
    - Lab/Imaging orders
    - Treatment
    - Disposition and discharge
    """
    
    # Service time distributions (mean, std in minutes)
    SERVICE_TIMES = {
        'triage': (7, 2),
        'registration': (5, 1),
        'physician_eval': (20, 10),
        'lab_turnaround': (45, 15),
        'imaging_turnaround': (60, 20),
        'treatment': (45, 20)
    }
    
    # Probability of needing labs/imaging by acuity
    LAB_PROBABILITY = {1: 0.95, 2: 0.85, 3: 0.60, 4: 0.30, 5: 0.10}
    IMAGING_PROBABILITY = {1: 0.80, 2: 0.65, 3: 0.40, 4: 0.20, 5: 0.05}
    
    # Admission probability by acuity
    ADMISSION_PROBABILITY = {1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.02}
    
    # Hourly arrival multipliers
    HOURLY_PATTERNS = {
        0: 0.5, 1: 0.4, 2: 0.35, 3: 0.3, 4: 0.3, 5: 0.35,
        6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.15,
        12: 1.2, 13: 1.15, 14: 1.1, 15: 1.1, 16: 1.15, 17: 1.2,
        18: 1.25, 19: 1.2, 20: 1.1, 21: 1.0, 22: 0.8, 23: 0.65
    }
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize ED simulation
        
        Args:
            config: Simulation configuration
        """
        if not SIMPY_AVAILABLE:
            raise ImportError("SimPy is required for simulation")
        
        self.config = config or SimulationConfig()
        self.env: Optional[simpy.Environment] = None
        self.results: Optional[SimulationResults] = None
        
        # Resources (will be created during run)
        self.triage_nurses: Optional[simpy.PriorityResource] = None
        self.physicians: Optional[simpy.PriorityResource] = None
        self.nurses: Optional[simpy.Resource] = None
        self.beds: Optional[simpy.PriorityResource] = None
        self.lab_techs: Optional[simpy.Resource] = None
        self.imaging_techs: Optional[simpy.Resource] = None
        
        # Patient tracking
        self.patients: List[SimulatedPatient] = []
        self.current_census: int = 0
        self.patient_counter: int = 0
        
        # Hourly tracking
        self.hourly_arrivals: Dict[int, int] = {}
        self.hourly_discharges: Dict[int, int] = {}
        self.hourly_census: Dict[int, int] = {}
        
        # Set random seed
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
    
    def _generate_acuity(self) -> int:
        """Generate patient acuity based on distribution"""
        levels = list(self.config.acuity_distribution.keys())
        probs = list(self.config.acuity_distribution.values())
        return np.random.choice(levels, p=probs)
    
    def _get_arrival_rate(self, time: float) -> float:
        """Get arrival rate at given time"""
        hour = int(time / 60) % 24
        multiplier = self.HOURLY_PATTERNS.get(hour, 1.0)
        base_rate = self.config.arrival_rate_per_hour * self.config.arrival_multiplier
        return base_rate * multiplier
    
    def _service_time(self, service: str, acuity: int = 3) -> float:
        """Generate service time for a given service"""
        mean, std = self.SERVICE_TIMES.get(service, (15, 5))
        
        # Adjust for acuity (higher acuity = longer times)
        acuity_factor = 1 + (3 - acuity) * 0.2
        
        time = max(1, np.random.normal(mean * acuity_factor, std))
        return time
    
    def arrival_process(self, env: simpy.Environment) -> Generator:
        """Generate patient arrivals"""
        while True:
            # Get current arrival rate
            rate = self._get_arrival_rate(env.now)
            
            # Inter-arrival time (exponential)
            inter_arrival = np.random.exponential(60 / rate)
            yield env.timeout(inter_arrival)
            
            # Create new patient
            self.patient_counter += 1
            acuity = self._generate_acuity()
            
            patient = SimulatedPatient(
                patient_id=self.patient_counter,
                acuity=acuity,
                arrival_time=env.now
            )
            
            patient.log_event(env.now, "arrival", f"Acuity: {acuity}")
            
            # Track hourly arrivals
            hour = int(env.now / 60)
            self.hourly_arrivals[hour] = self.hourly_arrivals.get(hour, 0) + 1
            
            # Start patient journey
            env.process(self.patient_journey(env, patient))
    
    def patient_journey(
        self,
        env: simpy.Environment,
        patient: SimulatedPatient
    ) -> Generator:
        """Complete patient journey through ED"""
        self.current_census += 1
        
        # Priority based on acuity (lower = higher priority)
        priority = patient.acuity
        
        # =========================================================
        # TRIAGE
        # =========================================================
        triage_request_time = env.now
        
        with self.triage_nurses.request(priority=priority) as req:
            yield req
            
            patient.wait_for_triage = env.now - triage_request_time
            patient.triage_start = env.now
            patient.state = PatientState.IN_TRIAGE
            patient.log_event(env.now, "triage_start", f"Wait: {patient.wait_for_triage:.1f} min")
            
            # Triage time
            triage_duration = self._service_time('triage', patient.acuity)
            yield env.timeout(triage_duration)
            
            patient.triage_end = env.now
            patient.log_event(env.now, "triage_end", f"Duration: {triage_duration:.1f} min")
        
        # =========================================================
        # WAIT FOR BED (check for LWBS)
        # =========================================================
        patient.state = PatientState.WAITING_FOR_BED
        bed_request_time = env.now
        
        # LWBS check for low acuity
        lwbs_threshold = None
        if patient.acuity == 4:
            lwbs_threshold = self.config.lwbs_threshold_esi4
        elif patient.acuity == 5:
            lwbs_threshold = self.config.lwbs_threshold_esi5
        
        with self.beds.request(priority=priority) as bed_req:
            if lwbs_threshold:
                # Wait with timeout for LWBS
                result = yield bed_req | env.timeout(lwbs_threshold)
                
                if bed_req not in result:
                    # Patient left without being seen
                    patient.lwbs = True
                    patient.state = PatientState.LWBS
                    patient.disposition = "lwbs"
                    patient.departure_time = env.now
                    patient.los_minutes = env.now - patient.arrival_time
                    patient.log_event(env.now, "lwbs", f"Wait exceeded {lwbs_threshold} min")
                    
                    self.current_census -= 1
                    self.patients.append(patient)
                    return
            else:
                yield bed_req
            
            patient.wait_for_bed = env.now - bed_request_time
            patient.bed_assignment_time = env.now
            patient.state = PatientState.IN_BED
            patient.log_event(env.now, "bed_assigned", f"Wait: {patient.wait_for_bed:.1f} min")
            
            # =========================================================
            # PHYSICIAN EVALUATION
            # =========================================================
            physician_request_time = env.now
            
            with self.physicians.request(priority=priority) as phys_req:
                yield phys_req
                
                patient.wait_for_physician = env.now - physician_request_time
                patient.physician_start = env.now
                patient.state = PatientState.WITH_PHYSICIAN
                patient.log_event(env.now, "physician_start", f"Wait: {patient.wait_for_physician:.1f} min")
                
                # Physician evaluation time
                eval_duration = self._service_time('physician_eval', patient.acuity)
                yield env.timeout(eval_duration)
                
                patient.physician_end = env.now
                patient.log_event(env.now, "physician_end", f"Duration: {eval_duration:.1f} min")
            
            # =========================================================
            # LABS (if needed)
            # =========================================================
            if random.random() < self.LAB_PROBABILITY[patient.acuity]:
                patient.state = PatientState.AWAITING_LABS
                patient.log_event(env.now, "lab_ordered", "")
                
                with self.lab_techs.request() as lab_req:
                    yield lab_req
                    
                    lab_time = self._service_time('lab_turnaround', patient.acuity)
                    yield env.timeout(lab_time)
                    
                    patient.log_event(env.now, "lab_result", f"Turnaround: {lab_time:.1f} min")
            
            # =========================================================
            # IMAGING (if needed)
            # =========================================================
            if random.random() < self.IMAGING_PROBABILITY[patient.acuity]:
                patient.state = PatientState.AWAITING_IMAGING
                patient.log_event(env.now, "imaging_ordered", "")
                
                with self.imaging_techs.request() as img_req:
                    yield img_req
                    
                    imaging_time = self._service_time('imaging_turnaround', patient.acuity)
                    yield env.timeout(imaging_time)
                    
                    patient.log_event(env.now, "imaging_result", f"Turnaround: {imaging_time:.1f} min")
            
            # =========================================================
            # TREATMENT
            # =========================================================
            patient.state = PatientState.IN_TREATMENT
            treatment_time = self._service_time('treatment', patient.acuity)
            yield env.timeout(treatment_time)
            patient.log_event(env.now, "treatment_complete", f"Duration: {treatment_time:.1f} min")
            
            # =========================================================
            # DISPOSITION
            # =========================================================
            patient.state = PatientState.AWAITING_DISPOSITION
            
            # Determine disposition
            if random.random() < self.ADMISSION_PROBABILITY[patient.acuity]:
                patient.disposition = "admitted"
                patient.state = PatientState.ADMITTED
                # Admission process takes additional time
                yield env.timeout(np.random.uniform(30, 60))
            else:
                patient.disposition = "discharged"
                patient.state = PatientState.DISCHARGED
                # Discharge paperwork
                yield env.timeout(np.random.uniform(10, 20))
        
        # =========================================================
        # DEPARTURE
        # =========================================================
        patient.departure_time = env.now
        patient.los_minutes = env.now - patient.arrival_time
        patient.log_event(env.now, "departure", f"LOS: {patient.los_minutes:.1f} min, Disposition: {patient.disposition}")
        
        # Track hourly discharges
        hour = int(env.now / 60)
        self.hourly_discharges[hour] = self.hourly_discharges.get(hour, 0) + 1
        
        self.current_census -= 1
        self.patients.append(patient)
    
    def census_tracker(self, env: simpy.Environment) -> Generator:
        """Track census every hour"""
        while True:
            hour = int(env.now / 60)
            self.hourly_census[hour] = self.current_census
            yield env.timeout(60)  # Check every hour
    
    def run(self) -> SimulationResults:
        """
        Run the simulation
        
        Returns:
            SimulationResults with all metrics
        """
        logger.info(f"Starting simulation: {self.config.simulation_hours} hours, "
                   f"{self.config.num_beds} beds, {self.config.num_physicians} physicians")
        
        # Reset state
        self.patients = []
        self.current_census = 0
        self.patient_counter = 0
        self.hourly_arrivals = {}
        self.hourly_discharges = {}
        self.hourly_census = {}
        
        # Create environment
        self.env = simpy.Environment()
        
        # Create resources
        self.triage_nurses = simpy.PriorityResource(self.env, capacity=self.config.num_triage_nurses)
        self.physicians = simpy.PriorityResource(self.env, capacity=self.config.num_physicians)
        self.nurses = simpy.Resource(self.env, capacity=self.config.num_nurses)
        self.beds = simpy.PriorityResource(self.env, capacity=self.config.num_beds)
        self.lab_techs = simpy.Resource(self.env, capacity=self.config.num_lab_techs)
        self.imaging_techs = simpy.Resource(self.env, capacity=self.config.num_imaging_techs)
        
        # Start processes
        self.env.process(self.arrival_process(self.env))
        self.env.process(self.census_tracker(self.env))
        
        # Run simulation
        self.env.run(until=self.config.simulation_hours * 60)  # Convert hours to minutes
        
        # Compile results
        self.results = self._compile_results()
        
        logger.info(f"Simulation complete. {self.results.total_patients} patients, "
                   f"Mean LOS: {self.results.mean_los_minutes:.1f} min")
        
        return self.results
    
    def _compile_results(self) -> SimulationResults:
        """Compile simulation results"""
        results = SimulationResults(config=self.config)
        
        # Filter completed patients
        completed = [p for p in self.patients if not p.lwbs]
        lwbs = [p for p in self.patients if p.lwbs]
        
        results.total_patients = len(self.patients)
        results.completed_patients = len(completed)
        results.lwbs_count = len(lwbs)
        
        if completed:
            los_values = [p.los_minutes for p in completed]
            results.mean_los_minutes = float(np.mean(los_values))
            results.median_los_minutes = float(np.median(los_values))
            results.p90_los_minutes = float(np.percentile(los_values, 90))
            results.std_los_minutes = float(np.std(los_values))
            
            # Wait times
            results.mean_wait_to_triage = float(np.mean([p.wait_for_triage for p in completed]))
            results.mean_wait_to_bed = float(np.mean([p.wait_for_bed for p in completed]))
            results.mean_wait_to_physician = float(np.mean([p.wait_for_physician for p in completed]))
            
            # By acuity
            for acuity in [1, 2, 3, 4, 5]:
                acuity_patients = [p for p in completed if p.acuity == acuity]
                if acuity_patients:
                    results.los_by_acuity[acuity] = float(np.mean([p.los_minutes for p in acuity_patients]))
                    results.count_by_acuity[acuity] = len(acuity_patients)
        
        # Utilization (simplified estimate)
        total_minutes = self.config.simulation_hours * 60
        if results.completed_patients > 0:
            avg_los = results.mean_los_minutes
            results.bed_utilization = min(1.0, (results.completed_patients * avg_los) / 
                                         (self.config.num_beds * total_minutes))
        
        # Time series
        results.hourly_arrivals = [self.hourly_arrivals.get(h, 0) 
                                   for h in range(self.config.simulation_hours)]
        results.hourly_discharges = [self.hourly_discharges.get(h, 0) 
                                     for h in range(self.config.simulation_hours)]
        results.hourly_census = [self.hourly_census.get(h, 0) 
                                 for h in range(self.config.simulation_hours)]
        
        results.patients = self.patients
        
        return results
