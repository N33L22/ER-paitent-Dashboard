"""
Real-Time Data Streaming for ER Patient Flow Intelligence Platform
Supports WebSocket, Server-Sent Events, and Polling

Authors: Neel, Harsh, Tanishk
"""

import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger


class StreamEventType(Enum):
    """Types of real-time events"""
    NEW_ARRIVAL = "new_arrival"
    PATIENT_DEPARTURE = "patient_departure"
    TRIAGE_COMPLETE = "triage_complete"
    BED_ASSIGNMENT = "bed_assignment"
    SYSTEM_STATE_UPDATE = "system_state_update"
    ALERT = "alert"
    METRICS_UPDATE = "metrics_update"


@dataclass
class StreamEvent:
    """Standard stream event format"""
    event_type: str
    timestamp: str
    data: Dict
    event_id: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    def to_sse(self) -> str:
        """Format for Server-Sent Events"""
        return f"event: {self.event_type}\ndata: {json.dumps(self.data)}\n\n"


class RealtimeDataStreamer:
    """
    Real-time data streaming for live ED monitoring
    
    Supports:
    1. WebSocket connections
    2. Server-Sent Events (SSE)
    3. Polling endpoints
    4. Simulated real-time for demos
    """
    
    def __init__(self):
        self.active_connections: List = []
        self.simulation_mode: bool = True
        self.is_streaming: bool = False
        self.current_state: Dict = self._initialize_state()
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.patients_in_ed: List[Dict] = []
        self._event_counter: int = 0
        
    def _initialize_state(self) -> Dict:
        """Initialize ED system state"""
        return {
            'patients_in_ed': np.random.randint(18, 28),
            'beds_total': 25,
            'beds_occupied': np.random.randint(15, 22),
            'beds_available': 0,
            'waiting_room': np.random.randint(3, 10),
            'triage_queue': np.random.randint(0, 4),
            'mean_wait_time': np.random.uniform(25, 60),
            'physician_utilization': np.random.uniform(0.65, 0.90),
            'nurse_utilization': np.random.uniform(0.70, 0.95),
            'acuity_distribution': {1: 0, 2: 3, 3: 12, 4: 8, 5: 2},
            'lwbs_last_hour': np.random.randint(0, 2),
            'arrivals_last_hour': np.random.randint(8, 18),
            'discharges_last_hour': np.random.randint(6, 15)
        }
    
    def get_time_varying_rate(self) -> float:
        """
        Get arrival rate based on time of day (patients per hour)
        """
        hour = datetime.now().hour
        
        # Realistic ED arrival pattern
        hourly_rates = {
            0: 8, 1: 6, 2: 5, 3: 4, 4: 4, 5: 5,
            6: 7, 7: 10, 8: 14, 9: 18, 10: 22, 11: 24,
            12: 23, 13: 22, 14: 21, 15: 20, 16: 21, 17: 23,
            18: 25, 19: 24, 20: 22, 21: 18, 22: 14, 23: 10
        }
        
        base_rate = hourly_rates.get(hour, 15)
        
        # Add day-of-week variation
        dow = datetime.now().weekday()
        if dow in [5, 6]:  # Weekend
            base_rate *= 1.15
        if dow == 0:  # Monday
            base_rate *= 1.10
            
        return base_rate
    
    def generate_patient_arrival(self) -> Dict:
        """Generate synthetic patient arrival"""
        acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
        age = max(0, min(100, int(np.random.normal(45, 22))))
        
        chief_complaints = {
            1: ['Cardiac Arrest', 'Severe Trauma', 'Stroke Symptoms', 'Anaphylaxis'],
            2: ['Chest Pain', 'Difficulty Breathing', 'Altered Mental Status', 'Severe Bleeding'],
            3: ['Abdominal Pain', 'Moderate Trauma', 'High Fever', 'Persistent Vomiting'],
            4: ['Minor Injury', 'Mild Pain', 'Rash', 'Cold Symptoms'],
            5: ['Prescription Refill', 'Minor Abrasion', 'Suture Removal']
        }
        
        complaint = np.random.choice(chief_complaints.get(acuity, ['General Complaint']))
        
        # Estimate LOS based on acuity
        los_means = {1: 480, 2: 360, 3: 240, 4: 120, 5: 60}
        expected_los = np.random.gamma(4, los_means.get(acuity, 180) / 4)
        
        return {
            'patient_id': f'PAT_{np.random.randint(100000, 999999)}',
            'arrival_time': datetime.now().isoformat(),
            'acuity': int(acuity),
            'age': age,
            'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
            'chief_complaint': complaint,
            'expected_los_minutes': round(expected_los, 1),
            'mode_of_arrival': np.random.choice(
                ['Walk-in', 'Ambulance', 'Transfer', 'Police'],
                p=[0.65, 0.28, 0.05, 0.02]
            ),
            'status': 'Waiting'
        }
    
    def update_system_state(self) -> Dict:
        """Update and return current system state"""
        state = self.current_state
        
        # Random walk for counts
        state['patients_in_ed'] = max(5, min(40, 
            state['patients_in_ed'] + np.random.randint(-2, 3)))
        state['beds_occupied'] = max(0, min(state['beds_total'],
            state['beds_occupied'] + np.random.randint(-1, 2)))
        state['beds_available'] = state['beds_total'] - state['beds_occupied']
        state['waiting_room'] = max(0, min(20,
            state['waiting_room'] + np.random.randint(-1, 2)))
        state['triage_queue'] = max(0, min(8,
            state['triage_queue'] + np.random.randint(-1, 2)))
        
        # Update rates
        state['mean_wait_time'] = max(5, min(120,
            state['mean_wait_time'] + np.random.uniform(-5, 5)))
        state['physician_utilization'] = max(0.3, min(1.0,
            state['physician_utilization'] + np.random.uniform(-0.05, 0.05)))
        state['nurse_utilization'] = max(0.4, min(1.0,
            state['nurse_utilization'] + np.random.uniform(-0.05, 0.05)))
        
        # Update counts
        state['arrivals_last_hour'] = max(0, int(self.get_time_varying_rate() + np.random.normal(0, 3)))
        state['discharges_last_hour'] = max(0, np.random.randint(5, 15))
        state['lwbs_last_hour'] = 1 if np.random.random() < 0.1 else 0
        
        # Update acuity distribution
        total_patients = state['patients_in_ed']
        state['acuity_distribution'] = {
            1: max(0, int(total_patients * 0.02)),
            2: max(0, int(total_patients * 0.15)),
            3: max(0, int(total_patients * 0.50)),
            4: max(0, int(total_patients * 0.28)),
            5: max(0, int(total_patients * 0.05))
        }
        
        return state
    
    async def stream_patient_arrivals(self, speed_factor: float = 1.0) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream patient arrivals in real-time
        
        Args:
            speed_factor: 1.0 = real-time, 10.0 = 10x faster for demos
        """
        self.is_streaming = True
        
        while self.is_streaming:
            # Calculate wait time based on arrival rate
            arrival_rate = self.get_time_varying_rate()
            mean_interval = 3600 / arrival_rate / speed_factor  # seconds
            wait_time = np.random.exponential(mean_interval)
            
            # Cap wait time for responsive demos
            wait_time = min(wait_time, 10 / speed_factor)
            
            await asyncio.sleep(wait_time)
            
            # Generate arrival
            patient = self.generate_patient_arrival()
            self.patients_in_ed.append(patient)
            self._event_counter += 1
            
            yield StreamEvent(
                event_type=StreamEventType.NEW_ARRIVAL.value,
                timestamp=datetime.now().isoformat(),
                data=patient,
                event_id=f"evt_{self._event_counter:08d}"
            )
    
    async def stream_system_state(self, interval_seconds: int = 30) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream current ED system state at regular intervals
        """
        self.is_streaming = True
        
        while self.is_streaming:
            state = self.update_system_state()
            self._event_counter += 1
            
            yield StreamEvent(
                event_type=StreamEventType.SYSTEM_STATE_UPDATE.value,
                timestamp=datetime.now().isoformat(),
                data=state,
                event_id=f"evt_{self._event_counter:08d}"
            )
            
            await asyncio.sleep(interval_seconds)
    
    async def stream_metrics(self, interval_seconds: int = 60) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream real-time metrics updates
        """
        self.is_streaming = True
        history = []
        
        while self.is_streaming:
            state = self.current_state
            
            # Calculate derived metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'bed_utilization': state['beds_occupied'] / state['beds_total'],
                'queue_pressure': state['waiting_room'] / max(1, state['beds_available']),
                'congestion_score': self._calculate_congestion_score(state),
                'predicted_arrivals_1h': round(self.get_time_varying_rate()),
                'predicted_arrivals_4h': round(self.get_time_varying_rate() * 4),
                'throughput_rate': state['discharges_last_hour'],
                'lwbs_rate': state['lwbs_last_hour'] / max(1, state['arrivals_last_hour']),
                'trend': self._calculate_trend(history)
            }
            
            history.append(metrics.copy())
            if len(history) > 60:
                history.pop(0)
            
            self._event_counter += 1
            
            yield StreamEvent(
                event_type=StreamEventType.METRICS_UPDATE.value,
                timestamp=datetime.now().isoformat(),
                data=metrics,
                event_id=f"evt_{self._event_counter:08d}"
            )
            
            await asyncio.sleep(interval_seconds)
    
    async def stream_alerts(self) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream alerts based on threshold violations
        """
        self.is_streaming = True
        
        while self.is_streaming:
            alerts = self._check_alerts()
            
            for alert in alerts:
                self._event_counter += 1
                yield StreamEvent(
                    event_type=StreamEventType.ALERT.value,
                    timestamp=datetime.now().isoformat(),
                    data=alert,
                    event_id=f"evt_{self._event_counter:08d}"
                )
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    def _calculate_congestion_score(self, state: Dict) -> float:
        """Calculate overall congestion score (0-100)"""
        bed_util = state['beds_occupied'] / state['beds_total']
        wait_factor = min(1, state['mean_wait_time'] / 60)
        queue_factor = min(1, state['waiting_room'] / 15)
        
        score = (
            bed_util * 40 +
            wait_factor * 35 +
            queue_factor * 25
        )
        
        return round(score, 1)
    
    def _calculate_trend(self, history: List[Dict]) -> str:
        """Calculate trend from recent history"""
        if len(history) < 3:
            return 'stable'
        
        recent = [h['congestion_score'] for h in history[-5:]]
        
        if recent[-1] > recent[0] * 1.1:
            return 'increasing'
        elif recent[-1] < recent[0] * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        state = self.current_state
        
        # Bed capacity alert
        bed_util = state['beds_occupied'] / state['beds_total']
        if bed_util >= 0.95:
            alerts.append({
                'severity': 'critical',
                'type': 'capacity',
                'message': f'CRITICAL: Bed utilization at {bed_util:.0%}. Consider diversion.',
                'metric': 'bed_utilization',
                'value': bed_util
            })
        elif bed_util >= 0.85:
            alerts.append({
                'severity': 'warning',
                'type': 'capacity',
                'message': f'Warning: Bed utilization at {bed_util:.0%}. Monitor closely.',
                'metric': 'bed_utilization',
                'value': bed_util
            })
        
        # Wait time alert
        if state['mean_wait_time'] >= 90:
            alerts.append({
                'severity': 'critical',
                'type': 'wait_time',
                'message': f'CRITICAL: Average wait time {state["mean_wait_time"]:.0f} min exceeds 90 min threshold.',
                'metric': 'mean_wait_time',
                'value': state['mean_wait_time']
            })
        elif state['mean_wait_time'] >= 60:
            alerts.append({
                'severity': 'warning',
                'type': 'wait_time',
                'message': f'Warning: Average wait time {state["mean_wait_time"]:.0f} min approaching threshold.',
                'metric': 'mean_wait_time',
                'value': state['mean_wait_time']
            })
        
        # Queue depth alert
        if state['waiting_room'] >= 15:
            alerts.append({
                'severity': 'warning',
                'type': 'queue',
                'message': f'Warning: {state["waiting_room"]} patients in waiting room.',
                'metric': 'waiting_room',
                'value': state['waiting_room']
            })
        
        # High acuity alert
        high_acuity = state['acuity_distribution'].get(1, 0) + state['acuity_distribution'].get(2, 0)
        if high_acuity >= 5:
            alerts.append({
                'severity': 'info',
                'type': 'acuity',
                'message': f'Info: {high_acuity} high-acuity patients (ESI 1-2) currently in ED.',
                'metric': 'high_acuity_count',
                'value': high_acuity
            })
        
        return alerts
    
    def stop_streaming(self):
        """Stop all active streams"""
        self.is_streaming = False
        logger.info("Streaming stopped")
    
    def get_current_state_snapshot(self) -> Dict:
        """Get current state without streaming"""
        return {
            'timestamp': datetime.now().isoformat(),
            'state': self.update_system_state(),
            'alerts': self._check_alerts(),
            'patients_in_ed_count': len(self.patients_in_ed)
        }
    
    def simulate_historical_stream(
        self, 
        start_date: datetime,
        end_date: datetime,
        interval_minutes: int = 60
    ) -> List[Dict]:
        """
        Generate historical stream data for analysis
        
        Returns list of system states at each interval
        """
        states = []
        current = start_date
        
        while current <= end_date:
            # Adjust state based on time of day
            hour = current.hour
            dow = current.weekday()
            
            # Base patient count varies by time
            base_patients = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            if dow in [5, 6]:
                base_patients *= 1.15
            
            state = {
                'timestamp': current.isoformat(),
                'patients_in_ed': max(5, int(base_patients + np.random.normal(0, 3))),
                'beds_occupied': max(0, int(base_patients * 0.8 + np.random.normal(0, 2))),
                'waiting_room': max(0, int(base_patients * 0.3 + np.random.normal(0, 2))),
                'mean_wait_time': max(5, 30 + base_patients * 2 + np.random.normal(0, 10)),
                'arrivals': max(0, int(self.get_time_varying_rate() + np.random.normal(0, 2))),
                'congestion_score': 0
            }
            
            state['congestion_score'] = (
                (state['beds_occupied'] / 25) * 40 +
                min(1, state['mean_wait_time'] / 60) * 35 +
                min(1, state['waiting_room'] / 15) * 25
            )
            
            states.append(state)
            current += timedelta(minutes=interval_minutes)
        
        return states


# Singleton instance
streamer = RealtimeDataStreamer()
