"""
Enhanced Real-Time Streaming Service with SSE (Server-Sent Events)
WebSocket and SSE endpoints for live ED monitoring

Authors: Neel, Harsh, Tanishk
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger


class StreamEventType(str, Enum):
    """Types of real-time events"""
    NEW_ARRIVAL = "new_arrival"
    PATIENT_UPDATE = "patient_update"
    PATIENT_DEPARTURE = "patient_departure"
    TRIAGE_COMPLETE = "triage_complete"
    BED_ASSIGNMENT = "bed_assignment"
    SYSTEM_STATE = "system_state"
    ALERT = "alert"
    METRICS = "metrics"
    PREDICTION = "prediction"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamConfig:
    """Configuration for streaming behavior"""
    speed_factor: float = 1.0  # 1.0 = real-time, 10.0 = 10x faster
    include_predictions: bool = True
    include_alerts: bool = True
    alert_threshold_bed_util: float = 0.85
    alert_threshold_wait_time: float = 60
    heartbeat_interval: int = 30


class EnhancedRealtimeStreamer:
    """
    Enhanced Real-Time Data Streaming Service
    
    Features:
    - Server-Sent Events (SSE) support
    - Configurable simulation speed
    - Multiple event streams
    - Alert generation
    - Prediction integration
    """
    
    # Realistic hourly arrival rates
    HOURLY_RATES = {
        0: 8, 1: 6, 2: 5, 3: 4, 4: 4, 5: 5,
        6: 7, 7: 10, 8: 14, 9: 18, 10: 22, 11: 24,
        12: 23, 13: 22, 14: 21, 15: 20, 16: 21, 17: 23,
        18: 25, 19: 24, 20: 22, 21: 18, 22: 14, 23: 10
    }
    
    # Chief complaints by acuity
    COMPLAINTS = {
        1: ['Cardiac Arrest', 'Severe Trauma', 'Stroke', 'Respiratory Failure'],
        2: ['Chest Pain', 'Difficulty Breathing', 'Severe Pain', 'Altered Mental Status'],
        3: ['Abdominal Pain', 'Moderate Injury', 'High Fever', 'Vomiting'],
        4: ['Minor Injury', 'Mild Pain', 'Rash', 'Cold Symptoms'],
        5: ['Prescription Refill', 'Minor Cut', 'Suture Removal']
    }
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.is_streaming = False
        self.active_connections: List = []
        self.event_counter = 0
        self.current_state = self._initialize_state()
        self.patients_in_ed: List[Dict] = []
        self.event_history: List[Dict] = []
    
    def _initialize_state(self) -> Dict:
        """Initialize ED system state"""
        return {
            'timestamp': datetime.now().isoformat(),
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
            'departures_last_hour': np.random.randint(6, 15)
        }
    
    def get_current_arrival_rate(self) -> float:
        """Get arrival rate based on time of day"""
        hour = datetime.now().hour
        base_rate = self.HOURLY_RATES.get(hour, 15)
        
        # Weekend adjustment
        if datetime.now().weekday() >= 5:
            base_rate *= 1.15
        
        return base_rate
    
    def generate_patient(self) -> Dict:
        """Generate a synthetic patient arrival"""
        acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
        age = max(0, min(100, int(np.random.normal(45, 22))))
        
        complaint = np.random.choice(self.COMPLAINTS.get(acuity, ['General Complaint']))
        
        # Expected LOS by acuity
        los_means = {1: 480, 2: 360, 3: 240, 4: 120, 5: 60}
        expected_los = max(30, np.random.gamma(4, los_means[acuity] / 4))
        
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
            'status': 'Waiting',
            'predicted_admission_prob': round(
                {1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.02}[acuity] + 
                np.random.uniform(-0.1, 0.1), 2
            )
        }
    
    def update_state(self) -> Dict:
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
        
        # Update metrics
        state['mean_wait_time'] = max(5, min(120,
            state['mean_wait_time'] + np.random.uniform(-5, 5)))
        state['physician_utilization'] = max(0.3, min(1.0,
            state['physician_utilization'] + np.random.uniform(-0.05, 0.05)))
        state['nurse_utilization'] = max(0.4, min(1.0,
            state['nurse_utilization'] + np.random.uniform(-0.05, 0.05)))
        
        # Update counts
        state['arrivals_last_hour'] = max(0, int(
            self.get_current_arrival_rate() + np.random.normal(0, 3)
        ))
        state['departures_last_hour'] = max(0, np.random.randint(5, 15))
        state['lwbs_last_hour'] = 1 if np.random.random() < 0.1 else 0
        
        # Update acuity distribution
        total = state['patients_in_ed']
        state['acuity_distribution'] = {
            1: max(0, int(total * 0.02)),
            2: max(0, int(total * 0.15)),
            3: max(0, int(total * 0.50)),
            4: max(0, int(total * 0.28)),
            5: max(0, int(total * 0.05))
        }
        
        state['timestamp'] = datetime.now().isoformat()
        return state
    
    def check_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        state = self.current_state
        
        bed_util = state['beds_occupied'] / state['beds_total']
        
        if bed_util >= 0.95:
            alerts.append({
                'severity': 'critical',
                'type': 'capacity',
                'message': f'CRITICAL: Bed utilization at {bed_util:.0%}',
                'metric': 'bed_utilization',
                'value': round(bed_util, 2),
                'threshold': 0.95,
                'timestamp': datetime.now().isoformat()
            })
        elif bed_util >= self.config.alert_threshold_bed_util:
            alerts.append({
                'severity': 'warning',
                'type': 'capacity',
                'message': f'Warning: Bed utilization at {bed_util:.0%}',
                'metric': 'bed_utilization',
                'value': round(bed_util, 2),
                'threshold': self.config.alert_threshold_bed_util,
                'timestamp': datetime.now().isoformat()
            })
        
        if state['mean_wait_time'] >= 90:
            alerts.append({
                'severity': 'critical',
                'type': 'wait_time',
                'message': f'CRITICAL: Wait time {state["mean_wait_time"]:.0f} min',
                'metric': 'mean_wait_time',
                'value': round(state['mean_wait_time'], 1),
                'threshold': 90,
                'timestamp': datetime.now().isoformat()
            })
        elif state['mean_wait_time'] >= self.config.alert_threshold_wait_time:
            alerts.append({
                'severity': 'warning',
                'type': 'wait_time',
                'message': f'Warning: Wait time {state["mean_wait_time"]:.0f} min',
                'metric': 'mean_wait_time',
                'value': round(state['mean_wait_time'], 1),
                'threshold': self.config.alert_threshold_wait_time,
                'timestamp': datetime.now().isoformat()
            })
        
        if state['waiting_room'] >= 15:
            alerts.append({
                'severity': 'warning',
                'type': 'queue',
                'message': f'{state["waiting_room"]} patients in waiting room',
                'metric': 'waiting_room',
                'value': state['waiting_room'],
                'threshold': 15,
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def format_sse(self, event_type: str, data: Any) -> str:
        """Format data as Server-Sent Event"""
        event_data = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
        return f"event: {event_type}\ndata: {event_data}\n\n"
    
    async def stream_arrivals(self) -> AsyncGenerator[str, None]:
        """Stream patient arrivals as SSE"""
        self.is_streaming = True
        
        while self.is_streaming:
            # Calculate wait time based on arrival rate
            rate = self.get_current_arrival_rate()
            mean_interval = 3600 / rate / self.config.speed_factor
            wait_time = min(10 / self.config.speed_factor, np.random.exponential(mean_interval))
            
            await asyncio.sleep(wait_time)
            
            patient = self.generate_patient()
            self.patients_in_ed.append(patient)
            self.event_counter += 1
            
            event = {
                'event_id': f'evt_{self.event_counter:08d}',
                'timestamp': datetime.now().isoformat(),
                'patient': patient
            }
            
            yield self.format_sse(StreamEventType.NEW_ARRIVAL.value, event)
    
    async def stream_system_state(self, interval: int = 30) -> AsyncGenerator[str, None]:
        """Stream system state updates as SSE"""
        self.is_streaming = True
        
        while self.is_streaming:
            state = self.update_state()
            self.event_counter += 1
            
            event = {
                'event_id': f'evt_{self.event_counter:08d}',
                'state': state
            }
            
            yield self.format_sse(StreamEventType.SYSTEM_STATE.value, event)
            
            # Also stream alerts if enabled
            if self.config.include_alerts:
                alerts = self.check_alerts()
                for alert in alerts:
                    self.event_counter += 1
                    yield self.format_sse(StreamEventType.ALERT.value, {
                        'event_id': f'evt_{self.event_counter:08d}',
                        'alert': alert
                    })
            
            await asyncio.sleep(interval / self.config.speed_factor)
    
    async def stream_metrics(self, interval: int = 60) -> AsyncGenerator[str, None]:
        """Stream real-time metrics as SSE"""
        self.is_streaming = True
        history = []
        
        while self.is_streaming:
            state = self.current_state
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'bed_utilization': round(state['beds_occupied'] / state['beds_total'], 3),
                'queue_pressure': round(state['waiting_room'] / max(1, state['beds_available']), 2),
                'congestion_score': self._calculate_congestion(),
                'predicted_arrivals_1h': round(self.get_current_arrival_rate()),
                'predicted_arrivals_4h': round(self.get_current_arrival_rate() * 4),
                'throughput_rate': state['departures_last_hour'],
                'lwbs_rate': round(
                    state['lwbs_last_hour'] / max(1, state['arrivals_last_hour']), 3
                )
            }
            
            history.append(metrics)
            if len(history) > 60:
                history.pop(0)
            
            # Add trend
            if len(history) >= 5:
                recent = [h['congestion_score'] for h in history[-5:]]
                if recent[-1] > recent[0] * 1.1:
                    metrics['trend'] = 'increasing'
                elif recent[-1] < recent[0] * 0.9:
                    metrics['trend'] = 'decreasing'
                else:
                    metrics['trend'] = 'stable'
            else:
                metrics['trend'] = 'stable'
            
            self.event_counter += 1
            yield self.format_sse(StreamEventType.METRICS.value, {
                'event_id': f'evt_{self.event_counter:08d}',
                'metrics': metrics
            })
            
            await asyncio.sleep(interval / self.config.speed_factor)
    
    async def stream_combined(self, interval: int = 10) -> AsyncGenerator[str, None]:
        """Stream combined events (state + arrivals + metrics + alerts)"""
        self.is_streaming = True
        last_arrival = datetime.now()
        last_metrics = datetime.now()
        arrival_interval = 60 / self.get_current_arrival_rate()
        
        while self.is_streaming:
            now = datetime.now()
            events = []
            
            # Check for arrival
            if (now - last_arrival).total_seconds() >= arrival_interval / self.config.speed_factor:
                patient = self.generate_patient()
                self.patients_in_ed.append(patient)
                self.event_counter += 1
                events.append({
                    'type': StreamEventType.NEW_ARRIVAL.value,
                    'event_id': f'evt_{self.event_counter:08d}',
                    'data': patient
                })
                last_arrival = now
                arrival_interval = 60 / self.get_current_arrival_rate()
            
            # Update state
            state = self.update_state()
            self.event_counter += 1
            events.append({
                'type': StreamEventType.SYSTEM_STATE.value,
                'event_id': f'evt_{self.event_counter:08d}',
                'data': state
            })
            
            # Check alerts
            if self.config.include_alerts:
                alerts = self.check_alerts()
                for alert in alerts:
                    self.event_counter += 1
                    events.append({
                        'type': StreamEventType.ALERT.value,
                        'event_id': f'evt_{self.event_counter:08d}',
                        'data': alert
                    })
            
            # Metrics every 60 seconds
            if (now - last_metrics).total_seconds() >= 60 / self.config.speed_factor:
                self.event_counter += 1
                events.append({
                    'type': StreamEventType.METRICS.value,
                    'event_id': f'evt_{self.event_counter:08d}',
                    'data': {
                        'timestamp': now.isoformat(),
                        'congestion_score': self._calculate_congestion(),
                        'bed_utilization': round(
                            state['beds_occupied'] / state['beds_total'], 3
                        ),
                        'mean_wait_time': round(state['mean_wait_time'], 1)
                    }
                })
                last_metrics = now
            
            # Yield combined event
            yield self.format_sse('combined', {
                'timestamp': now.isoformat(),
                'events': events
            })
            
            await asyncio.sleep(interval / self.config.speed_factor)
    
    async def stream_heartbeat(self) -> AsyncGenerator[str, None]:
        """Stream heartbeat for connection keep-alive"""
        while self.is_streaming:
            yield self.format_sse(StreamEventType.HEARTBEAT.value, {
                'timestamp': datetime.now().isoformat(),
                'status': 'connected'
            })
            await asyncio.sleep(self.config.heartbeat_interval)
    
    def _calculate_congestion(self) -> float:
        """Calculate congestion score (0-100)"""
        state = self.current_state
        bed_util = state['beds_occupied'] / state['beds_total']
        wait_factor = min(1, state['mean_wait_time'] / 60)
        queue_factor = min(1, state['waiting_room'] / 15)
        
        return round(bed_util * 40 + wait_factor * 35 + queue_factor * 25, 1)
    
    def get_snapshot(self) -> Dict:
        """Get current state snapshot"""
        return {
            'timestamp': datetime.now().isoformat(),
            'state': self.update_state(),
            'alerts': self.check_alerts(),
            'congestion_score': self._calculate_congestion(),
            'patients_in_ed_count': len(self.patients_in_ed),
            'recent_events_count': self.event_counter
        }
    
    def stop(self):
        """Stop streaming"""
        self.is_streaming = False
        logger.info("Streaming stopped")
    
    def reset(self):
        """Reset streamer state"""
        self.current_state = self._initialize_state()
        self.patients_in_ed = []
        self.event_counter = 0
        self.event_history = []
        logger.info("Streamer reset")


# Singleton instance
enhanced_streamer = EnhancedRealtimeStreamer()
