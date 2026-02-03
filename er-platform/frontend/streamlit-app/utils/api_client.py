"""
Enhanced API Client for ER Patient Flow Intelligence Platform
Supports both REST API calls and demo data generation

Authors: Neel, Harsh, Tanishk
"""

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import streamlit as st
import os


class APIClient:
    """
    HTTP client for backend services with fallback to demo data
    
    Handles:
    - Connection to backend microservices
    - Graceful fallback to synthetic data when services unavailable
    - Caching for performance
    """
    
    def __init__(self):
        self.timeout = 30.0
        
        # Service URLs from environment or defaults
        self.data_service_url = os.getenv("DATA_SERVICE_URL", "http://localhost:8001")
        self.ml_service_url = os.getenv("ML_SERVICE_URL", "http://localhost:8002")
        self.sim_service_url = os.getenv("SIM_SERVICE_URL", "http://localhost:8003")
        self.analytics_service_url = os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8004")
        
        # Track service status
        self.services_available = {
            'data': None,
            'ml': None,
            'sim': None,
            'analytics': None
        }
    
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError:
            return None
        except httpx.HTTPStatusError:
            return None
        except Exception:
            return None
    
    def get(self, url: str, **kwargs) -> Optional[Dict]:
        return self._make_request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs) -> Optional[Dict]:
        return self._make_request("POST", url, **kwargs)
    
    def check_services(self) -> Dict[str, bool]:
        """Check which backend services are available"""
        self.services_available['data'] = self.get(f"{self.data_service_url}/health") is not None
        self.services_available['ml'] = self.get(f"{self.ml_service_url}/health") is not None
        self.services_available['sim'] = self.get(f"{self.sim_service_url}/health") is not None
        self.services_available['analytics'] = self.get(f"{self.analytics_service_url}/health") is not None
        return self.services_available
    
    # =========================================================================
    # DATA SERVICE METHODS
    # =========================================================================
    
    def get_current_state(self) -> Dict:
        """Get current ED state"""
        result = self.get(f"{self.data_service_url}/current-state")
        if result:
            return result
        
        # Generate demo data
        return self._generate_demo_state()
    
    def get_hourly_metrics(self, hours: int = 168) -> pd.DataFrame:
        """Get hourly metrics for the past N hours"""
        result = self.get(f"{self.data_service_url}/hourly-metrics", params={"hours": hours})
        if result and 'metrics' in result:
            return pd.DataFrame(result['metrics'])
        
        # Generate demo data
        return self._generate_demo_hourly_metrics(hours)
    
    def get_patients(self, limit: int = 100) -> pd.DataFrame:
        """Get patient list"""
        result = self.get(f"{self.data_service_url}/patients", params={"limit": limit})
        if result and 'patients' in result:
            return pd.DataFrame(result['patients'])
        
        return self._generate_demo_patients(limit)
    
    def upload_data(self, file, filename: str) -> Dict:
        """Upload file to data service"""
        try:
            files = {'file': (filename, file)}
            result = self.post(f"{self.data_service_url}/upload", files=files)
            if result:
                return result
        except:
            pass
        
        return {
            'success': False,
            'message': 'Data service not available. Try using synthetic data instead.',
            'row_count': 0
        }
    
    # =========================================================================
    # ML SERVICE METHODS
    # =========================================================================
    
    def predict_los(self, patient_data: Dict) -> Dict:
        """Predict length of stay for a patient"""
        result = self.post(f"{self.ml_service_url}/predict/los", json=patient_data)
        if result:
            return result
        
        # Demo prediction
        acuity = patient_data.get('acuity', 3)
        base_los = {1: 480, 2: 300, 3: 180, 4: 90, 5: 45}
        predicted = base_los.get(acuity, 180) + np.random.normal(0, 30)
        
        return {
            'predicted_los_minutes': max(15, predicted),
            'confidence_interval': [max(10, predicted * 0.7), predicted * 1.4],
            'model': 'demo'
        }
    
    def predict_arrivals(self, hours: int = 24) -> pd.DataFrame:
        """Get arrival forecast"""
        result = self.get(f"{self.ml_service_url}/predict/arrivals", params={"hours": hours})
        if result and 'forecast' in result:
            return pd.DataFrame(result['forecast'])
        
        return self._generate_demo_forecast(hours)
    
    # =========================================================================
    # SIMULATION SERVICE METHODS
    # =========================================================================
    
    def run_simulation(self, config: Dict) -> Dict:
        """Run ED simulation"""
        result = self.post(f"{self.sim_service_url}/simulate", json=config)
        if result:
            return result
        
        return self._generate_demo_simulation_results()
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run what-if scenario"""
        result = self.post(f"{self.sim_service_url}/scenario", json=scenario)
        if result:
            return result
        
        return self._generate_demo_scenario_results(scenario)
    
    # =========================================================================
    # ANALYTICS SERVICE METHODS
    # =========================================================================
    
    def detect_anomalies(self, data: Dict) -> Dict:
        """Detect anomalies in data"""
        result = self.post(f"{self.analytics_service_url}/anomaly/detect", json=data)
        if result:
            return result
        
        return {'anomalies': [], 'score': 0.0}
    
    def run_bias_audit(self, data: Dict) -> Dict:
        """Run bias/fairness audit"""
        result = self.post(f"{self.analytics_service_url}/bias/audit", json=data)
        if result:
            return result
        
        return {'metrics': {}, 'recommendations': []}
    
    def get_fairness_scorecard(self, data: Dict) -> Dict:
        """Get comprehensive fairness scorecard"""
        result = self.post(f"{self.analytics_service_url}/fairness/scorecard", json=data)
        if result:
            return result
        
        return self._generate_demo_fairness_scorecard()
    
    def evaluate_classification(self, data: Dict) -> Dict:
        """Evaluate classification model"""
        result = self.post(f"{self.analytics_service_url}/evaluate/classification", json=data)
        if result:
            return result
        
        return self._generate_demo_classification_eval()
    
    def evaluate_regression(self, data: Dict) -> Dict:
        """Evaluate regression model"""
        result = self.post(f"{self.analytics_service_url}/evaluate/regression", json=data)
        if result:
            return result
        
        return self._generate_demo_regression_eval()
    
    def compare_models(self, models_data: List[Dict]) -> Dict:
        """Compare multiple model evaluations"""
        result = self.post(f"{self.analytics_service_url}/evaluate/compare", json=models_data)
        if result:
            return result
        
        return self._generate_demo_model_comparison()
    
    def _generate_demo_fairness_scorecard(self) -> Dict:
        """Generate demo fairness scorecard"""
        return {
            'model_name': 'XGBoost LOS Predictor',
            'timestamp': datetime.now().isoformat(),
            'overall_fairness_score': 82.5,
            'grade': 'B',
            'disparities': [
                {'attribute': 'age_group', 'metric': 'accuracy', 'reference_group': '41-64',
                 'comparison_group': '65+', 'ratio': 0.92, 'is_significant': False},
                {'attribute': 'insurance', 'metric': 'tpr', 'reference_group': 'Private',
                 'comparison_group': 'Medicaid', 'ratio': 0.78, 'is_significant': True}
            ],
            'recommendations': [
                'Review model performance for Medicaid patients',
                'Consider reweighting training data for elderly patients'
            ],
            'risk_areas': ['Insurance disparity in TPR']
        }
    
    def _generate_demo_classification_eval(self) -> Dict:
        """Generate demo classification evaluation"""
        return {
            'model_name': 'Admission Predictor',
            'task_type': 'classification',
            'sample_size': 1000,
            'metrics': {
                'accuracy': 0.847,
                'precision': 0.823,
                'recall': 0.789,
                'f1_score': 0.806,
                'specificity': 0.891,
                'roc_auc': 0.891
            },
            'confusion_matrix': {
                'matrix': [[612, 88], [65, 235]],
                'labels': ['Negative', 'Positive'],
                'true_positives': 235,
                'true_negatives': 612,
                'false_positives': 88,
                'false_negatives': 65
            }
        }
    
    def _generate_demo_regression_eval(self) -> Dict:
        """Generate demo regression evaluation"""
        return {
            'model_name': 'LOS Predictor',
            'task_type': 'regression',
            'sample_size': 1000,
            'metrics': {
                'mae': 15.3,
                'rmse': 22.7,
                'r2': 0.847,
                'mape': 12.4
            },
            'error_distribution': {
                'p10': 3.2,
                'p25': 7.8,
                'p50': 12.1,
                'p75': 19.5,
                'p90': 31.2
            }
        }
    
    def _generate_demo_model_comparison(self) -> Dict:
        """Generate demo model comparison"""
        return {
            'models': [
                {'name': 'XGBoost', 'accuracy': 0.847, 'f1': 0.806, 'auc': 0.891},
                {'name': 'Random Forest', 'accuracy': 0.823, 'f1': 0.778, 'auc': 0.867},
                {'name': 'LightGBM', 'accuracy': 0.852, 'f1': 0.816, 'auc': 0.897},
                {'name': 'Neural Network', 'accuracy': 0.831, 'f1': 0.795, 'auc': 0.876},
                {'name': 'Logistic Regression', 'accuracy': 0.789, 'f1': 0.739, 'auc': 0.834}
            ],
            'best_model': 'LightGBM',
            'recommendation': 'LightGBM shows best overall performance with highest AUC'
        }
    
    # =========================================================================
    # VISUALIZATION DATA METHODS
    # =========================================================================
    
    def get_queue_evolution_data(self) -> pd.DataFrame:
        """Get data for queue evolution surface"""
        result = self.get(f"{self.data_service_url}/queue-evolution")
        if result and 'data' in result:
            return pd.DataFrame(result['data'])
        
        return self._generate_demo_queue_data()
    
    def get_patient_flow_data(self) -> pd.DataFrame:
        """Get data for Sankey diagram"""
        result = self.get(f"{self.data_service_url}/patient-flow")
        if result and 'flows' in result:
            return pd.DataFrame(result['flows'])
        
        return self._generate_demo_flow_data()
    
    def get_historical_arrivals(self, hours: int = 48) -> pd.DataFrame:
        """Get historical arrival data"""
        result = self.get(f"{self.data_service_url}/arrivals/history", params={"hours": hours})
        if result and 'arrivals' in result:
            return pd.DataFrame(result['arrivals'])
        
        return self._generate_demo_historical_arrivals(hours)
    
    def get_arrival_forecast(self, hours: int = 24) -> pd.DataFrame:
        """Get arrival forecast with confidence intervals"""
        return self.predict_arrivals(hours)
    
    def get_congestion_history(self, weeks: int = 4) -> pd.DataFrame:
        """Get congestion history for heatmap"""
        result = self.get(f"{self.data_service_url}/congestion/history", params={"weeks": weeks})
        if result and 'data' in result:
            return pd.DataFrame(result['data'])
        
        return self._generate_demo_congestion_data(weeks)
    
    def get_current_patients(self) -> pd.DataFrame:
        """Get list of current patients in ED"""
        return self.get_patients(limit=50)
    
    def get_recent_arrivals(self, hours: int = 1) -> pd.DataFrame:
        """Get recent patient arrivals"""
        result = self.get(f"{self.data_service_url}/arrivals/recent", params={"hours": hours})
        if result and 'arrivals' in result:
            return pd.DataFrame(result['arrivals'])
        
        return self._generate_demo_recent_arrivals(hours)
    
    def get_active_alerts(self) -> List[Dict]:
        """Get active system alerts"""
        result = self.get(f"{self.data_service_url}/alerts/active")
        if result and 'alerts' in result:
            return result['alerts']
        
        return self._generate_demo_alerts()
    
    # =========================================================================
    # DEMO DATA GENERATORS
    # =========================================================================
    
    def _generate_demo_state(self) -> Dict:
        """Generate demo ED state"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        patients = np.random.randint(18, 32)
        beds_total = 25
        beds_occupied = min(beds_total, np.random.randint(15, 24))
        
        return {
            'total_patients': patients,
            'patient_delta': np.random.randint(-3, 4),
            'beds_total': beds_total,
            'beds_occupied': beds_occupied,
            'beds_available': beds_total - beds_occupied,
            'bed_utilization': beds_occupied / beds_total,
            'bed_util_delta': np.random.uniform(-0.05, 0.05),
            'waiting_room': np.random.randint(3, 12),
            'triage_queue': np.random.randint(0, 5),
            'triage_delta': np.random.randint(-2, 3),
            'mean_wait_time': np.random.uniform(25, 65),
            'wait_delta': np.random.uniform(-10, 10),
            'forecast_4h': np.random.randint(60, 90),
            'forecast_uncertainty': np.random.randint(8, 15),
            'lwbs_rate': np.random.uniform(0.02, 0.06),
            'lwbs_delta': np.random.uniform(-0.01, 0.01),
            'physician_utilization': np.random.uniform(0.65, 0.92),
            'nurse_utilization': np.random.uniform(0.70, 0.95)
        }
    
    def _generate_demo_hourly_metrics(self, hours: int) -> pd.DataFrame:
        """Generate demo hourly metrics"""
        np.random.seed(42)
        
        now = datetime.now()
        times = [now - timedelta(hours=i) for i in range(hours, 0, -1)]
        
        data = []
        for t in times:
            hour = t.hour
            base_arrivals = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            
            data.append({
                'timestamp': t.isoformat(),
                'arrivals': max(2, int(base_arrivals + np.random.normal(0, 3))),
                'departures': max(1, int(base_arrivals * 0.9 + np.random.normal(0, 2))),
                'mean_wait_time': max(10, 30 + base_arrivals * 2 + np.random.normal(0, 8)),
                'mean_los': max(60, 180 + np.random.normal(0, 30)),
                'bed_utilization': min(1, max(0.4, 0.7 + base_arrivals * 0.01 + np.random.normal(0, 0.05))),
                'census': max(10, int(20 + base_arrivals * 0.5 + np.random.normal(0, 3)))
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_patients(self, limit: int) -> pd.DataFrame:
        """Generate demo patient list"""
        np.random.seed(42)
        
        data = []
        now = datetime.now()
        
        for i in range(min(limit, 50)):
            acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.50, 0.28, 0.05])
            arrival = now - timedelta(minutes=np.random.randint(30, 480))
            
            statuses = ['Waiting', 'Triage', 'Bed Assigned', 'With Physician', 'Testing', 'Treatment']
            
            data.append({
                'patient_id': f'PAT_{100000 + i:06d}',
                'arrival_time': arrival.isoformat(),
                'acuity': acuity,
                'age': int(np.clip(np.random.normal(45, 20), 1, 95)),
                'chief_complaint': np.random.choice([
                    'Chest Pain', 'Abdominal Pain', 'Shortness of Breath',
                    'Trauma', 'Headache', 'Fever', 'Back Pain'
                ]),
                'status': np.random.choice(statuses, p=[0.15, 0.05, 0.20, 0.25, 0.20, 0.15]),
                'wait_time_minutes': int((now - arrival).total_seconds() / 60)
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_forecast(self, hours: int) -> pd.DataFrame:
        """Generate demo arrival forecast"""
        np.random.seed(42)
        
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [now + timedelta(hours=i) for i in range(1, hours + 1)]
        
        data = []
        for t in times:
            hour = t.hour
            base = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            
            data.append({
                'timestamp': t.isoformat(),
                'forecast': base,
                'lower_95': base * 0.6,
                'upper_95': base * 1.5,
                'lower_80': base * 0.75,
                'upper_80': base * 1.35,
                'lower_50': base * 0.85,
                'upper_50': base * 1.2
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_queue_data(self) -> pd.DataFrame:
        """Generate demo queue evolution data"""
        np.random.seed(42)
        
        data = []
        for _ in range(500):
            hour = np.random.uniform(0, 24)
            queue = max(2, 10 + 8 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 3))
            wait = max(5, 15 + 2 * queue + 0.15 * queue**2 + np.random.normal(0, 8))
            
            data.append({
                'hour': hour,
                'queue_length': queue,
                'mean_wait_time': wait
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_flow_data(self) -> pd.DataFrame:
        """Generate demo patient flow data"""
        np.random.seed(42)
        
        transitions = [
            ('Arrival', 'Triage', 100, 8),
            ('Triage', 'Waiting Room', 65, 5),
            ('Triage', 'Bed Assignment', 35, 3),
            ('Waiting Room', 'Bed Assignment', 60, 35),
            ('Waiting Room', 'LWBS', 5, 45),
            ('Bed Assignment', 'Physician Eval', 95, 15),
            ('Physician Eval', 'Testing/Labs', 70, 12),
            ('Physician Eval', 'Treatment', 25, 8),
            ('Testing/Labs', 'Treatment', 70, 45),
            ('Treatment', 'Discharge', 75, 25),
            ('Treatment', 'Admission', 18, 60),
            ('Treatment', 'Transfer', 2, 40),
        ]
        
        data = []
        for source, target, value, avg_time in transitions:
            data.append({
                'source': source,
                'target': target,
                'value': value + np.random.randint(-5, 10),
                'avg_time': avg_time + np.random.uniform(-5, 10)
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_historical_arrivals(self, hours: int) -> pd.DataFrame:
        """Generate demo historical arrivals"""
        np.random.seed(42)
        
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [now - timedelta(hours=i) for i in range(hours, 0, -1)]
        
        data = []
        for t in times:
            hour = t.hour
            base = 15 + 10 * np.sin(np.pi * (hour - 6) / 12)
            actual = max(2, base + np.random.normal(0, 4))
            
            data.append({
                'timestamp': t,
                'actual_arrivals': actual
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_congestion_data(self, weeks: int) -> pd.DataFrame:
        """Generate demo congestion history"""
        np.random.seed(42)
        
        now = datetime.now()
        start = now - timedelta(weeks=weeks)
        times = pd.date_range(start=start, end=now, freq='H')
        
        data = []
        for t in times:
            hour = t.hour
            dow = t.dayofweek
            
            base = 30 + 35 * np.sin(np.pi * (hour - 6) / 12)
            if dow in [5, 6]:
                base *= 1.1
            if dow == 0:
                base *= 1.15
            
            score = max(5, min(95, base + np.random.normal(0, 10)))
            
            data.append({
                'timestamp': t,
                'congestion_score': score
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_recent_arrivals(self, hours: int) -> pd.DataFrame:
        """Generate demo recent arrivals"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        data = []
        now = datetime.now()
        
        n_arrivals = np.random.randint(5, 15)
        
        for i in range(n_arrivals):
            arrival = now - timedelta(minutes=np.random.randint(5, 60))
            acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.50, 0.28, 0.05])
            
            data.append({
                'patient_id': f'PAT_{np.random.randint(100000, 999999)}',
                'arrival_time': arrival.isoformat(),
                'acuity': acuity,
                'age': int(np.clip(np.random.normal(45, 20), 1, 95)),
                'chief_complaint': np.random.choice([
                    'Chest Pain', 'Abdominal Pain', 'Shortness of Breath',
                    'Trauma', 'Headache', 'Fever'
                ]),
                'status': 'Arrived'
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_alerts(self) -> List[Dict]:
        """Generate demo alerts"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        alerts = []
        
        # Random chance of various alerts
        if np.random.random() < 0.3:
            alerts.append({
                'severity': 'warning',
                'type': 'capacity',
                'message': f'Bed utilization at {np.random.randint(82, 92)}%. Monitor closely.'
            })
        
        if np.random.random() < 0.2:
            alerts.append({
                'severity': 'info',
                'type': 'forecast',
                'message': f'Arrival surge expected in next 2 hours ({np.random.randint(22, 28)} patients/hr forecast)'
            })
        
        if np.random.random() < 0.15:
            alerts.append({
                'severity': 'warning',
                'type': 'wait_time',
                'message': f'Average wait time ({np.random.randint(55, 75)} min) approaching threshold'
            })
        
        return alerts
    
    def _generate_demo_simulation_results(self) -> Dict:
        """Generate demo simulation results"""
        return {
            'summary': {
                'total_patients': np.random.randint(200, 300),
                'mean_wait_time': np.random.uniform(30, 60),
                'mean_los': np.random.uniform(150, 240),
                'bed_utilization': np.random.uniform(0.70, 0.90),
                'lwbs_rate': np.random.uniform(0.02, 0.05)
            },
            'hourly_metrics': [],
            'bottlenecks': ['Triage', 'Bed Assignment'],
            'recommendations': [
                'Consider adding staff during peak hours (10am-6pm)',
                'Fast-track protocol for ESI 4-5 patients could reduce wait times by 15%'
            ]
        }
    
    def _generate_demo_scenario_results(self, scenario: Dict) -> Dict:
        """Generate demo scenario comparison results"""
        return {
            'baseline': {
                'mean_wait_time': 45,
                'mean_los': 200,
                'lwbs_rate': 0.04
            },
            'scenario': {
                'mean_wait_time': 45 * (1 - scenario.get('improvement', 0.1)),
                'mean_los': 200 * (1 - scenario.get('improvement', 0.05)),
                'lwbs_rate': 0.04 * (1 - scenario.get('improvement', 0.15))
            },
            'improvement': {
                'wait_time_reduction': scenario.get('improvement', 0.1) * 100,
                'los_reduction': scenario.get('improvement', 0.05) * 100,
                'lwbs_reduction': scenario.get('improvement', 0.15) * 100
            }
        }
