"""
Synthetic Data CSV Generator for ER Patient Flow Intelligence Platform
Generates CSV files for testing model uploads and evaluation

Authors: Neel, Harsh, Tanishk
"""

import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger


class SyntheticCSVGenerator:
    """
    Generates realistic synthetic ER patient data CSV files for:
    1. Testing data upload functionality
    2. Model training and evaluation
    3. Demo and simulation purposes
    
    Generates multiple CSV file types:
    - Patient arrivals with demographics
    - Patient events/journey data
    - Hourly metrics for forecasting
    - Labeled data for ML evaluation
    """
    
    # Chief complaints with categories
    CHIEF_COMPLAINTS = {
        'chest_pain': ['Chest Pain', 'Chest Tightness', 'Chest Discomfort'],
        'abdominal_pain': ['Abdominal Pain', 'Stomach Ache', 'GI Pain'],
        'shortness_of_breath': ['Shortness of Breath', 'Difficulty Breathing', 'Dyspnea'],
        'headache': ['Headache', 'Migraine', 'Head Pain'],
        'back_pain': ['Back Pain', 'Lower Back Pain', 'Spine Pain'],
        'fever': ['Fever', 'High Temperature', 'Chills'],
        'fall_injury': ['Fall', 'Fall Injury', 'Mechanical Fall'],
        'laceration': ['Laceration', 'Cut', 'Wound'],
        'nausea_vomiting': ['Nausea', 'Vomiting', 'GI Distress'],
        'dizziness': ['Dizziness', 'Vertigo', 'Lightheaded'],
        'general_pain': ['General Pain', 'Body Ache', 'Generalized Pain'],
        'respiratory': ['Cough', 'Cold Symptoms', 'URI'],
        'cardiac': ['Palpitations', 'Irregular Heartbeat', 'Racing Heart'],
        'trauma': ['Injury', 'Trauma', 'Accident']
    }
    
    # Age distribution by acuity
    AGE_PARAMS = {
        1: (68, 15),  # Critical - older patients
        2: (60, 18),
        3: (48, 22),
        4: (38, 20),
        5: (32, 18)   # Non-urgent - younger
    }
    
    # LOS parameters by acuity (gamma distribution)
    LOS_PARAMS = {
        1: {'shape': 3.0, 'scale': 180},  # ~540 min mean
        2: {'shape': 2.5, 'scale': 120},  # ~300 min mean
        3: {'shape': 2.0, 'scale': 90},   # ~180 min mean
        4: {'shape': 1.8, 'scale': 50},   # ~90 min mean
        5: {'shape': 1.5, 'scale': 30}    # ~45 min mean
    }
    
    # Disposition probabilities by acuity
    DISPOSITION_PROBS = {
        1: {'admitted': 0.70, 'discharged': 0.15, 'transferred': 0.10, 'expired': 0.05},
        2: {'admitted': 0.45, 'discharged': 0.45, 'transferred': 0.08, 'expired': 0.02},
        3: {'admitted': 0.25, 'discharged': 0.70, 'transferred': 0.04, 'lwbs': 0.01},
        4: {'admitted': 0.08, 'discharged': 0.85, 'transferred': 0.02, 'lwbs': 0.05},
        5: {'admitted': 0.02, 'discharged': 0.88, 'transferred': 0.01, 'lwbs': 0.09}
    }
    
    # Hourly arrival rate multipliers
    HOURLY_PATTERN = {
        0: 0.5, 1: 0.4, 2: 0.35, 3: 0.3, 4: 0.3, 5: 0.35,
        6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.15,
        12: 1.2, 13: 1.15, 14: 1.1, 15: 1.1, 16: 1.15, 17: 1.2,
        18: 1.25, 19: 1.2, 20: 1.1, 21: 1.0, 22: 0.8, 23: 0.65
    }
    
    def __init__(self, output_dir: str = "./data/synthetic", seed: int = 42):
        """
        Initialize generator.
        
        Parameters
        ----------
        output_dir : str
            Directory to save generated CSV files
        seed : int
            Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"SyntheticCSVGenerator initialized. Output: {output_dir}")
    
    def generate_patient_arrivals(
        self,
        num_patients: int = 10000,
        start_date: Optional[datetime] = None,
        days: int = 90,
        filename: str = "patient_arrivals.csv"
    ) -> pd.DataFrame:
        """
        Generate patient arrival data with demographics.
        
        Parameters
        ----------
        num_patients : int
            Number of patients to generate
        start_date : datetime, optional
            Start date for data generation
        days : int
            Number of days to span
        filename : str
            Output filename
        
        Returns
        -------
        pd.DataFrame
            Generated patient data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        data = []
        
        for i in range(num_patients):
            # Generate arrival time
            day_offset = random.randint(0, days - 1)
            hour = self._weighted_random_hour()
            minute = random.randint(0, 59)
            arrival_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)
            
            # Generate acuity
            acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
            
            # Generate age based on acuity
            mean_age, std_age = self.AGE_PARAMS[acuity]
            age = max(0, min(100, int(np.random.normal(mean_age, std_age))))
            
            # Demographics
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            race = np.random.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                p=[0.58, 0.18, 0.14, 0.06, 0.04]
            )
            insurance = np.random.choice(
                ['Private', 'Medicare', 'Medicaid', 'Self-Pay', 'Other'],
                p=[0.40, 0.25, 0.20, 0.10, 0.05]
            )
            
            # Chief complaint
            category = np.random.choice(list(self.CHIEF_COMPLAINTS.keys()))
            complaint = random.choice(self.CHIEF_COMPLAINTS[category])
            
            # Mode of arrival
            mode = np.random.choice(
                ['Walk-in', 'Ambulance', 'Transfer', 'Police'],
                p=[0.65, 0.28, 0.05, 0.02]
            )
            
            # Generate LOS
            los_params = self.LOS_PARAMS[acuity]
            los_minutes = max(15, np.random.gamma(los_params['shape'], los_params['scale']))
            
            # Generate wait time
            base_wait = 30 if acuity >= 3 else 10
            wait_minutes = max(0, np.random.exponential(base_wait * (6 - acuity) / 3))
            
            # Disposition
            disp_probs = self.DISPOSITION_PROBS[acuity]
            disposition = np.random.choice(
                list(disp_probs.keys()),
                p=list(disp_probs.values())
            )
            
            data.append({
                'patient_id': f'PAT_{i+1:06d}',
                'stay_id': f'STAY_{i+1:06d}',
                'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'age': age,
                'gender': gender,
                'race': race,
                'insurance_type': insurance,
                'acuity': acuity,
                'chief_complaint': complaint,
                'chief_complaint_category': category,
                'mode_of_arrival': mode,
                'wait_time_minutes': round(wait_minutes, 1),
                'los_minutes': round(los_minutes, 1),
                'disposition': disposition,
                'hour': arrival_time.hour,
                'day_of_week': arrival_time.weekday(),
                'is_weekend': 1 if arrival_time.weekday() >= 5 else 0
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {len(df)} patient arrivals -> {filepath}")
        
        return df
    
    def generate_hourly_metrics(
        self,
        start_date: Optional[datetime] = None,
        days: int = 90,
        base_rate: float = 20.0,
        filename: str = "hourly_metrics.csv"
    ) -> pd.DataFrame:
        """
        Generate hourly ED metrics for time series forecasting.
        
        Parameters
        ----------
        start_date : datetime, optional
            Start date
        days : int
            Number of days
        base_rate : float
            Base hourly arrival rate
        filename : str
            Output filename
        
        Returns
        -------
        pd.DataFrame
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        data = []
        current = start_date
        
        for _ in range(days * 24):
            hour = current.hour
            dow = current.weekday()
            
            # Calculate arrival rate
            hour_mult = self.HOURLY_PATTERN[hour]
            dow_mult = 1.1 if dow in [5, 6] else 1.0
            rate = base_rate * hour_mult * dow_mult
            
            # Generate arrivals (Poisson)
            arrivals = max(0, int(np.random.poisson(rate)))
            
            # Generate correlated metrics
            census = max(5, int(20 + np.random.normal(arrivals * 0.8, 5)))
            beds_occupied = min(25, max(0, int(census * 0.85 + np.random.normal(0, 2))))
            wait_time = max(5, 20 + arrivals * 2 + np.random.normal(0, 10))
            departures = max(0, int(arrivals * 0.9 + np.random.normal(0, 3)))
            lwbs = 1 if np.random.random() < 0.05 else 0
            
            data.append({
                'timestamp': current.strftime('%Y-%m-%d %H:%M:%S'),
                'arrivals': arrivals,
                'departures': departures,
                'census': census,
                'beds_occupied': beds_occupied,
                'beds_available': 25 - beds_occupied,
                'mean_wait_time': round(wait_time, 1),
                'lwbs_count': lwbs,
                'hour': hour,
                'day_of_week': dow,
                'is_weekend': 1 if dow >= 5 else 0,
                'is_night': 1 if hour < 6 or hour >= 22 else 0
            })
            
            current += timedelta(hours=1)
        
        df = pd.DataFrame(data)
        
        # Add lag features
        for lag in [1, 24, 168]:
            df[f'arrivals_lag_{lag}h'] = df['arrivals'].shift(lag)
        
        # Add rolling features
        df['arrivals_rolling_24h'] = df['arrivals'].rolling(24).mean()
        df['arrivals_rolling_168h'] = df['arrivals'].rolling(168).mean()
        
        df = df.dropna()
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {len(df)} hourly records -> {filepath}")
        
        return df
    
    def generate_ml_evaluation_data(
        self,
        num_samples: int = 5000,
        task: str = 'los_prediction',
        filename: str = "ml_evaluation_data.csv"
    ) -> pd.DataFrame:
        """
        Generate labeled data for ML model evaluation.
        
        Includes:
        - Ground truth labels
        - Simulated model predictions
        - Protected attributes for fairness analysis
        
        Parameters
        ----------
        num_samples : int
            Number of samples
        task : str
            'los_prediction' or 'admission_prediction'
        filename : str
            Output filename
        
        Returns
        -------
        pd.DataFrame
        """
        data = []
        
        for i in range(num_samples):
            # Demographics
            age = max(0, min(100, int(np.random.normal(50, 22))))
            gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
            race = np.random.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                p=[0.58, 0.18, 0.14, 0.06, 0.04]
            )
            insurance = np.random.choice(
                ['Private', 'Medicare', 'Medicaid', 'Self-Pay'],
                p=[0.40, 0.28, 0.22, 0.10]
            )
            
            # Clinical
            acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
            category = np.random.choice(list(self.CHIEF_COMPLAINTS.keys()))
            
            # Age group
            if age < 18:
                age_group = 'pediatric'
            elif age < 40:
                age_group = 'young_adult'
            elif age < 65:
                age_group = 'middle_age'
            else:
                age_group = 'senior'
            
            if task == 'los_prediction':
                # Generate true LOS
                los_params = self.LOS_PARAMS[acuity]
                true_los = max(15, np.random.gamma(los_params['shape'], los_params['scale']))
                
                # Generate predicted LOS (with realistic error)
                error_std = true_los * 0.25  # 25% relative error
                predicted_los = max(15, true_los + np.random.normal(0, error_std))
                
                # Add systematic bias for demonstration
                # Slight underestimation for elderly
                if age >= 65:
                    predicted_los *= 0.92
                
                data.append({
                    'sample_id': i + 1,
                    'age': age,
                    'age_group': age_group,
                    'gender': gender,
                    'race': race,
                    'insurance': insurance,
                    'acuity': acuity,
                    'chief_complaint_category': category,
                    'true_los_minutes': round(true_los, 1),
                    'predicted_los_minutes': round(predicted_los, 1),
                    'error_minutes': round(predicted_los - true_los, 1),
                    'abs_error_minutes': round(abs(predicted_los - true_los), 1)
                })
            
            else:  # admission_prediction
                # True admission
                base_prob = {1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.02}[acuity]
                if age >= 65:
                    base_prob *= 1.2
                true_admitted = 1 if np.random.random() < base_prob else 0
                
                # Predicted probability
                pred_prob = base_prob + np.random.normal(0, 0.15)
                pred_prob = max(0.01, min(0.99, pred_prob))
                predicted_admitted = 1 if pred_prob >= 0.5 else 0
                
                data.append({
                    'sample_id': i + 1,
                    'age': age,
                    'age_group': age_group,
                    'gender': gender,
                    'race': race,
                    'insurance': insurance,
                    'acuity': acuity,
                    'chief_complaint_category': category,
                    'true_admitted': true_admitted,
                    'predicted_admitted': predicted_admitted,
                    'predicted_probability': round(pred_prob, 4)
                })
        
        df = pd.DataFrame(data)
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {len(df)} ML evaluation samples -> {filepath}")
        
        return df
    
    def generate_patient_events(
        self,
        num_patients: int = 1000,
        filename: str = "patient_events.csv"
    ) -> pd.DataFrame:
        """
        Generate detailed patient event/journey data.
        
        Each patient has multiple events (arrival, triage, bed assignment, etc.)
        """
        events = []
        
        for i in range(num_patients):
            patient_id = f'PAT_{i+1:06d}'
            stay_id = f'STAY_{i+1:06d}'
            
            # Random arrival time in last 30 days
            arrival = datetime.now() - timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
            
            # Event 1: Arrival
            events.append({
                'patient_id': patient_id,
                'stay_id': stay_id,
                'event_type': 'arrival',
                'timestamp': arrival.strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'waiting_room',
                'duration_minutes': None,
                'acuity': acuity
            })
            
            # Event 2: Triage
            triage_time = arrival + timedelta(minutes=random.randint(5, 15))
            triage_duration = random.randint(5, 15)
            events.append({
                'patient_id': patient_id,
                'stay_id': stay_id,
                'event_type': 'triage',
                'timestamp': triage_time.strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'triage',
                'duration_minutes': triage_duration,
                'acuity': acuity
            })
            
            # Event 3: Bed Assignment
            wait = 10 if acuity <= 2 else random.randint(20, 90)
            bed_time = triage_time + timedelta(minutes=wait)
            events.append({
                'patient_id': patient_id,
                'stay_id': stay_id,
                'event_type': 'bed_assignment',
                'timestamp': bed_time.strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'main_ed' if acuity >= 3 else 'resuscitation',
                'duration_minutes': None,
                'acuity': acuity
            })
            
            # Event 4: Physician Evaluation
            physician_time = bed_time + timedelta(minutes=random.randint(5, 30))
            events.append({
                'patient_id': patient_id,
                'stay_id': stay_id,
                'event_type': 'physician_evaluation',
                'timestamp': physician_time.strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'main_ed',
                'duration_minutes': random.randint(10, 30),
                'acuity': acuity
            })
            
            # Event 5: Disposition
            los = max(30, np.random.gamma(
                self.LOS_PARAMS[acuity]['shape'],
                self.LOS_PARAMS[acuity]['scale']
            ))
            depart_time = arrival + timedelta(minutes=los)
            disposition = np.random.choice(
                list(self.DISPOSITION_PROBS[acuity].keys()),
                p=list(self.DISPOSITION_PROBS[acuity].values())
            )
            events.append({
                'patient_id': patient_id,
                'stay_id': stay_id,
                'event_type': 'disposition',
                'timestamp': depart_time.strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'main_ed',
                'duration_minutes': None,
                'acuity': acuity,
                'disposition': disposition
            })
        
        df = pd.DataFrame(events)
        
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {len(df)} patient events -> {filepath}")
        
        return df
    
    def generate_all_datasets(
        self,
        num_patients: int = 10000,
        days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all synthetic datasets.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of all generated datasets
        """
        logger.info(f"Generating all synthetic datasets: {num_patients} patients, {days} days")
        
        datasets = {
            'patient_arrivals': self.generate_patient_arrivals(
                num_patients=num_patients, days=days
            ),
            'hourly_metrics': self.generate_hourly_metrics(days=days),
            'patient_events': self.generate_patient_events(
                num_patients=min(num_patients, 2000)
            ),
            'los_evaluation': self.generate_ml_evaluation_data(
                num_samples=5000, task='los_prediction',
                filename='los_evaluation_data.csv'
            ),
            'admission_evaluation': self.generate_ml_evaluation_data(
                num_samples=5000, task='admission_prediction',
                filename='admission_evaluation_data.csv'
            )
        }
        
        logger.info(f"All datasets generated in {self.output_dir}")
        return datasets
    
    def _weighted_random_hour(self) -> int:
        """Generate random hour weighted by arrival pattern"""
        hours = list(self.HOURLY_PATTERN.keys())
        weights = list(self.HOURLY_PATTERN.values())
        weights = [w / sum(weights) for w in weights]
        return np.random.choice(hours, p=weights)


# Convenience function
def generate_synthetic_csvs(
    output_dir: str = "./data/synthetic",
    num_patients: int = 10000,
    days: int = 90,
    seed: int = 42
) -> Dict[str, str]:
    """
    Generate all synthetic CSV files.
    
    Returns dict of filename -> filepath
    """
    generator = SyntheticCSVGenerator(output_dir=output_dir, seed=seed)
    datasets = generator.generate_all_datasets(num_patients=num_patients, days=days)
    
    return {
        name: os.path.join(output_dir, f"{name}.csv")
        for name in datasets.keys()
    }


# Singleton instance
csv_generator = SyntheticCSVGenerator()
