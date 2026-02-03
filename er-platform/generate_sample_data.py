"""
Script to generate synthetic CSV data files for testing
Run this script to create sample datasets in data/synthetic/

Usage:
    python generate_sample_data.py
"""

import os
import sys

# Add services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'data-service'))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random


def generate_patient_arrivals(num_patients=10000, days=90, output_dir="./data/synthetic"):
    """Generate patient arrivals dataset"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    # Hourly patterns
    hourly_pattern = {
        0: 0.5, 1: 0.4, 2: 0.35, 3: 0.3, 4: 0.3, 5: 0.35,
        6: 0.5, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.15,
        12: 1.2, 13: 1.15, 14: 1.1, 15: 1.1, 16: 1.15, 17: 1.2,
        18: 1.25, 19: 1.2, 20: 1.1, 21: 1.0, 22: 0.8, 23: 0.65
    }
    
    chief_complaints = {
        'chest_pain': ['Chest Pain', 'Chest Tightness'],
        'abdominal_pain': ['Abdominal Pain', 'Stomach Pain'],
        'shortness_of_breath': ['Shortness of Breath', 'Difficulty Breathing'],
        'headache': ['Headache', 'Migraine'],
        'back_pain': ['Back Pain', 'Lower Back Pain'],
        'fever': ['Fever', 'High Temperature'],
        'fall_injury': ['Fall', 'Mechanical Fall'],
        'laceration': ['Laceration', 'Cut'],
        'nausea_vomiting': ['Nausea', 'Vomiting'],
        'general_pain': ['General Pain', 'Body Pain']
    }
    
    los_params = {
        1: {'shape': 3.0, 'scale': 180},
        2: {'shape': 2.5, 'scale': 120},
        3: {'shape': 2.0, 'scale': 90},
        4: {'shape': 1.8, 'scale': 50},
        5: {'shape': 1.5, 'scale': 30}
    }
    
    disposition_probs = {
        1: {'admitted': 0.70, 'discharged': 0.15, 'transferred': 0.10, 'expired': 0.05},
        2: {'admitted': 0.45, 'discharged': 0.45, 'transferred': 0.08, 'expired': 0.02},
        3: {'admitted': 0.25, 'discharged': 0.70, 'transferred': 0.04, 'lwbs': 0.01},
        4: {'admitted': 0.08, 'discharged': 0.85, 'transferred': 0.02, 'lwbs': 0.05},
        5: {'admitted': 0.02, 'discharged': 0.88, 'transferred': 0.01, 'lwbs': 0.09}
    }
    
    start_date = datetime.now() - timedelta(days=days)
    data = []
    
    for i in range(num_patients):
        # Random arrival time weighted by hour
        day_offset = random.randint(0, days - 1)
        hours = list(hourly_pattern.keys())
        weights = list(hourly_pattern.values())
        weights = [w / sum(weights) for w in weights]
        hour = int(np.random.choice(hours, p=weights))
        minute = random.randint(0, 59)
        
        arrival_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # Acuity
        acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
        
        # Age based on acuity
        age_params = {1: (68, 15), 2: (60, 18), 3: (48, 22), 4: (38, 20), 5: (32, 18)}
        age = max(0, min(100, int(np.random.normal(*age_params[acuity]))))
        
        # Demographics
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                                p=[0.58, 0.18, 0.14, 0.06, 0.04])
        insurance = np.random.choice(['Private', 'Medicare', 'Medicaid', 'Self-Pay', 'Other'],
                                     p=[0.40, 0.25, 0.20, 0.10, 0.05])
        
        # Chief complaint
        category = np.random.choice(list(chief_complaints.keys()))
        complaint = random.choice(chief_complaints[category])
        
        # LOS
        params = los_params[acuity]
        los_minutes = max(15, np.random.gamma(params['shape'], params['scale']))
        
        # Wait time
        base_wait = 30 if acuity >= 3 else 10
        wait_minutes = max(0, np.random.exponential(base_wait * (6 - acuity) / 3))
        
        # Disposition
        disp_probs = disposition_probs[acuity]
        disposition = np.random.choice(list(disp_probs.keys()), p=list(disp_probs.values()))
        
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
            'mode_of_arrival': np.random.choice(['Walk-in', 'Ambulance', 'Transfer', 'Police'],
                                                p=[0.65, 0.28, 0.05, 0.02]),
            'wait_time_minutes': round(wait_minutes, 1),
            'los_minutes': round(los_minutes, 1),
            'disposition': disposition,
            'hour': arrival_time.hour,
            'day_of_week': arrival_time.weekday(),
            'is_weekend': 1 if arrival_time.weekday() >= 5 else 0
        })
    
    df = pd.DataFrame(data)
    filepath = os.path.join(output_dir, 'patient_arrivals.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} patient arrivals -> {filepath}")
    return df


def generate_ml_evaluation_data(num_samples=5000, output_dir="./data/synthetic"):
    """Generate data for model evaluation with ground truth and predictions"""
    
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    data = []
    
    for i in range(num_samples):
        # Demographics
        age = max(0, min(100, int(np.random.normal(50, 22))))
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'],
                                p=[0.58, 0.18, 0.14, 0.06, 0.04])
        insurance = np.random.choice(['Private', 'Medicare', 'Medicaid', 'Self-Pay'],
                                     p=[0.40, 0.28, 0.22, 0.10])
        
        acuity = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.15, 0.48, 0.30, 0.05])
        
        # Age group
        if age < 18:
            age_group = 'pediatric'
        elif age < 40:
            age_group = 'young_adult'
        elif age < 65:
            age_group = 'middle_age'
        else:
            age_group = 'senior'
        
        # True LOS
        los_params = {1: (3.0, 180), 2: (2.5, 120), 3: (2.0, 90), 4: (1.8, 50), 5: (1.5, 30)}
        shape, scale = los_params[acuity]
        true_los = max(15, np.random.gamma(shape, scale))
        
        # Predicted LOS with realistic error
        error_std = true_los * 0.25
        predicted_los = max(15, true_los + np.random.normal(0, error_std))
        
        # Slight bias for elderly (for demo)
        if age >= 65:
            predicted_los *= 0.92
        
        # True admission
        base_prob = {1: 0.70, 2: 0.45, 3: 0.25, 4: 0.08, 5: 0.02}[acuity]
        if age >= 65:
            base_prob *= 1.2
        true_admitted = 1 if np.random.random() < min(base_prob, 0.95) else 0
        
        # Predicted admission
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
            'true_los_minutes': round(true_los, 1),
            'predicted_los_minutes': round(predicted_los, 1),
            'los_error': round(predicted_los - true_los, 1),
            'los_abs_error': round(abs(predicted_los - true_los), 1),
            'true_admitted': true_admitted,
            'predicted_admitted': predicted_admitted,
            'admission_probability': round(pred_prob, 4)
        })
    
    df = pd.DataFrame(data)
    filepath = os.path.join(output_dir, 'ml_evaluation_data.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} evaluation samples -> {filepath}")
    return df


def generate_hourly_metrics(days=90, output_dir="./data/synthetic"):
    """Generate hourly ED metrics for forecasting"""
    
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    hourly_pattern = {
        0: 8, 1: 6, 2: 5, 3: 4, 4: 4, 5: 5,
        6: 7, 7: 10, 8: 14, 9: 18, 10: 22, 11: 24,
        12: 23, 13: 22, 14: 21, 15: 20, 16: 21, 17: 23,
        18: 25, 19: 24, 20: 22, 21: 18, 22: 14, 23: 10
    }
    
    start_date = datetime.now() - timedelta(days=days)
    data = []
    current = start_date
    
    for _ in range(days * 24):
        hour = current.hour
        dow = current.weekday()
        
        base_rate = hourly_pattern[hour]
        if dow in [5, 6]:
            base_rate *= 1.1
        
        arrivals = max(0, int(np.random.poisson(base_rate)))
        census = max(5, int(20 + np.random.normal(arrivals * 0.8, 5)))
        beds_occupied = min(25, max(0, int(census * 0.85 + np.random.normal(0, 2))))
        wait_time = max(5, 20 + arrivals * 2 + np.random.normal(0, 10))
        departures = max(0, int(arrivals * 0.9 + np.random.normal(0, 3)))
        
        data.append({
            'timestamp': current.strftime('%Y-%m-%d %H:%M:%S'),
            'arrivals': arrivals,
            'departures': departures,
            'census': census,
            'beds_occupied': beds_occupied,
            'beds_available': 25 - beds_occupied,
            'mean_wait_time': round(wait_time, 1),
            'hour': hour,
            'day_of_week': dow,
            'is_weekend': 1 if dow >= 5 else 0
        })
        
        current += timedelta(hours=1)
    
    df = pd.DataFrame(data)
    filepath = os.path.join(output_dir, 'hourly_metrics.csv')
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} hourly records -> {filepath}")
    return df


def main():
    """Generate all sample datasets"""
    output_dir = "./data/synthetic"
    
    print("=" * 60)
    print("Generating Synthetic Data for ER Platform")
    print("=" * 60)
    
    # Generate datasets
    generate_patient_arrivals(num_patients=10000, days=90, output_dir=output_dir)
    generate_ml_evaluation_data(num_samples=5000, output_dir=output_dir)
    generate_hourly_metrics(days=90, output_dir=output_dir)
    
    print("=" * 60)
    print(f"All datasets generated in {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
