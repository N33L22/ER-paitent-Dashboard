"""
Universal Data Uploader for ER Patient Flow Intelligence Platform
Supports CSV, Excel, JSON, Parquet with auto-schema detection

Authors: Neel, Harsh, Tanishk
"""

from typing import Union, Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import json
from loguru import logger


class UniversalDataUploader:
    """
    Intelligent data uploader that:
    1. Accepts CSV, Excel, JSON, Parquet
    2. Auto-detects schema
    3. Validates ER data structure
    4. Maps to canonical event schema
    5. Generates synthetic features if missing
    """
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    def __init__(self):
        self.schema_detector = SchemaDetector()
        self.feature_generator = AutoFeatureGenerator()
        self.last_upload_result: Optional[Dict] = None
        
    def upload_and_process(
        self, 
        file_obj: Union[BytesIO, str], 
        filename: str,
        sheet_name: Optional[str] = None
    ) -> Dict:
        """
        Main upload handler
        
        Args:
            file_obj: File object or path
            filename: Original filename
            sheet_name: For Excel files, specify sheet
        
        Returns:
        {
            'success': bool,
            'message': str,
            'row_count': int,
            'columns': List[str],
            'detected_schema': Dict,
            'validation_warnings': List[str],
            'preview_data': pd.DataFrame,
            'canonical_data': pd.DataFrame
        }
        """
        try:
            # 1. Detect file type
            file_ext = filename.lower().split('.')[-1]
            
            logger.info(f"Processing uploaded file: {filename} ({file_ext})")
            
            # 2. Load data
            df = self._load_file(file_obj, file_ext, sheet_name)
            
            if df is None or len(df) == 0:
                return {
                    'success': False,
                    'message': 'Empty file or unsupported format',
                    'row_count': 0,
                    'columns': [],
                    'detected_schema': {},
                    'validation_warnings': ['File is empty or could not be parsed'],
                    'preview_data': pd.DataFrame(),
                    'canonical_data': pd.DataFrame()
                }
            
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # 3. Auto-detect schema
            schema_result = self.schema_detector.analyze(df)
            
            # 4. Validate ER data requirements
            validation = self.validate_er_data(df, schema_result)
            
            # 5. Generate missing features
            df_enriched = self.feature_generator.enrich(df, schema_result)
            
            # 6. Map to canonical schema
            df_canonical = self.map_to_canonical_schema(df_enriched, schema_result)
            
            result = {
                'success': True,
                'message': f'Successfully loaded {len(df)} records from {filename}',
                'row_count': len(df),
                'columns': df.columns.tolist(),
                'detected_schema': schema_result,
                'validation_warnings': validation['warnings'],
                'preview_data': df.head(100).to_dict('records'),
                'canonical_data': df_canonical.to_dict('records'),
                'original_data': df.to_dict('records')
            }
            
            self.last_upload_result = result
            return result
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing file: {str(e)}',
                'row_count': 0,
                'columns': [],
                'detected_schema': {},
                'validation_warnings': [f'Processing error: {str(e)}'],
                'preview_data': [],
                'canonical_data': []
            }
    
    def _load_file(
        self, 
        file_obj: Union[BytesIO, str], 
        file_ext: str,
        sheet_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load file based on extension"""
        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_obj, parse_dates=True)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_obj, sheet_name=sheet_name or 0)
            elif file_ext == 'json':
                df = pd.read_json(file_obj)
            elif file_ext == 'parquet':
                df = pd.read_parquet(file_obj)
            else:
                logger.error(f'Unsupported format: {file_ext}')
                return None
            
            # Convert datetime columns
            df = self._parse_datetime_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return None
    
    def _parse_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to parse datetime columns"""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to parse as datetime
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    # If >50% parsed successfully, use it
                    if parsed.notna().sum() / len(df) > 0.5:
                        df[col] = parsed
                except:
                    pass
        return df
    
    def validate_er_data(self, df: pd.DataFrame, schema: Dict) -> Dict:
        """
        Validate essential ER fields exist or can be inferred
        
        Required fields (or can be generated):
        - patient_id / stay_id
        - arrival_time / timestamp
        - departure_time (optional, can compute from LOS)
        - acuity / severity (can infer from other fields)
        """
        warnings = []
        errors = []
        
        # Check for patient identifier
        if not schema.get('has_patient_id'):
            warnings.append("⚠️ No patient ID found - generating synthetic IDs")
        
        # Check for timestamps
        if not schema.get('has_arrival_time'):
            warnings.append("⚠️ No arrival timestamp found - using row index as time sequence")
        
        # Check for LOS or departure time
        if not (schema.get('has_departure_time') or schema.get('has_los')):
            warnings.append("⚠️ No departure time or LOS - will estimate from data patterns")
        
        # Check for acuity
        if not schema.get('has_acuity'):
            warnings.append("ℹ️ No acuity field - will cluster patients into severity levels")
        
        # Check row count
        if len(df) < 10:
            warnings.append("⚠️ Very small dataset - some analytics may be unreliable")
        
        # Check for duplicate patient IDs
        if schema.get('patient_id_col'):
            dupes = df[schema['patient_id_col']].duplicated().sum()
            if dupes > 0:
                warnings.append(f"ℹ️ {dupes} duplicate patient IDs found - treating as repeat visits")
        
        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors
        }
    
    def map_to_canonical_schema(self, df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
        """
        Map uploaded data to canonical ER schema
        
        Canonical columns:
        - patient_id
        - arrival_time
        - departure_time
        - los_minutes
        - acuity
        - disposition
        """
        canonical = pd.DataFrame()
        
        # Patient ID
        if schema.get('patient_id_col'):
            canonical['patient_id'] = df[schema['patient_id_col']]
        elif 'generated_patient_id' in df.columns:
            canonical['patient_id'] = df['generated_patient_id']
        else:
            canonical['patient_id'] = [f'PAT_{i:06d}' for i in range(len(df))]
        
        # Arrival time
        if schema.get('arrival_time_col'):
            canonical['arrival_time'] = pd.to_datetime(df[schema['arrival_time_col']])
        elif 'generated_arrival_time' in df.columns:
            canonical['arrival_time'] = df['generated_arrival_time']
        else:
            base = datetime.now() - timedelta(days=30)
            canonical['arrival_time'] = [base + timedelta(hours=i) for i in range(len(df))]
        
        # Departure time / LOS
        if schema.get('departure_time_col'):
            canonical['departure_time'] = pd.to_datetime(df[schema['departure_time_col']])
            canonical['los_minutes'] = (
                canonical['departure_time'] - canonical['arrival_time']
            ).dt.total_seconds() / 60
        elif schema.get('los_col'):
            canonical['los_minutes'] = df[schema['los_col']].astype(float)
            canonical['departure_time'] = (
                canonical['arrival_time'] + 
                pd.to_timedelta(canonical['los_minutes'], unit='m')
            )
        elif 'generated_los_minutes' in df.columns:
            canonical['los_minutes'] = df['generated_los_minutes']
            canonical['departure_time'] = (
                canonical['arrival_time'] + 
                pd.to_timedelta(canonical['los_minutes'], unit='m')
            )
        else:
            # Default LOS distribution
            canonical['los_minutes'] = np.random.gamma(4, 60, len(df))
            canonical['departure_time'] = (
                canonical['arrival_time'] + 
                pd.to_timedelta(canonical['los_minutes'], unit='m')
            )
        
        # Acuity
        if schema.get('acuity_col'):
            canonical['acuity'] = df[schema['acuity_col']]
        elif 'generated_acuity' in df.columns:
            canonical['acuity'] = df['generated_acuity']
        else:
            canonical['acuity'] = np.random.choice(
                [1, 2, 3, 4, 5], 
                size=len(df),
                p=[0.01, 0.15, 0.50, 0.30, 0.04]
            )
        
        # Disposition
        if schema.get('disposition_col'):
            canonical['disposition'] = df[schema['disposition_col']]
        else:
            canonical['disposition'] = np.random.choice(
                ['Discharged', 'Admitted', 'Transferred', 'LWBS', 'AMA'],
                size=len(df),
                p=[0.70, 0.20, 0.03, 0.05, 0.02]
            )
        
        # Add derived features
        canonical['hour'] = canonical['arrival_time'].dt.hour
        canonical['day_of_week'] = canonical['arrival_time'].dt.dayofweek
        canonical['is_weekend'] = canonical['day_of_week'].isin([5, 6]).astype(int)
        canonical['shift'] = canonical['hour'].apply(self._get_shift)
        
        return canonical
    
    def _get_shift(self, hour: int) -> str:
        """Determine shift from hour"""
        if 7 <= hour < 15:
            return 'Day'
        elif 15 <= hour < 23:
            return 'Evening'
        else:
            return 'Night'
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for uploaded data"""
        summary = {
            'total_records': len(df),
            'date_range': None,
            'columns': len(df.columns),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_cols': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        }
        
        # Get date range if datetime columns exist
        dt_cols = df.select_dtypes(include=['datetime64']).columns
        if len(dt_cols) > 0:
            first_col = dt_cols[0]
            summary['date_range'] = {
                'start': df[first_col].min().isoformat() if pd.notna(df[first_col].min()) else None,
                'end': df[first_col].max().isoformat() if pd.notna(df[first_col].max()) else None
            }
        
        return summary


class SchemaDetector:
    """
    Intelligent schema detection using pattern matching and statistics
    """
    
    # Common column name patterns for ER data
    PATIENT_ID_PATTERNS = ['patient', 'subject', 'mrn', 'id', 'stay', 'visit', 'encounter']
    ARRIVAL_PATTERNS = ['arrival', 'admit', 'checkin', 'intime', 'start', 'registration']
    DEPARTURE_PATTERNS = ['departure', 'discharge', 'checkout', 'outtime', 'end', 'leave']
    LOS_PATTERNS = ['los', 'length', 'duration', 'stay', 'time_in_ed']
    ACUITY_PATTERNS = ['acuity', 'esi', 'triage', 'severity', 'priority', 'level']
    DISPOSITION_PATTERNS = ['disposition', 'outcome', 'discharge_type', 'dispo']
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze dataframe and detect ER-relevant fields
        """
        result = {
            'has_patient_id': False,
            'patient_id_col': None,
            'has_arrival_time': False,
            'arrival_time_col': None,
            'has_departure_time': False,
            'departure_time_col': None,
            'has_los': False,
            'los_col': None,
            'has_acuity': False,
            'acuity_col': None,
            'has_disposition': False,
            'disposition_col': None,
            'numeric_cols': [],
            'datetime_cols': [],
            'categorical_cols': [],
            'text_cols': [],
            'column_info': {}
        }
        
        for col in df.columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            col_info = self._analyze_column(df[col], col)
            result['column_info'][col] = col_info
            
            # Patient ID detection
            if any(keyword in col_lower for keyword in self.PATIENT_ID_PATTERNS):
                if df[col].nunique() / len(df) > 0.8:  # High cardinality
                    result['has_patient_id'] = True
                    result['patient_id_col'] = col
            
            # Timestamp detection
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                result['datetime_cols'].append(col)
                
                if any(keyword in col_lower for keyword in self.ARRIVAL_PATTERNS):
                    result['has_arrival_time'] = True
                    result['arrival_time_col'] = col
                elif any(keyword in col_lower for keyword in self.DEPARTURE_PATTERNS):
                    result['has_departure_time'] = True
                    result['departure_time_col'] = col
            
            # LOS detection
            if any(keyword in col_lower for keyword in self.LOS_PATTERNS):
                if pd.api.types.is_numeric_dtype(df[col]):
                    result['has_los'] = True
                    result['los_col'] = col
            
            # Acuity detection
            if any(keyword in col_lower for keyword in self.ACUITY_PATTERNS):
                result['has_acuity'] = True
                result['acuity_col'] = col
            
            # Disposition detection
            if any(keyword in col_lower for keyword in self.DISPOSITION_PATTERNS):
                result['has_disposition'] = True
                result['disposition_col'] = col
            
            # Type classification
            if pd.api.types.is_numeric_dtype(df[col]):
                result['numeric_cols'].append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05:  # Low cardinality = categorical
                    result['categorical_cols'].append(col)
                else:
                    result['text_cols'].append(col)
        
        # If no arrival time detected, use first datetime column
        if not result['has_arrival_time'] and result['datetime_cols']:
            result['has_arrival_time'] = True
            result['arrival_time_col'] = result['datetime_cols'][0]
        
        return result
    
    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict:
        """Analyze a single column"""
        info = {
            'name': col_name,
            'dtype': str(series.dtype),
            'non_null_count': series.notna().sum(),
            'null_count': series.isna().sum(),
            'null_pct': series.isna().sum() / len(series) * 100,
            'unique_count': series.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(series):
            info['min'] = float(series.min()) if pd.notna(series.min()) else None
            info['max'] = float(series.max()) if pd.notna(series.max()) else None
            info['mean'] = float(series.mean()) if pd.notna(series.mean()) else None
            info['std'] = float(series.std()) if pd.notna(series.std()) else None
        
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            info['top_values'] = series.value_counts().head(5).to_dict()
        
        if pd.api.types.is_datetime64_any_dtype(series):
            info['min_date'] = series.min().isoformat() if pd.notna(series.min()) else None
            info['max_date'] = series.max().isoformat() if pd.notna(series.max()) else None
        
        return info


class AutoFeatureGenerator:
    """
    Automatically generate missing ER features
    """
    
    def enrich(self, df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
        """
        Generate synthetic features for missing fields
        """
        df = df.copy()
        
        # Generate patient IDs if missing
        if not schema.get('has_patient_id'):
            df['generated_patient_id'] = [f'PAT_{i:06d}' for i in range(len(df))]
            logger.info("Generated synthetic patient IDs")
        
        # Generate timestamps if missing
        if not schema.get('has_arrival_time'):
            # Use row index as sequential time (variable intervals)
            base_time = datetime.now() - timedelta(days=30)
            intervals = np.random.exponential(60, len(df))  # ~60 min between arrivals
            cumulative = np.cumsum(intervals)
            df['generated_arrival_time'] = [
                base_time + timedelta(minutes=int(m)) for m in cumulative
            ]
            logger.info("Generated synthetic arrival times")
        
        # Generate LOS if missing (when both departure time and LOS are missing)
        if not schema.get('has_departure_time') and not schema.get('has_los'):
            # Gamma distribution for LOS (realistic ED times)
            df['generated_los_minutes'] = np.random.gamma(4, 60, len(df))  # Mean ~4 hours
            logger.info("Generated synthetic LOS values")
        
        # Generate acuity if missing
        if not schema.get('has_acuity'):
            # Use K-means clustering on numeric features to infer severity
            numeric_cols = [c for c in schema.get('numeric_cols', [])[:5] if c in df.columns]
            
            if numeric_cols and len(numeric_cols) >= 2:
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    scaler = StandardScaler()
                    features = scaler.fit_transform(df[numeric_cols].fillna(0))
                    
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    df['generated_acuity'] = kmeans.fit_predict(features) + 1  # 1-5 scale
                    logger.info("Generated acuity levels using clustering")
                except Exception as e:
                    logger.warning(f"Clustering failed: {e}, using random assignment")
                    df['generated_acuity'] = np.random.choice(
                        [1, 2, 3, 4, 5], 
                        size=len(df),
                        p=[0.01, 0.15, 0.50, 0.30, 0.04]
                    )
            else:
                # Random assignment matching typical ESI distribution
                df['generated_acuity'] = np.random.choice(
                    [1, 2, 3, 4, 5], 
                    size=len(df),
                    p=[0.01, 0.15, 0.50, 0.30, 0.04]
                )
                logger.info("Generated acuity using ESI distribution")
        
        return df
    
    def generate_temporal_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Generate time-based features from timestamp"""
        df = df.copy()
        
        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col])
            
            df['hour'] = ts.dt.hour
            df['day_of_week'] = ts.dt.dayofweek
            df['day_name'] = ts.dt.day_name()
            df['month'] = ts.dt.month
            df['is_weekend'] = ts.dt.dayofweek.isin([5, 6]).astype(int)
            df['is_night'] = ((ts.dt.hour >= 23) | (ts.dt.hour < 7)).astype(int)
            df['quarter'] = ts.dt.quarter
            df['week_of_year'] = ts.dt.isocalendar().week
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
            df['dow_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            df['dow_cos'] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        
        return df


# Singleton instance
uploader = UniversalDataUploader()
