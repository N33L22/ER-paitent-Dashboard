# ER Patient Flow Intelligence Platform - Implementation Guide

## 🏗️ Architecture Overview

This platform implements a **microservices architecture** with 4 backend services and a Streamlit frontend:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (8501)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │Operational│ │Executive │ │Simulation│ │Research Explorer │   │
│  │   View   │ │   View   │ │   Lab    │ │                  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP/REST
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Data Service  │ │  ML Service   │ │  Simulation   │ │  Analytics    │
│    (8001)     │ │    (8002)     │ │   Service     │ │   Service     │
│               │ │               │ │    (8003)     │ │    (8004)     │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
│• MIMIC Loader │ │• LOS Predictor│ │• ED Simulation│ │• Granger      │
│• Synthetic Gen│ │• LSTM Forecast│ │• Scenario     │ │• Network Flow │
│• Feature Eng. │ │• Survival     │ │• Monte Carlo  │ │• Anomaly Det. │
│• API Endpoints│ │• SHAP Explain │ │• What-If      │ │• Bias Audit   │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

## 📁 Project Structure

```
er-platform/
├── docker-compose.yml          # Container orchestration
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
├── requirements-dev.txt        # Development dependencies
│
├── services/
│   ├── data-service/           # Port 8001
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py         # FastAPI application
│   │       ├── schemas.py      # Pydantic models
│   │       ├── data_loader.py  # MIMIC-IV-ED loader
│   │       ├── synthetic_generator.py  # Synthetic data
│   │       └── feature_engineer.py     # 50+ features
│   │
│   ├── ml-service/             # Port 8002
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py         # FastAPI application
│   │       ├── los_predictor.py      # XGBoost + SHAP
│   │       ├── arrival_forecaster.py # LSTM + uncertainty
│   │       └── survival_analysis.py  # Kaplan-Meier, Cox PH
│   │
│   ├── simulation-service/     # Port 8003
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app/
│   │       ├── __init__.py
│   │       ├── main.py         # FastAPI application
│   │       ├── discrete_event_sim.py # SimPy simulation
│   │       ├── scenario_engine.py    # What-if analysis
│   │       └── monte_carlo.py        # Uncertainty quantification
│   │
│   └── analytics-service/      # Port 8004
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── __init__.py
│           ├── main.py         # FastAPI application
│           ├── granger_causality.py  # Causal inference
│           ├── network_flow.py       # Graph analysis
│           ├── anomaly_detection.py  # Multi-method detection
│           └── bias_audit.py         # Fairness analysis
│
└── frontend/
    └── streamlit-app/
        ├── Dockerfile
        ├── requirements.txt
        ├── app.py              # Main dashboard
        └── pages/
            ├── 1_🔬_Operational_View.py
            ├── 2_📈_Executive_View.py
            ├── 3_🎮_Simulation_Lab.py
            └── 4_🧪_Research_Explorer.py
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone and enter directory
cd er-platform

# Copy environment file
cp .env.example .env

# Build and start all services
docker-compose up --build

# Access the platform
# Frontend: http://localhost:8501
# Data API: http://localhost:8001/docs
# ML API: http://localhost:8002/docs
# Simulation API: http://localhost:8003/docs
# Analytics API: http://localhost:8004/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies for each service
pip install -r services/data-service/requirements.txt
pip install -r services/ml-service/requirements.txt
pip install -r services/simulation-service/requirements.txt
pip install -r services/analytics-service/requirements.txt
pip install -r frontend/streamlit-app/requirements.txt

# Start each service in separate terminals
# Terminal 1: Data Service
cd services/data-service && uvicorn app.main:app --port 8001 --reload

# Terminal 2: ML Service
cd services/ml-service && uvicorn app.main:app --port 8002 --reload

# Terminal 3: Simulation Service
cd services/simulation-service && uvicorn app.main:app --port 8003 --reload

# Terminal 4: Analytics Service
cd services/analytics-service && uvicorn app.main:app --port 8004 --reload

# Terminal 5: Streamlit Frontend
cd frontend/streamlit-app && streamlit run app.py
```

## 📊 Service Details

### Data Service (Port 8001)

**Purpose**: Data ingestion, transformation, and feature engineering

**Key Endpoints**:
- `GET /patients` - Retrieve patient records
- `GET /hourly-metrics` - Aggregated hourly metrics
- `GET /queue-evolution` - Queue length over time
- `GET /current-state` - Current ED snapshot
- `GET /features/{patient_id}` - ML-ready features

**Components**:
- **MIMICDataLoader**: Loads MIMIC-IV-ED dataset from Parquet files
- **SyntheticDataGenerator**: Creates realistic synthetic data using:
  - Non-homogeneous Poisson process for arrivals
  - Gamma distributions for LOS by acuity
  - Realistic ESI distributions
- **FeatureEngineer**: Generates 50+ features including:
  - Temporal features (hour, day, cyclic encoding)
  - Demographic features (age groups, patient history)
  - Clinical features (vital signs, acuity)
  - System state features (census, queue length, utilization)

### ML Service (Port 8002)

**Purpose**: Predictive analytics and model serving

**Key Endpoints**:
- `POST /predict/los` - Predict length of stay
- `GET /predict/arrivals` - Forecast arrivals
- `GET /survival/{cohort}` - Survival curves
- `POST /train` - Train models on new data

**Models**:
1. **LOS Predictor** (XGBoost)
   - 50+ input features
   - SHAP explainability
   - 95% prediction intervals
   
2. **Arrival Forecaster** (LSTM)
   - 168-hour lookback window
   - 24-hour forecast horizon
   - Monte Carlo dropout for uncertainty

3. **Survival Analyzer** (Lifelines)
   - Kaplan-Meier curves
   - Cox Proportional Hazards
   - Stratified analysis

### Simulation Service (Port 8003)

**Purpose**: Digital twin simulation and scenario planning

**Key Endpoints**:
- `POST /simulate` - Run single simulation
- `POST /scenario` - Run custom scenario
- `GET /scenario/presets` - Available preset scenarios
- `POST /monte-carlo` - Uncertainty quantification

**Components**:
1. **EDSimulation** (SimPy)
   - Complete patient journey modeling
   - Resource constraints (beds, staff, equipment)
   - Priority queuing by acuity
   - LWBS modeling

2. **ScenarioEngine**
   - Preset scenarios (surge, staffing changes)
   - Custom scenario builder
   - Scenario matrix for sensitivity analysis

3. **MonteCarloSimulator**
   - Multiple replications
   - Risk probabilities
   - Confidence intervals

### Analytics Service (Port 8004)

**Purpose**: Advanced analytics and causal inference

**Key Endpoints**:
- `POST /causality/granger` - Granger causality analysis
- `POST /network/flow` - Patient flow network
- `POST /anomaly/detect` - Anomaly detection
- `POST /bias/audit` - Fairness audit

**Components**:
1. **GrangerCausalityAnalyzer**
   - Tests causal relationships between time series
   - Builds causal network
   - Impulse response functions

2. **NetworkFlowAnalyzer**
   - Graph-based flow modeling
   - Bottleneck identification
   - Critical path analysis

3. **AnomalyDetector**
   - Statistical (Z-score)
   - Isolation Forest
   - Local Outlier Factor
   - Temporal change detection

4. **BiasAuditor**
   - Demographic parity
   - Equalized odds
   - Equal opportunity
   - Predictive parity

## 🎨 Frontend Pages

### 1. Home Dashboard
- Real-time KPIs (census, wait times, LWBS)
- Census gauge
- Acuity distribution
- Arrival trends

### 2. Operational View
- Resource status (beds, staff)
- Queue evolution
- Patient tracking board
- Active alerts

### 3. Executive View
- Performance trends
- Benchmark comparisons
- Census heatmaps
- Financial impact

### 4. Simulation Lab
- Single simulation configuration
- Scenario comparison
- Monte Carlo analysis
- Risk assessment

### 5. Research Explorer
- SHAP explainability
- Survival analysis
- Anomaly detection
- Fairness audit

## 🔧 Configuration

### Environment Variables

```env
# Data Sources
MIMIC_DATA_PATH=/data/mimic-iv-ed

# Service URLs (for containerized deployment)
DATA_SERVICE_URL=http://data-service:8000
ML_SERVICE_URL=http://ml-service:8000
SIM_SERVICE_URL=http://simulation-service:8000
ANALYTICS_SERVICE_URL=http://analytics-service:8000

# Model Configuration
LOS_MODEL_PATH=/models/los_predictor.joblib
ARRIVAL_MODEL_PATH=/models/arrival_forecaster.h5

# Simulation Defaults
DEFAULT_SEED=42
SIMULATION_HOURS=168
```

## 📈 Key Metrics Tracked

| Metric | Definition | Target |
|--------|------------|--------|
| Door-to-Provider | Time from arrival to first provider contact | < 20 min |
| Door-to-Bed | Time from arrival to bed assignment | < 30 min |
| Length of Stay | Total time in ED | < 180 min |
| LWBS Rate | Left Without Being Seen | < 2% |
| Bed Utilization | Occupied beds / Total beds | 70-85% |
| Boarding Hours | Time admitted patients wait for bed | < 3 hours |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific service tests
pytest tests/test_data_service.py -v
pytest tests/test_ml_service.py -v
pytest tests/test_simulation_service.py -v
pytest tests/test_analytics_service.py -v

# Run with coverage
pytest tests/ --cov=services --cov-report=html
```

## 🔐 Security Considerations

1. **API Authentication**: Add JWT or OAuth2 for production
2. **Data Encryption**: Use TLS for all communications
3. **PHI Protection**: Ensure HIPAA compliance
4. **Access Logging**: Audit all data access
5. **Input Validation**: Pydantic models validate all inputs

## 📚 References

1. MIMIC-IV-ED Dataset: https://physionet.org/content/mimic-iv-ed/
2. XGBoost: https://xgboost.readthedocs.io/
3. SimPy: https://simpy.readthedocs.io/
4. SHAP: https://shap.readthedocs.io/
5. Lifelines: https://lifelines.readthedocs.io/

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.
