# 🏥 Emergency Department Patient Flow Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)

A PhD-level, production-grade analytics platform for Emergency Department operations optimization. This system combines advanced machine learning, discrete-event simulation, causal inference, and interactive visualizations to provide actionable insights for healthcare administrators.

## 🎯 Core Capabilities

### 1. Predictive Analytics Engine
- **Arrival Forecasting**: LSTM/GRU models predicting patient volumes 1-24 hours ahead with 90% confidence intervals
- **Length of Stay Prediction**: XGBoost regression + survival analysis for individual patients
- **Bottleneck Detection**: Unsupervised learning to identify system constraints in real-time
- **Resource Demand Forecasting**: Predict bed, physician, nurse, imaging, lab utilization

### 2. Digital Twin Simulation
- **Discrete-Event Simulation**: Agent-based modeling of patient flow through ED stages
- **What-If Scenarios**: Test staffing changes, demand surges, policy modifications
- **Counterfactual Reasoning**: Compare intervention outcomes under uncertainty
- **Monte Carlo Analysis**: Quantify risk and variability in operational decisions

### 3. Causal Inference & Network Analysis
- **Granger Causality Testing**: Identify temporal cause-effect relationships
- **Network Flow Analysis**: Betweenness centrality to find critical bottleneck nodes
- **Lag Structure Mapping**: Understand how congestion propagates through the system
- **Change-Point Detection**: Identify regime shifts from stable to unstable states

### 4. Advanced Visualizations (12 Decision Instruments)
- Temporal Queue Evolution Surfaces (3D)
- Patient Trajectory Manifolds (UMAP/t-SNE)
- Bottleneck Centrality Network Graphs
- Congestion Propagation Heatmaps
- Length-of-Stay Survival Curves
- Counterfactual Scenario Lattices
- Operational Pareto Frontiers
- Temporal Causal Graphs
- Flow Volatility Maps
- Ethical Bias Audit Panels
- Digital Twin Animated Replays
- Uncertainty-Aware Forecast Bands

## 🏗️ Architecture

```
er-platform/
├── services/
│   ├── data-service/          # Port 8001 - Data ingestion & feature engineering
│   ├── ml-service/            # Port 8002 - ML models & predictions
│   ├── simulation-service/    # Port 8003 - SimPy discrete-event simulation
│   └── analytics-service/     # Port 8004 - Causal analysis & network flow
├── frontend/
│   └── streamlit-app/         # Port 8501 - Interactive dashboard
├── data/
│   ├── raw/                   # MIMIC-IV-ED CSVs
│   ├── processed/             # Feature-engineered data
│   ├── synthetic/             # Simulated scenarios
│   └── models/                # Trained models
├── docs/                      # Documentation
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- MIMIC-IV-ED dataset (optional, synthetic data available)

### One-Command Launch
```bash
# Clone and navigate to project
cd er-platform

# Build and start all services
docker-compose up --build

# Access the platform
# Frontend: http://localhost:8501
# Data API: http://localhost:8001/docs
# ML API: http://localhost:8002/docs
# Simulation API: http://localhost:8003/docs
# Analytics API: http://localhost:8004/docs
```

### Local Development
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Start individual services
cd services/data-service && uvicorn app.main:app --port 8001 --reload
cd services/ml-service && uvicorn app.main:app --port 8002 --reload
cd services/simulation-service && uvicorn app.main:app --port 8003 --reload
cd services/analytics-service && uvicorn app.main:app --port 8004 --reload

# Start frontend
cd frontend/streamlit-app && streamlit run app.py
```

## 📊 Data Sources

### Primary: MIMIC-IV-ED
- **Source**: MIT Critical Data, PhysioNet
- **Access**: Requires credentialing (free for researchers)
- **Size**: 560,000+ ED visits (2011-2019)
- **URL**: https://physionet.org/content/mimic-iv-ed/2.2/

### Synthetic Data
When real data is unavailable, the platform generates realistic synthetic data:
- Non-homogeneous Poisson arrivals (λ(t) varies by hour/day/season)
- ESI acuity distribution: ESI 1 (1%), 2 (15%), 3 (50%), 4 (30%), 5 (4%)
- LOS distributions: Gamma-distributed, stratified by acuity
- Service times: Empirical distributions from MIMIC-IV-ED benchmarks

#### Generating Synthetic CSV Files
The platform includes tools to generate CSV files for testing model uploads:

```bash
# Using the standalone script
python generate_sample_data.py

# Via API endpoint
curl http://localhost:8001/synthetic/generate-csv?dataset_type=all
```

Generated files in `data/synthetic/`:
- `patient_arrivals.csv` - Patient arrival records with demographics and outcomes
- `ml_evaluation_data.csv` - Labeled data for model evaluation (predictions vs actuals)
- `hourly_metrics.csv` - Aggregated hourly ED metrics

## 🔄 Real-Time Streaming

The platform supports Server-Sent Events (SSE) for real-time data streaming:

### Streaming Endpoints (Data Service - Port 8001)
| Endpoint | Description |
|----------|-------------|
| `/stream/arrivals` | Real-time patient arrival events |
| `/stream/state` | ED system state updates (occupancy, wait times) |
| `/stream/metrics` | Aggregated performance metrics |
| `/stream/combined` | Unified stream with all event types |
| `/stream/snapshot` | Current system state snapshot |

### Usage Example
```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream('GET', 'http://localhost:8001/stream/arrivals') as response:
        async for line in response.aiter_lines():
            if line.startswith('data:'):
                event = json.loads(line[5:])
                print(f"New arrival: {event}")
```

## 📊 Model Evaluation

### Classification Metrics
- **Confusion Matrix**: TP, TN, FP, FN breakdown
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC and PR-AUC curves**
- **Per-class metrics for multi-class classification**

### Regression Metrics
- **MAE, MSE, RMSE, MAPE**
- **R² Score**
- **Clinical LOS metrics**: Within 30/60/120 min accuracy

### Evaluation Endpoints (ML Service - Port 8002)
| Endpoint | Description |
|----------|-------------|
| `/evaluate/classification` | Evaluate classification models |
| `/evaluate/regression` | Evaluate regression models |
| `/evaluate/los` | Specialized LOS prediction evaluation |
| `/evaluate/threshold-analysis` | Analyze different threshold values |
| `/evaluate/compare` | Compare multiple models |

### Example: LOS Model Evaluation
```python
import requests

response = requests.post("http://localhost:8002/evaluate/los", json={
    "y_true": [120, 240, 60, 180],
    "y_pred": [130, 220, 75, 190],
    "feature_names": ["acuity", "age", "chief_complaint"]
})
print(response.json())
# Returns MAE, RMSE, within-threshold accuracy, category-level metrics
```

## ⚖️ Fairness & Bias Analysis

### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Predictive Parity**: Equal PPV across groups
- **Disparate Impact Ratio**: 80% rule compliance

### Bias Audit Endpoints (Analytics Service - Port 8004)
| Endpoint | Description |
|----------|-------------|
| `/fairness/scorecard` | Complete fairness scorecard with grades |
| `/fairness/intersectional` | Intersectional bias analysis |
| `/fairness/visualization-data` | Data formatted for bias visualizations |
| `/evaluate/classification` | Evaluation with fairness context |

### Fairness Grading System
| Grade | Score | Status |
|-------|-------|--------|
| A | 90-100 | Excellent - No significant bias detected |
| B | 80-89 | Good - Minor disparities, monitor |
| C | 70-79 | Fair - Moderate bias, mitigation recommended |
| D | 60-69 | Poor - Significant bias, action required |
| F | <60 | Failing - Critical bias, immediate action needed |

## 📊 Frontend Dashboard Pages

The Streamlit frontend provides five interactive dashboard pages:

| Page | Description |
|------|-------------|
| 🔬 **Operational View** | Real-time ED monitoring with queue status, wait times, and patient tracking |
| 📈 **Executive View** | High-level KPIs, trend analysis, and strategic insights for administrators |
| 🎮 **Simulation Lab** | What-if scenarios, Monte Carlo simulations, and intervention testing |
| 🧪 **Research Explorer** | ML explainability with SHAP values, survival analysis, and causal inference |
| ⚖️ **Fairness & Evaluation** | Model performance metrics, bias detection, and multi-model comparison |

### ⚖️ Fairness & Evaluation Dashboard

The new Fairness & Evaluation page provides comprehensive model analysis:

**Model Evaluation Features:**
- Interactive Confusion Matrix with counts and percentages
- Classification metrics (Accuracy, Precision, Recall, F1, Specificity, ROC-AUC)
- ROC Curve comparison across multiple models
- Radar chart for multi-model performance comparison
- Performance vs. Speed tradeoff analysis

**Fairness Analysis Features:**
- Fairness Scorecard with letter grades (A-F)
- Demographic disparity analysis across protected attributes:
  - Age groups (18-40, 41-64, 65+)
  - Sex (Male, Female)
  - Insurance type (Private, Medicare, Medicaid, Uninsured)
  - Acuity levels (ESI 1-2, ESI 3, ESI 4-5)
- Group performance comparisons with 4/5 rule thresholds
- Intersectional bias detection
- Fairness trends over time
- Mitigation recommendations

## 📤 Data Upload & Testing

### Upload Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload/csv` | POST | Upload CSV file for model testing |
| `/evaluation/los-data` | GET | Get LOS evaluation dataset |
| `/evaluation/admission-data` | GET | Get admission prediction dataset |

### CSV Upload Example
```python
import requests

with open('my_data.csv', 'rb') as f:
    response = requests.post(
        "http://localhost:8001/upload/csv",
        files={"file": ("data.csv", f, "text/csv")}
    )
print(response.json())
# Returns upload status and data summary
```

## 🤖 Machine Learning Models

| Model | Purpose | Architecture | Target Metrics |
|-------|---------|--------------|----------------|
| XGBoost LOS | Predict individual patient LOS | XGBoost Regressor (50+ features) | MAE < 45 min, R² > 0.65 |
| LSTM Arrival | Forecast arrivals 1-24h ahead | LSTM(128→64) + Dense | MAPE < 10% |
| Survival Analysis | Risk-stratified LOS curves | Kaplan-Meier + Cox PH | Coverage > 90% |

## 📈 Performance Benchmarks

- **API Latency**: P95 < 200ms
- **Frontend Load**: < 2 seconds
- **Test Coverage**: > 80%
- **ML Accuracy**: MAPE < 10%
- **Scalability**: 100K+ patient records

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Data Schema](docs/DATA_SCHEMA.md)
- [Model Documentation](docs/MODEL_DOCUMENTATION.md)
- [API Reference](docs/API_REFERENCE.md)
- [User Guide](docs/USER_GUIDE.md)

## 🔬 Research Foundations

### Key References
1. Morley et al. (2018) - ED overcrowding mortality impact
2. Singer et al. (2011) - Boarding effects on outcomes
3. Raita et al. (2019) - ML for ED triage
4. Karakra et al. (2019) - Hospital digital twins
5. Green & Kolesar (2004) - Queueing theory fundamentals

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the development team.

---

**Built with ❤️ for healthcare operations excellence**
