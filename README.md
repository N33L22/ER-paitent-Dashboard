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
