"""
Analytics Service - FastAPI Application
Granger causality, network flow, anomaly detection, bias auditing
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from .granger_causality import GrangerCausalityAnalyzer, CausalNetwork
from .network_flow import NetworkFlowAnalyzer, FlowMetrics
from .anomaly_detection import AnomalyDetector, AnomalyReport
from .bias_audit import BiasAuditor, BiasReport


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class TimeSeriesData(BaseModel):
    """Time series data for analysis"""
    timestamps: List[str]
    metrics: Dict[str, List[float]]


class GrangerRequest(BaseModel):
    """Request for Granger causality analysis"""
    data: TimeSeriesData
    variables: List[str]
    max_lag: int = Field(default=12, ge=1, le=48)
    significance_level: float = Field(default=0.05, ge=0.01, le=0.1)


class ImpulseResponseRequest(BaseModel):
    """Request for impulse response analysis"""
    data: TimeSeriesData
    variables: List[str]
    shock_variable: str
    periods: int = Field(default=24, ge=6, le=72)


class NetworkFlowRequest(BaseModel):
    """Request for network flow analysis"""
    journeys: List[Dict[str, Any]]
    total_hours: float = Field(default=168, ge=24)


class AnomalyRequest(BaseModel):
    """Request for anomaly detection"""
    data: TimeSeriesData
    metrics: List[str]
    z_threshold: float = Field(default=3.0, ge=2.0, le=5.0)
    contamination: float = Field(default=0.05, ge=0.01, le=0.2)


class BiasAuditRequest(BaseModel):
    """Request for bias audit"""
    y_true: List[float]
    y_pred: List[float]
    y_prob: Optional[List[float]] = None
    protected_attributes: Dict[str, List[str]]
    is_regression: bool = False
    model_name: str = "model"


class HealthCheck(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime


# =============================================================================
# GLOBAL STATE
# =============================================================================

class AnalyticsStore:
    """Store for analytics state"""
    def __init__(self):
        self.granger_analyzer: Optional[GrangerCausalityAnalyzer] = None
        self.network_analyzer: Optional[NetworkFlowAnalyzer] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.bias_auditor: Optional[BiasAuditor] = None
        self.last_causal_network: Optional[CausalNetwork] = None
        self.last_anomaly_report: Optional[AnomalyReport] = None
        self.is_initialized: bool = False


analytics_store = AnalyticsStore()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize analytics service"""
    logger.info("Starting Analytics Service...")
    
    analytics_store.granger_analyzer = GrangerCausalityAnalyzer()
    analytics_store.network_analyzer = NetworkFlowAnalyzer()
    analytics_store.anomaly_detector = AnomalyDetector()
    analytics_store.bias_auditor = BiasAuditor()
    analytics_store.is_initialized = True
    
    logger.info("Analytics Service initialized")
    
    yield
    
    logger.info("Shutting down Analytics Service...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ER Patient Flow - Analytics Service",
    description="Granger causality, network flow, anomaly detection, bias auditing",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy" if analytics_store.is_initialized else "initializing",
        service="analytics-service",
        version="1.0.0",
        timestamp=datetime.now()
    )


# =============================================================================
# GRANGER CAUSALITY ENDPOINTS
# =============================================================================

@app.post("/causality/granger")
async def analyze_granger_causality(request: GrangerRequest):
    """
    Perform Granger causality analysis.
    
    Identifies which metrics "Granger-cause" other metrics.
    """
    logger.info(f"Running Granger causality analysis: {len(request.variables)} variables")
    
    # Build DataFrame
    df = pd.DataFrame(request.data.metrics)
    df['timestamp'] = pd.to_datetime(request.data.timestamps)
    df = df.set_index('timestamp')
    
    # Initialize analyzer
    analyzer = GrangerCausalityAnalyzer(
        max_lag=request.max_lag,
        significance_level=request.significance_level
    )
    
    # Build causal network
    network = analyzer.build_causal_network(df, request.variables)
    analytics_store.last_causal_network = network
    
    # Get summary
    summary = analyzer.get_summary()
    
    return {
        "status": "completed",
        "num_variables": len(request.variables),
        "max_lag_tested": request.max_lag,
        "significance_level": request.significance_level,
        "network": {
            "nodes": network.nodes,
            "edges": network.edges,
            "root_causes": network.root_causes,
            "downstream_effects": network.downstream_effects
        },
        "summary": summary
    }


@app.post("/causality/impulse-response")
async def analyze_impulse_response(request: ImpulseResponseRequest):
    """
    Compute impulse response functions.
    
    Shows how a shock to one variable propagates through the system.
    """
    logger.info(f"Computing impulse response for {request.shock_variable}")
    
    df = pd.DataFrame(request.data.metrics)
    df['timestamp'] = pd.to_datetime(request.data.timestamps)
    df = df.set_index('timestamp')
    
    analyzer = GrangerCausalityAnalyzer()
    responses = analyzer.get_impulse_response(
        df, request.variables, request.shock_variable, request.periods
    )
    
    return {
        "status": "completed",
        "shock_variable": request.shock_variable,
        "periods": request.periods,
        "responses": {
            var: [round(v, 4) for v in vals]
            for var, vals in responses.items()
        }
    }


@app.get("/causality/network")
async def get_causal_network():
    """Get last computed causal network"""
    if analytics_store.last_causal_network is None:
        raise HTTPException(status_code=404, detail="No causal network computed yet")
    
    network = analytics_store.last_causal_network
    return {
        "nodes": network.nodes,
        "edges": network.edges,
        "root_causes": network.root_causes,
        "downstream_effects": network.downstream_effects
    }


# =============================================================================
# NETWORK FLOW ENDPOINTS
# =============================================================================

@app.post("/network/flow")
async def analyze_network_flow(request: NetworkFlowRequest):
    """
    Analyze patient flow network.
    
    Models ED as directed graph with transition probabilities.
    """
    logger.info(f"Analyzing network flow: {len(request.journeys)} journeys")
    
    analyzer = NetworkFlowAnalyzer()
    analyzer.build_network_from_journeys(request.journeys)
    metrics = analyzer.compute_flow_metrics(request.total_hours)
    
    return {
        "status": "completed",
        "network": analyzer.get_network_data(),
        "transition_matrix": analyzer.get_transition_matrix().to_dict()
    }


@app.post("/network/critical-path")
async def find_critical_path(request: NetworkFlowRequest):
    """Find the critical path through the ED"""
    analyzer = NetworkFlowAnalyzer()
    analyzer.build_network_from_journeys(request.journeys)
    analyzer.compute_flow_metrics(request.total_hours)
    
    critical_path = analyzer.find_critical_path()
    
    return {
        "critical_path": critical_path,
        "path_length": len(critical_path),
        "bottlenecks": [
            {
                "location": node.location,
                "mean_time": round(node.mean_time_spent, 1),
                "visits": node.total_visits
            }
            for name, node in analyzer.nodes.items()
            if node.is_bottleneck
        ]
    }


@app.post("/network/simulate-path")
async def simulate_patient_path(
    request: NetworkFlowRequest,
    num_simulations: int = 100
):
    """Simulate patient paths through the network"""
    analyzer = NetworkFlowAnalyzer()
    analyzer.build_network_from_journeys(request.journeys)
    
    paths = []
    times = []
    
    for _ in range(num_simulations):
        path, time = analyzer.simulate_patient_path()
        paths.append(path)
        times.append(time)
    
    # Analyze path patterns
    path_lengths = [len(p) for p in paths]
    
    return {
        "num_simulations": num_simulations,
        "mean_path_length": round(np.mean(path_lengths), 1),
        "mean_total_time": round(np.mean(times), 1),
        "std_total_time": round(np.std(times), 1),
        "sample_paths": paths[:5],
        "time_distribution": {
            "min": round(min(times), 1),
            "p25": round(np.percentile(times, 25), 1),
            "median": round(np.median(times), 1),
            "p75": round(np.percentile(times, 75), 1),
            "max": round(max(times), 1)
        }
    }


# =============================================================================
# ANOMALY DETECTION ENDPOINTS
# =============================================================================

@app.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyRequest):
    """
    Detect anomalies in time series data.
    
    Uses multiple methods: statistical, isolation forest, LOF, temporal.
    """
    logger.info(f"Detecting anomalies: {len(request.metrics)} metrics")
    
    df = pd.DataFrame(request.data.metrics)
    df['timestamp'] = pd.to_datetime(request.data.timestamps)
    
    detector = AnomalyDetector(
        z_threshold=request.z_threshold,
        contamination=request.contamination
    )
    
    # Fit on the data
    detector.fit(df, request.metrics)
    
    # Detect anomalies
    report = detector.detect_all(df, request.metrics)
    analytics_store.last_anomaly_report = report
    
    # Get timeline
    timeline = detector.get_anomaly_timeline(report)
    
    return {
        "status": "completed",
        "period": {
            "start": report.period_start.isoformat(),
            "end": report.period_end.isoformat()
        },
        "summary": {
            "total_anomalies": report.total_anomalies,
            "by_severity": report.by_severity,
            "by_type": report.by_type,
            "by_metric": report.by_metric,
            "health_score": round(report.health_score, 1)
        },
        "top_anomalies": [
            {
                "timestamp": a.timestamp.isoformat(),
                "metric": a.metric,
                "value": round(a.value, 2),
                "expected": round(a.expected_value, 2),
                "deviation": round(a.deviation, 2),
                "type": a.anomaly_type.value,
                "severity": a.severity.value,
                "description": a.description
            }
            for a in report.anomalies[:20]
        ],
        "timeline": timeline
    }


@app.get("/anomaly/report")
async def get_anomaly_report():
    """Get last anomaly detection report"""
    if analytics_store.last_anomaly_report is None:
        raise HTTPException(status_code=404, detail="No anomaly report available")
    
    report = analytics_store.last_anomaly_report
    return {
        "detection_time": report.detection_time.isoformat(),
        "total_anomalies": report.total_anomalies,
        "by_severity": report.by_severity,
        "health_score": round(report.health_score, 1),
        "num_anomalies": len(report.anomalies)
    }


# =============================================================================
# BIAS AUDIT ENDPOINTS
# =============================================================================

@app.post("/bias/audit")
async def run_bias_audit(request: BiasAuditRequest):
    """
    Run comprehensive bias audit on model predictions.
    
    Checks demographic parity, equalized odds, equal opportunity,
    and predictive parity across protected attributes.
    """
    logger.info(f"Running bias audit: {request.model_name}")
    
    y_true = np.array(request.y_true)
    y_pred = np.array(request.y_pred)
    y_prob = np.array(request.y_prob) if request.y_prob else None
    
    protected_df = pd.DataFrame(request.protected_attributes)
    
    auditor = BiasAuditor()
    report = auditor.run_full_audit(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        protected_df=protected_df,
        is_regression=request.is_regression,
        model_name=request.model_name
    )
    
    disparity_summary = auditor.get_disparity_summary()
    
    return {
        "status": "completed",
        "model_name": report.model_name,
        "audit_timestamp": report.audit_timestamp,
        "total_samples": report.total_samples,
        "protected_attributes": report.protected_attributes,
        "overall_fairness_score": round(report.overall_fairness_score, 1),
        "group_metrics": {
            attr: [
                {
                    "group_value": str(m.group_value),
                    "sample_size": m.sample_size,
                    "positive_rate": round(m.positive_rate, 3),
                    "true_positive_rate": round(m.true_positive_rate, 3),
                    "false_positive_rate": round(m.false_positive_rate, 3),
                    "precision": round(m.precision, 3),
                    "f1_score": round(m.f1_score, 3)
                }
                for m in metrics
            ]
            for attr, metrics in report.group_metrics.items()
        },
        "fairness_results": [
            {
                "metric": r.metric.value,
                "attribute": r.protected_attribute,
                "reference_group": r.reference_group,
                "compared_group": r.compared_group,
                "ratio": round(r.ratio, 3),
                "difference": round(r.difference, 3),
                "is_fair": r.is_fair,
                "description": r.description
            }
            for r in report.fairness_results
        ],
        "bias_alerts": report.bias_alerts,
        "recommendations": report.recommendations,
        "disparity_summary": disparity_summary
    }


@app.get("/bias/metrics")
async def get_fairness_metrics():
    """Get available fairness metrics and their descriptions"""
    return {
        "metrics": {
            "demographic_parity": {
                "name": "Demographic Parity",
                "description": "Equal positive prediction rates across groups",
                "formula": "P(Ŷ=1|A=a) = P(Ŷ=1|A=b)"
            },
            "equalized_odds": {
                "name": "Equalized Odds",
                "description": "Equal TPR and FPR across groups",
                "formula": "TPR_a = TPR_b AND FPR_a = FPR_b"
            },
            "equal_opportunity": {
                "name": "Equal Opportunity",
                "description": "Equal TPR across groups",
                "formula": "P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)"
            },
            "predictive_parity": {
                "name": "Predictive Parity",
                "description": "Equal precision across groups",
                "formula": "P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b)"
            }
        },
        "thresholds": {
            "ratio_threshold": BiasAuditor.DEFAULT_RATIO_THRESHOLD,
            "difference_threshold": BiasAuditor.DEFAULT_DIFF_THRESHOLD
        }
    }


# =============================================================================
# DEMO DATA ENDPOINTS
# =============================================================================

@app.get("/demo/time-series")
async def get_demo_time_series(hours: int = 168):
    """Generate demo time series data for testing"""
    np.random.seed(42)
    
    timestamps = [
        (datetime.now() - timedelta(hours=hours-i)).isoformat()
        for i in range(hours)
    ]
    
    # Generate correlated metrics
    base = np.cumsum(np.random.randn(hours)) * 0.5
    
    census = 30 + base + np.random.randn(hours) * 3
    wait_time = 20 + base * 0.8 + np.random.randn(hours) * 5
    arrivals = 15 + np.sin(np.arange(hours) * 2 * np.pi / 24) * 5 + np.random.randn(hours) * 2
    los = 180 + base * 10 + np.random.randn(hours) * 20
    
    # Add some anomalies
    census[50] = 80
    wait_time[100] = 100
    
    return {
        "timestamps": timestamps,
        "metrics": {
            "census": census.tolist(),
            "wait_time": wait_time.tolist(),
            "arrivals": arrivals.tolist(),
            "los": los.tolist()
        }
    }


@app.get("/demo/journeys")
async def get_demo_journeys(num_patients: int = 100):
    """Generate demo patient journeys for testing"""
    np.random.seed(42)
    
    locations = ['arrival', 'triage', 'waiting_room', 'ed_bed', 'laboratory', 
                 'imaging', 'treatment', 'observation', 'discharge']
    
    journeys = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(num_patients):
        journey = {
            'patient_id': f'P{i:04d}',
            'events': []
        }
        
        current_time = base_time + timedelta(hours=np.random.exponential(2))
        
        # Standard path with variations
        path = ['arrival', 'triage', 'waiting_room', 'ed_bed']
        
        # Add lab/imaging for some patients
        if np.random.random() > 0.4:
            path.append('laboratory')
        if np.random.random() > 0.6:
            path.append('imaging')
        
        path.append('treatment')
        
        # Some go to observation
        if np.random.random() > 0.7:
            path.append('observation')
        
        path.append('discharge')
        
        for loc in path:
            journey['events'].append({
                'location': loc,
                'timestamp': current_time.isoformat()
            })
            current_time += timedelta(minutes=np.random.exponential(30))
        
        journeys.append(journey)
    
    return journeys


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
