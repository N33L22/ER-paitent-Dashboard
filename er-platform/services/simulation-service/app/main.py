"""
Simulation Service - FastAPI Application
SimPy discrete-event simulation, what-if scenarios, Monte Carlo
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from .discrete_event_sim import EDSimulation, SimulationConfig, SimulationResults
from .scenario_engine import ScenarioEngine, ScenarioDefinition
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class SimulationRequest(BaseModel):
    """Request for running a simulation"""
    num_triage_nurses: int = Field(default=2, ge=1, le=10)
    num_physicians: int = Field(default=4, ge=1, le=20)
    num_nurses: int = Field(default=8, ge=1, le=40)
    num_beds: int = Field(default=20, ge=5, le=100)
    num_lab_techs: int = Field(default=2, ge=1, le=10)
    num_imaging_techs: int = Field(default=2, ge=1, le=10)
    simulation_hours: int = Field(default=168, ge=24, le=720)
    arrival_rate_per_hour: float = Field(default=20.0, ge=5, le=100)
    arrival_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    seed: int = Field(default=42, ge=0)


class ScenarioRequest(BaseModel):
    """Request for running a scenario"""
    name: str
    description: str = ""
    physicians_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    nurses_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    beds_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)
    arrival_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    simulation_hours: int = Field(default=168, ge=24, le=720)


class MatrixRequest(BaseModel):
    """Request for running a scenario matrix"""
    staffing_levels: List[float] = Field(
        default=[0.8, 1.0, 1.2],
        min_length=2,
        max_length=6
    )
    arrival_levels: List[float] = Field(
        default=[0.9, 1.0, 1.1, 1.2],
        min_length=2,
        max_length=6
    )
    simulation_hours: int = Field(default=168, ge=24, le=720)


class MonteCarloRequest(BaseModel):
    """Request for Monte Carlo simulation"""
    num_runs: int = Field(default=100, ge=10, le=500)
    base_config: Optional[SimulationRequest] = None


class SimulationResponse(BaseModel):
    """Response from simulation"""
    status: str
    simulation_hours: int
    total_patients: int
    completed_patients: int
    lwbs_count: int
    mean_los_minutes: float
    median_los_minutes: float
    p90_los_minutes: float
    mean_wait_to_triage: float
    mean_wait_to_bed: float
    mean_wait_to_physician: float
    bed_utilization: float
    lwbs_rate: float
    los_by_acuity: Dict[int, float]
    count_by_acuity: Dict[int, int]
    hourly_census: List[int]
    hourly_arrivals: List[int]


class HealthCheck(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime


# =============================================================================
# GLOBAL STATE
# =============================================================================

class SimulationStore:
    """Store for simulation state and results"""
    def __init__(self):
        self.last_results: Optional[SimulationResults] = None
        self.scenario_engine: Optional[ScenarioEngine] = None
        self.is_initialized: bool = False


sim_store = SimulationStore()


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize simulation service"""
    logger.info("Starting Simulation Service...")
    
    # Initialize scenario engine with default config
    sim_store.scenario_engine = ScenarioEngine()
    sim_store.is_initialized = True
    
    logger.info("Simulation Service initialized")
    
    yield
    
    logger.info("Shutting down Simulation Service...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ER Patient Flow - Simulation Service",
    description="SimPy discrete-event simulation, what-if scenarios, Monte Carlo",
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
        status="healthy" if sim_store.is_initialized else "initializing",
        service="simulation-service",
        version="1.0.0",
        timestamp=datetime.now()
    )


# =============================================================================
# SIMULATION ENDPOINTS
# =============================================================================

@app.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a single ED simulation"""
    logger.info(f"Running simulation: {request.simulation_hours}h, {request.num_beds} beds")
    
    # Create config
    config = SimulationConfig(
        num_triage_nurses=request.num_triage_nurses,
        num_physicians=request.num_physicians,
        num_nurses=request.num_nurses,
        num_beds=request.num_beds,
        num_lab_techs=request.num_lab_techs,
        num_imaging_techs=request.num_imaging_techs,
        simulation_hours=request.simulation_hours,
        arrival_rate_per_hour=request.arrival_rate_per_hour,
        arrival_multiplier=request.arrival_multiplier,
        seed=request.seed
    )
    
    # Run simulation
    sim = EDSimulation(config)
    results = sim.run()
    
    # Store results
    sim_store.last_results = results
    
    # Build response
    return SimulationResponse(
        status="completed",
        simulation_hours=config.simulation_hours,
        total_patients=results.total_patients,
        completed_patients=results.completed_patients,
        lwbs_count=results.lwbs_count,
        mean_los_minutes=round(results.mean_los_minutes, 1),
        median_los_minutes=round(results.median_los_minutes, 1),
        p90_los_minutes=round(results.p90_los_minutes, 1),
        mean_wait_to_triage=round(results.mean_wait_to_triage, 1),
        mean_wait_to_bed=round(results.mean_wait_to_bed, 1),
        mean_wait_to_physician=round(results.mean_wait_to_physician, 1),
        bed_utilization=round(results.bed_utilization, 3),
        lwbs_rate=round(results.lwbs_count / max(1, results.total_patients) * 100, 2),
        los_by_acuity=results.los_by_acuity,
        count_by_acuity=results.count_by_acuity,
        hourly_census=results.hourly_census,
        hourly_arrivals=results.hourly_arrivals
    )


@app.get("/simulate/quick")
async def run_quick_simulation(
    physicians: int = 4,
    nurses: int = 8,
    beds: int = 20,
    arrival_multiplier: float = 1.0,
    hours: int = 24
):
    """Run a quick simulation with minimal parameters"""
    request = SimulationRequest(
        num_physicians=physicians,
        num_nurses=nurses,
        num_beds=beds,
        arrival_multiplier=arrival_multiplier,
        simulation_hours=hours
    )
    return await run_simulation(request)


# =============================================================================
# SCENARIO ENDPOINTS
# =============================================================================

@app.post("/scenario")
async def run_scenario(request: ScenarioRequest):
    """Run a custom scenario"""
    if sim_store.scenario_engine is None:
        raise HTTPException(status_code=503, detail="Scenario engine not initialized")
    
    scenario = ScenarioDefinition(
        name=request.name,
        description=request.description,
        physicians_multiplier=request.physicians_multiplier,
        nurses_multiplier=request.nurses_multiplier,
        beds_multiplier=request.beds_multiplier,
        arrival_multiplier=request.arrival_multiplier,
        simulation_hours=request.simulation_hours
    )
    
    result = sim_store.scenario_engine.run_scenario(scenario)
    
    return {
        "scenario_name": result.scenario.name,
        "description": result.scenario.description,
        "results": {
            "total_patients": result.results.total_patients,
            "mean_los_minutes": round(result.results.mean_los_minutes, 1),
            "median_los_minutes": round(result.results.median_los_minutes, 1),
            "p90_los_minutes": round(result.results.p90_los_minutes, 1),
            "mean_wait_to_bed": round(result.results.mean_wait_to_bed, 1),
            "lwbs_count": result.results.lwbs_count,
            "lwbs_rate": round(result.results.lwbs_count / max(1, result.results.total_patients) * 100, 2)
        },
        "comparison": {
            "los_change_percent": round(result.los_change_percent, 1) if result.los_change_percent else None,
            "wait_change_percent": round(result.wait_change_percent, 1) if result.wait_change_percent else None
        }
    }


@app.get("/scenario/presets")
async def get_preset_scenarios():
    """Get list of available preset scenarios"""
    return {
        name: {
            "description": scenario.description,
            "physicians_multiplier": scenario.physicians_multiplier,
            "nurses_multiplier": scenario.nurses_multiplier,
            "beds_multiplier": scenario.beds_multiplier,
            "arrival_multiplier": scenario.arrival_multiplier
        }
        for name, scenario in ScenarioEngine.PRESET_SCENARIOS.items()
    }


@app.post("/scenario/preset/{preset_name}")
async def run_preset_scenario(preset_name: str):
    """Run a predefined scenario"""
    if sim_store.scenario_engine is None:
        raise HTTPException(status_code=503, detail="Scenario engine not initialized")
    
    if preset_name not in ScenarioEngine.PRESET_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_name}")
    
    result = sim_store.scenario_engine.run_preset(preset_name)
    
    return {
        "scenario_name": result.scenario.name,
        "description": result.scenario.description,
        "results": {
            "total_patients": result.results.total_patients,
            "mean_los_minutes": round(result.results.mean_los_minutes, 1),
            "median_los_minutes": round(result.results.median_los_minutes, 1),
            "p90_los_minutes": round(result.results.p90_los_minutes, 1),
            "mean_wait_to_bed": round(result.results.mean_wait_to_bed, 1),
            "lwbs_count": result.results.lwbs_count
        },
        "comparison": {
            "los_change_percent": round(result.los_change_percent, 1) if result.los_change_percent else None,
            "wait_change_percent": round(result.wait_change_percent, 1) if result.wait_change_percent else None
        }
    }


@app.post("/scenario/compare-all")
async def run_all_preset_scenarios():
    """Run all preset scenarios and compare"""
    if sim_store.scenario_engine is None:
        raise HTTPException(status_code=503, detail="Scenario engine not initialized")
    
    # Create fresh engine
    engine = ScenarioEngine()
    engine.run_all_presets()
    
    return {
        "scenarios": engine.get_comparison_summary(),
        "baseline": {
            "mean_los": round(engine.baseline_results.mean_los_minutes, 1),
            "mean_wait": round(engine.baseline_results.mean_wait_to_bed, 1),
            "total_patients": engine.baseline_results.total_patients
        }
    }


# =============================================================================
# SCENARIO MATRIX ENDPOINTS
# =============================================================================

@app.post("/scenario/matrix")
async def run_scenario_matrix(request: MatrixRequest):
    """Run a matrix of scenarios (staffing × arrivals)"""
    logger.info(f"Running scenario matrix: {len(request.staffing_levels)}x{len(request.arrival_levels)}")
    
    engine = ScenarioEngine()
    matrix_results = engine.run_scenario_matrix(
        staffing_multipliers=request.staffing_levels,
        arrival_multipliers=request.arrival_levels,
        simulation_hours=request.simulation_hours
    )
    
    # Build results grid
    grid = []
    for (staff, arrival), result in matrix_results.items():
        grid.append({
            "staffing_level": staff,
            "arrival_level": arrival,
            "mean_los": round(result.results.mean_los_minutes, 1),
            "median_los": round(result.results.median_los_minutes, 1),
            "p90_los": round(result.results.p90_los_minutes, 1),
            "mean_wait": round(result.results.mean_wait_to_bed, 1),
            "lwbs_rate": round(result.results.lwbs_count / max(1, result.results.total_patients) * 100, 2)
        })
    
    # Heatmap data
    heatmap = engine.get_matrix_heatmap_data(matrix_results)
    
    return {
        "grid": grid,
        "heatmap": heatmap,
        "staffing_levels": request.staffing_levels,
        "arrival_levels": request.arrival_levels
    }


# =============================================================================
# MONTE CARLO ENDPOINTS
# =============================================================================

@app.post("/monte-carlo")
async def run_monte_carlo(request: MonteCarloRequest):
    """Run Monte Carlo simulation for uncertainty quantification"""
    logger.info(f"Running Monte Carlo simulation: {request.num_runs} runs")
    
    # Build base config
    if request.base_config:
        base = SimulationConfig(
            num_triage_nurses=request.base_config.num_triage_nurses,
            num_physicians=request.base_config.num_physicians,
            num_nurses=request.base_config.num_nurses,
            num_beds=request.base_config.num_beds,
            num_lab_techs=request.base_config.num_lab_techs,
            num_imaging_techs=request.base_config.num_imaging_techs,
            simulation_hours=request.base_config.simulation_hours,
            arrival_rate_per_hour=request.base_config.arrival_rate_per_hour,
            arrival_multiplier=request.base_config.arrival_multiplier
        )
    else:
        base = SimulationConfig()
    
    # Run Monte Carlo
    mc_config = MonteCarloConfig(
        num_runs=request.num_runs,
        base_config=base
    )
    
    simulator = MonteCarloSimulator(mc_config)
    results = simulator.run()
    
    # Risk analysis
    risk = simulator.get_risk_analysis()
    
    # Distribution data
    distributions = simulator.get_distribution_data()
    
    return {
        "num_runs": results.num_runs,
        "summary": {
            "los": {
                "mean": round(results.los_mean, 1),
                "std": round(results.los_std, 1),
                "p5": round(results.los_p5, 1),
                "p50": round(results.los_p50, 1),
                "p95": round(results.los_p95, 1),
                "ci_95": [round(x, 1) for x in results.los_ci_95]
            },
            "wait": {
                "mean": round(results.wait_mean, 1),
                "std": round(results.wait_std, 1),
                "p5": round(results.wait_p5, 1),
                "p95": round(results.wait_p95, 1),
                "ci_95": [round(x, 1) for x in results.wait_ci_95]
            },
            "lwbs_rate": {
                "mean": round(results.lwbs_rate_mean, 2),
                "std": round(results.lwbs_rate_std, 2)
            }
        },
        "risk_analysis": risk,
        "distributions": {
            "los_values": results.mean_los_distribution,
            "wait_values": results.mean_wait_distribution,
            "lwbs_values": results.lwbs_rate_distribution
        }
    }


@app.get("/monte-carlo/quick")
async def run_quick_monte_carlo(runs: int = 50):
    """Run quick Monte Carlo with default config"""
    request = MonteCarloRequest(num_runs=runs)
    return await run_monte_carlo(request)


# =============================================================================
# RESULTS ENDPOINTS
# =============================================================================

@app.get("/results/last")
async def get_last_results():
    """Get results from last simulation"""
    if sim_store.last_results is None:
        raise HTTPException(status_code=404, detail="No simulation results available")
    
    results = sim_store.last_results
    return {
        "total_patients": results.total_patients,
        "mean_los_minutes": round(results.mean_los_minutes, 1),
        "median_los_minutes": round(results.median_los_minutes, 1),
        "p90_los_minutes": round(results.p90_los_minutes, 1),
        "mean_wait_to_bed": round(results.mean_wait_to_bed, 1),
        "lwbs_count": results.lwbs_count,
        "los_by_acuity": results.los_by_acuity,
        "hourly_census": results.hourly_census,
        "hourly_arrivals": results.hourly_arrivals
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
