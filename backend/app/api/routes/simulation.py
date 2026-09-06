from fastapi import APIRouter, HTTPException, Depends
from typing import Any
import numpy as np
import os

from app.models.simulation import SimulationRequest, SimulationResponse
from app.services.dem_service import read_dem
from app.simulation.terrain_analysis import (
    calculate_slope,
    calculate_d8_flow_direction,
    calculate_flow_accumulation
)
from app.simulation.runoff import calculate_runoff
from app.simulation.flood_simulation import (
    calculate_water_depth,
    detect_flooded_cells,
    calculate_flood_risk,
    detect_first_pooling
)
from app.simulation.landslide_risk import (
    calculate_landslide_risk,
    classify_landslide_risk,
    detect_high_risk_zones
)

router = APIRouter()

@router.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest) -> Any:
    """
    Run a simulation for flood and landslide risk.
    """
    if request.dem_file_path is None:
        raise HTTPException(status_code=400, detail="dem_file_path is required for simulation.")

    if not os.path.exists(request.dem_file_path):
        raise HTTPException(status_code=404, detail="DEM file not found.")

    try:
        dem, metadata = read_dem(request.dem_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading DEM: {str(e)}")

    try:
        nodata = metadata.get("nodata", None)
        transform = metadata["transform"]

        # Terrain Analysis
        slope = calculate_slope(dem, transform, nodata)
        flow_direction = calculate_d8_flow_direction(dem, nodata)
        flow_accumulation = calculate_flow_accumulation(flow_direction)

        # Runoff
        runoff = calculate_runoff(
            request.rainfall_mm,
            request.infiltration_rate,
            flow_accumulation
        )

        # Flood Simulation
        water_depth = calculate_water_depth(runoff, dem, flow_direction)
        flooded_cells = detect_flooded_cells(water_depth, request.water_threshold)
        flood_risk = calculate_flood_risk(water_depth)
        first_pooling = detect_first_pooling(water_depth, dem)

        # Landslide Risk
        landslide_risk = calculate_landslide_risk(
            slope=slope,
            rainfall_mm=request.rainfall_mm,
            water_depth=water_depth,
            slope_threshold=request.slope_threshold,
            rainfall_threshold=request.rainfall_threshold,
            water_threshold=request.water_threshold
        )
        # Classify and detect high risk zones
        high_risk_zones = detect_high_risk_zones(landslide_risk, 0.5)

        return SimulationResponse(
            status="success",
            rainfall_mm=request.rainfall_mm,
            max_water_depth=float(np.max(water_depth)),
            flooded_cell_count=int(np.sum(flooded_cells)),
            first_pooling_cell_count=int(np.sum(first_pooling)),
            flood_risk_max=float(np.max(flood_risk)),
            flood_risk_mean=float(np.mean(flood_risk)),
            landslide_risk_max=float(np.max(landslide_risk)),
            landslide_risk_mean=float(np.mean(landslide_risk)),
            high_risk_landslide_cell_count=int(np.sum(high_risk_zones))
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")
