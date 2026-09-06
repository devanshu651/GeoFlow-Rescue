"""Run the prototype GeoFlow-Rescue simulation on the processed real DEM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Dict, Tuple

import numpy as np
import rasterio

from app.services.dem_service import read_dem
from app.simulation.flood_simulation import (
    calculate_flood_risk,
    calculate_water_depth,
    detect_first_pooling,
    detect_flooded_cells,
)
from app.simulation.landslide_risk import (
    calculate_landslide_risk,
    detect_high_risk_zones,
)
from app.simulation.runoff import calculate_runoff
from app.simulation.terrain_analysis import (
    calculate_d8_flow_direction,
    calculate_flow_accumulation,
    calculate_slope,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEM_PATH = PROJECT_ROOT / "data" / "dem" / "n26_e086_aoi_utm45.tif"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

RAINFALL_MM = 100.0  # Prototype scenario; this is not a weather forecast.
INFILTRATION_RATE = 0.2
WATER_THRESHOLD_M = 0.05
SLOPE_THRESHOLD_DEG = 30.0
RAINFALL_THRESHOLD_MM = 50.0


def _write_raster(
    path: Path,
    values: np.ndarray,
    source_profile: Dict,
    valid_mask: np.ndarray,
    dtype: str,
    nodata,
) -> None:
    """Write one derived raster with the source grid and an explicit nodata value."""
    output = np.asarray(values).astype(dtype, copy=True)
    output[~valid_mask] = nodata
    profile = source_profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output, 1)


def run_real_simulation(
    dem_path: Path = DEM_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    """Run and persist the prototype flood/landslide simulation."""
    dem, metadata = read_dem(str(dem_path))
    dem_for_simulation = np.asarray(dem, dtype=float)
    with rasterio.open(dem_path) as src:
        source_profile = src.profile
        source_nodata = src.nodata

    valid_mask = np.ones(dem.shape, dtype=bool)
    if source_nodata is not None:
        valid_mask = dem != source_nodata

    output_dir.mkdir(parents=True, exist_ok=True)

    slope = calculate_slope(dem_for_simulation, metadata["transform"], metadata.get("nodata"))
    flow_direction = calculate_d8_flow_direction(dem_for_simulation, metadata.get("nodata"))
    flow_accumulation = calculate_flow_accumulation(flow_direction)
    runoff = calculate_runoff(RAINFALL_MM, INFILTRATION_RATE, flow_accumulation)
    water_depth = calculate_water_depth(runoff, dem_for_simulation, flow_direction)
    flooded_cells = detect_flooded_cells(water_depth, WATER_THRESHOLD_M)
    flood_risk = calculate_flood_risk(water_depth)
    first_pooling = detect_first_pooling(water_depth, dem_for_simulation)
    landslide_risk = calculate_landslide_risk(
        slope,
        RAINFALL_MM,
        water_depth,
        slope_threshold=SLOPE_THRESHOLD_DEG,
        rainfall_threshold=RAINFALL_THRESHOLD_MM,
        water_threshold=WATER_THRESHOLD_M,
    )

    float_layers = {
        "slope.tif": slope,
        "runoff.tif": runoff,
        "water_depth.tif": water_depth,
        "flood_risk.tif": flood_risk,
        "landslide_risk.tif": landslide_risk,
    }
    for filename, values in float_layers.items():
        _write_raster(output_dir / filename, values, source_profile, valid_mask, "float32", -9999.0)
    _write_raster(output_dir / "flow_direction.tif", flow_direction, source_profile, valid_mask, "uint8", 0)
    _write_raster(output_dir / "flow_accumulation.tif", flow_accumulation, source_profile, valid_mask, "int32", -2147483648)
    _write_raster(output_dir / "flooded_cells.tif", flooded_cells, source_profile, valid_mask, "uint8", 255)

    first_pooling_locations = np.argwhere(first_pooling & valid_mask)
    return {
        "dimensions": dem.shape,
        "rainfall_mm": RAINFALL_MM,
        "slope": slope,
        "flow_accumulation": flow_accumulation,
        "runoff": runoff,
        "water_depth": water_depth,
        "flooded_cells": flooded_cells,
        "flood_risk": flood_risk,
        "landslide_risk": landslide_risk,
        "first_pooling_locations": first_pooling_locations,
    }


def main() -> None:
    print("Loading real DEM...")
    print("Prototype scenario: 100 mm rainfall; not a weather forecast.")
    result = run_real_simulation()
    first_pooling = result["first_pooling_locations"]
    first_location = tuple(int(value) for value in first_pooling[0]) if len(first_pooling) else None

    print("\nREAL SIMULATION SUMMARY")
    print("DEM dimensions:", result["dimensions"])
    print("Rainfall (mm):", result["rainfall_mm"])
    print("Minimum slope (degrees):", np.nanmin(result["slope"]))
    print("Maximum slope (degrees):", np.nanmax(result["slope"]))
    print("Mean slope (degrees):", np.nanmean(result["slope"]))
    print("Maximum flow accumulation:", np.max(result["flow_accumulation"]))
    print("Total runoff (m):", np.sum(result["runoff"]))
    print("Maximum water depth (m):", np.max(result["water_depth"]))
    print("Flooded cells:", np.sum(result["flooded_cells"]))
    print("First pooling location (row, col):", first_location)
    print("High flood-risk cells:", np.sum(result["flood_risk"] >= 0.5))
    print("High landslide-risk cells:", np.sum(result["landslide_risk"] >= 0.5))
    print("Outputs written to:", OUTPUT_DIR)
    print("\nPrototype limitation: outputs are not validated operational flood forecasts.")


if __name__ == "__main__":
    main()
