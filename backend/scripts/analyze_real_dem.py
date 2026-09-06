import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.services.dem_service import read_dem
from app.simulation.terrain_analysis import (
    calculate_slope,
    calculate_d8_flow_direction,
    calculate_flow_accumulation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = PROJECT_ROOT / "data" / "dem" / "n26_e086_aoi_utm45.tif"


print("Loading real DEM...")

dem, metadata = read_dem(str(DEM_PATH))

print("DEM shape:", dem.shape)
print("DEM CRS:", metadata["crs"])
print("DEM resolution:", metadata["transform"].a, "m")


# Pixel size
cell_size = metadata["transform"].a


print("\nCalculating slope...")

slope = calculate_slope(
    dem,
    metadata["transform"],
)

print("Slope calculated.")
print("Minimum slope:", np.nanmin(slope))
print("Maximum slope:", np.nanmax(slope))
print("Mean slope:", np.nanmean(slope))


print("\nCalculating D8 flow direction...")

flow_direction = calculate_d8_flow_direction(dem)

print("Flow direction calculated.")

valid_flow = flow_direction[flow_direction > 0]

print("Valid flow cells:", valid_flow.size)


print("\nCalculating flow accumulation...")

flow_accumulation = calculate_flow_accumulation(
    flow_direction
)

print("Flow accumulation calculated.")

print(
    "Maximum flow accumulation:",
    np.nanmax(flow_accumulation),
)

print(
    "Mean flow accumulation:",
    np.nanmean(flow_accumulation),
)


print("\n================================")
print("REAL TERRAIN ANALYSIS COMPLETE")
print("================================")