import os
import rasterio
import numpy as np
from typing import Dict, Any, Tuple

def read_dem(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a DEM (Digital Elevation Model) GeoTIFF file.

    Args:
        file_path (str): The path to the DEM GeoTIFF file.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the elevation data as a NumPy array
        and a dictionary with metadata (width, height, crs, transform, bounds, min_elevation, max_elevation).

    Raises:
        FileNotFoundError: If the provided file_path does not exist.
        rasterio.errors.RasterioIOError: If the file is not a valid GeoTIFF or cannot be opened.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DEM file not found: {file_path}")

    with rasterio.open(file_path) as src:
        if src.crs is None:
            raise ValueError("DEM file is missing CRS information.")
        if src.crs.is_geographic:
            raise ValueError(f"Geographic CRS ({src.crs.to_string()}) is not supported. A projected CRS with metre-based units is required.")
        horizontal_unit, horizontal_unit_factor = src.crs.linear_units_factor
        if not np.isclose(horizontal_unit_factor, 1.0):
            raise ValueError(
                f"Projected CRS ({src.crs.to_string()}) uses {horizontal_unit} horizontal units. "
                "A projected CRS with metre-based horizontal units is required."
            )

        # Read the first band
        elevation_data = src.read(1)

        # Handle NoData values for min/max calculation
        nodata_val = src.nodata
        if nodata_val is not None:
            valid_data = elevation_data[elevation_data != nodata_val]
        else:
            valid_data = elevation_data

        if valid_data.size > 0:
            min_elevation = float(np.min(valid_data))
            max_elevation = float(np.max(valid_data))
        else:
            min_elevation = None
            max_elevation = None

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "min_elevation": min_elevation,
            "max_elevation": max_elevation,
            "nodata": nodata_val
        }

        return elevation_data, metadata
