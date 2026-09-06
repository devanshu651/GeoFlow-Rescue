import numpy as np
from collections import deque

def calculate_slope(dem: np.ndarray, transform, nodata=None) -> np.ndarray:
    """
    Calculates the terrain slope from a DEM in degrees.

    Args:
        dem (np.ndarray): 2D numpy array representing the elevation data.
        transform (rasterio.Affine or tuple): Raster transform to determine cell dimensions.
        nodata: Value representing NoData in the DEM.

    Returns:
        np.ndarray: 2D numpy array of slope in degrees.
    """
    dem_float = np.array(dem, dtype=float, copy=True)
    if nodata is not None:
        dem_float[dem == nodata] = np.nan

    dx = abs(transform[0])
    dy = abs(transform[4])

    # Calculate gradients using central differences
    dz_dy, dz_dx = np.gradient(dem_float, dy, dx)

    # Slope magnitude
    slope_mag = np.sqrt(dz_dx**2 + dz_dy**2)

    # Convert to degrees
    slope_deg = np.degrees(np.arctan(slope_mag))

    return slope_deg


def calculate_d8_flow_direction(dem: np.ndarray, nodata=None) -> np.ndarray:
    """
    Calculates the standard D8 flow direction for a DEM.

    Encoding:
    1 = N, 2 = NE, 3 = E, 4 = SE, 5 = S, 6 = SW, 7 = W, 8 = NW, 0 = sink/no flow

    Note: The current D8 implementation assumes square cells / equal x and y cell dimensions,
    meaning distances to orthogonal neighbors are 1.0 and to diagonal neighbors are sqrt(2).
    This is a prototype assumption.

    Args:
        dem (np.ndarray): 2D numpy array of elevation.
        nodata: Value representing NoData.

    Returns:
        np.ndarray: 2D array of flow directions.
    """
    dem_float = np.array(dem, dtype=float, copy=True)
    if nodata is not None:
        dem_float[dem == nodata] = np.nan

    # Pad DEM with infinity to handle edges safely without out of bounds
    padded_dem = np.pad(dem_float, pad_width=1, mode='constant', constant_values=np.inf)

    z0 = dem_float
    sqrt2 = np.sqrt(2)

    # (slice_r, slice_c, direction, dist)
    neighbors = [
        (padded_dem[0:-2, 1:-1], 1, 1.0),    # N
        (padded_dem[0:-2, 2:], 2, sqrt2),    # NE
        (padded_dem[1:-1, 2:], 3, 1.0),      # E
        (padded_dem[2:, 2:], 4, sqrt2),      # SE
        (padded_dem[2:, 1:-1], 5, 1.0),      # S
        (padded_dem[2:, 0:-2], 6, sqrt2),    # SW
        (padded_dem[1:-1, 0:-2], 7, 1.0),    # W
        (padded_dem[0:-2, 0:-2], 8, sqrt2),  # NW
    ]

    max_slope = np.zeros_like(dem_float, dtype=float)
    flow_dir = np.zeros_like(dem_float, dtype=np.uint8)

    for z1, dir_code, dist in neighbors:
        drop = z0 - z1
        slope = drop / dist

        # We look for strictly greater slope to update (first tie wins)
        mask = (slope > max_slope) & (~np.isnan(slope))
        max_slope[mask] = slope[mask]
        flow_dir[mask] = dir_code

    if nodata is not None:
        flow_dir[np.isnan(dem_float)] = 0

    return flow_dir


def calculate_flow_accumulation(flow_direction: np.ndarray) -> np.ndarray:
    """
    Calculates the flow accumulation for a given flow direction grid.
    Uses Kahn's topological sorting algorithm to safely avoid infinite loops
    that could be caused by cycles or sinks in flat terrain.

    Args:
        flow_direction (np.ndarray): 2D array of flow directions.

    Returns:
        np.ndarray: 2D array of flow accumulation values (number of upstream cells).
    """
    rows, cols = flow_direction.shape
    in_degree = np.zeros((rows, cols), dtype=int)

    dr = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: -1}
    dc = {1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: -1, 7: -1, 8: -1}

    # Calculate in-degrees for each cell
    for r in range(rows):
        for c in range(cols):
            d = flow_direction[r, c]
            if d in dr:
                nr, nc = r + dr[d], c + dc[d]
                if 0 <= nr < rows and 0 <= nc < cols:
                    in_degree[nr, nc] += 1

    # Initialize accumulation array to 0
    accumulation = np.zeros((rows, cols), dtype=int)

    queue = deque()

    # Start with cells that have no incoming flow
    for r in range(rows):
        for c in range(cols):
            if in_degree[r, c] == 0:
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        d = flow_direction[r, c]

        if d in dr:
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < rows and 0 <= nc < cols:
                # Add current cell's accumulation plus 1 for the cell itself
                accumulation[nr, nc] += accumulation[r, c] + 1

                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue.append((nr, nc))

    return accumulation
