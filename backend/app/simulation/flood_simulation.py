import numpy as np

def calculate_water_depth(runoff: np.ndarray, dem: np.ndarray, flow_direction: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Estimates water depth by propagating runoff across the terrain using D8 flow directions.
    Simulates water spreading iteratively, moving water to equalize water surface elevation
    with downstream neighbors.

    Args:
        runoff (np.ndarray): Initial surface runoff volume/depth per cell.
        dem (np.ndarray): Elevation array.
        flow_direction (np.ndarray): D8 flow direction array.
        iterations (int): Number of iterations for the simulation.

    Returns:
        np.ndarray: Estimated water depth per cell.
    """
    if runoff.shape != dem.shape or runoff.shape != flow_direction.shape:
        raise ValueError("All input arrays must have the same shape.")

    depth = np.array(runoff, dtype=float, copy=True)
    rows, cols = depth.shape

    # Mapping for D8 encoded directions
    dr = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 0, 8: -1}
    dc = {1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: -1, 7: -1, 8: -1}

    # Precompute target row/col for each cell based on flow_direction
    target_r = np.arange(rows)[:, None] + np.zeros(cols, dtype=int)
    target_c = np.arange(cols)[None, :] + np.zeros((rows, 1), dtype=int)

    for d in range(1, 9):
        mask = (flow_direction == d)
        # Clip targets to ensure boundary cells don't route out of bounds
        target_r[mask] = np.clip(target_r[mask] + dr[d], 0, rows - 1)
        target_c[mask] = np.clip(target_c[mask] + dc[d], 0, cols - 1)

    # Iterative water routing
    for _ in range(iterations):
        # Calculate water surface elevation (WSE)
        wse = dem + depth
        target_wse = wse[target_r, target_c]

        # Calculate potential transfer to equalize WSE with target cell
        diff = wse - target_wse
        transfer = np.clip(diff / 2.0, 0, depth)

        # Don't transfer from cells that have no flow direction (sinks)
        transfer[flow_direction == 0] = 0

        # Apply transfer: subtract from source, add to target
        depth -= transfer
        np.add.at(depth, (target_r, target_c), transfer)

    return depth

def detect_flooded_cells(water_depth: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Detects cells that are considered flooded based on a depth threshold.
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    return water_depth >= threshold

def calculate_flood_risk(water_depth: np.ndarray) -> np.ndarray:
    """
    Normalizes water depth to a risk value between 0 and 1.
    """
    max_depth = np.max(water_depth)
    if max_depth == 0:
        return np.zeros_like(water_depth, dtype=float)
    return water_depth / max_depth

def detect_first_pooling(water_depth: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """
    Identifies local terrain depressions where water depth is positive.
    This is a prototype approximation to highlight initial pooling zones.
    """
    padded_dem = np.pad(dem, pad_width=1, mode='constant', constant_values=np.inf)

    is_depression = np.ones_like(dem, dtype=bool)

    neighbors = [
        padded_dem[0:-2, 1:-1],    # N
        padded_dem[0:-2, 2:],      # NE
        padded_dem[1:-1, 2:],      # E
        padded_dem[2:, 2:],        # SE
        padded_dem[2:, 1:-1],      # S
        padded_dem[2:, 0:-2],      # SW
        padded_dem[1:-1, 0:-2],    # W
        padded_dem[0:-2, 0:-2],    # NW
    ]

    # A cell is a depression if its elevation is <= all of its 8 neighbors
    for n_dem in neighbors:
        is_depression &= (dem <= n_dem)

    return is_depression & (water_depth > 0)
