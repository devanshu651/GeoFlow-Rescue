import numpy as np
from typing import Union

def calculate_runoff(
    rainfall_mm: Union[float, np.ndarray],
    infiltration_rate: float,
    flow_accumulation: np.ndarray
) -> np.ndarray:
    """
    Calculates surface runoff based on rainfall, infiltration, and flow accumulation.

    Args:
        rainfall_mm (float or np.ndarray): Rainfall amount in millimeters.
            Can be a single uniform value or a 2D array matching terrain dimensions.
        infiltration_rate (float): Fraction of rainfall that infiltrates the ground (0.0 to 1.0).
        flow_accumulation (np.ndarray): 2D array of upstream contributing cells.

    Returns:
        np.ndarray: Estimated surface runoff volume/depth per cell.

    Raises:
        ValueError: If infiltration_rate is not between 0.0 and 1.0.
        ValueError: If rainfall_mm is an array but its shape doesn't match flow_accumulation.
    """
    if not (0.0 <= infiltration_rate <= 1.0):
        raise ValueError("infiltration_rate must be between 0.0 and 1.0.")

    if isinstance(rainfall_mm, np.ndarray):
        if rainfall_mm.shape != flow_accumulation.shape:
            raise ValueError(
                f"rainfall_mm shape {rainfall_mm.shape} does not match "
                f"flow_accumulation shape {flow_accumulation.shape}."
            )
        # Avoid modifying input array by making a copy for calculations
        eff_rain = np.array(rainfall_mm, dtype=float, copy=True)
    else:
        # Uniform rainfall
        eff_rain = float(rainfall_mm)

    # Convert rainfall from mm to meters for dimensional consistency with DEM
    eff_rain_m = eff_rain / 1000.0

    # Surface runoff generated at each cell after infiltration (now in meters)
    generated_runoff = eff_rain_m * (1.0 - infiltration_rate)

    # Total runoff depth at a cell includes its locally generated runoff plus
    # the runoff from all upstream cells. Since flow_accumulation stores the number
    # of upstream cells, we scale the locally generated runoff by (flow_accumulation + 1)
    # to account for the local cell itself plus its upstream contributors.
    runoff = generated_runoff * (flow_accumulation + 1)

    return runoff.astype(float)
