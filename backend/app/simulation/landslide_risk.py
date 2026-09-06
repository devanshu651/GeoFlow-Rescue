import numpy as np
from typing import Union

def calculate_landslide_risk(
    slope: np.ndarray,
    rainfall_mm: Union[float, np.ndarray],
    water_depth: np.ndarray,
    slope_threshold: float = 30.0,
    rainfall_threshold: float = 50.0,
    water_threshold: float = 0.05
) -> np.ndarray:
    """
    Calculates a prototype landslide risk score based on slope, rainfall, and water depth.

    IMPORTANT:
    This is a prototype susceptibility model for demonstration and simulation purposes only.
    It is NOT a scientifically validated landslide prediction model.

    Args:
        slope (np.ndarray): 2D array of slope in degrees.
        rainfall_mm (float or np.ndarray): Rainfall amount in millimeters.
        water_depth (np.ndarray): 2D array of water depth.
        slope_threshold (float): Slope threshold below which risk is 0.
        rainfall_threshold (float): Rainfall threshold below which risk is 0.
        water_threshold (float): Water depth threshold below which risk is 0.

    Returns:
        np.ndarray: Risk array normalized to [0, 1].
    """
    # Validation
    if slope.shape != water_depth.shape:
        raise ValueError("slope and water_depth arrays must have the same shape.")

    if isinstance(rainfall_mm, np.ndarray) and rainfall_mm.shape != slope.shape:
        raise ValueError("rainfall_mm array must have the same shape as slope.")

    if slope_threshold < 0 or rainfall_threshold < 0 or water_threshold < 0:
        raise ValueError("Thresholds must be non-negative.")

    # Margins to gradually scale the risk to 1.0 above the threshold
    slope_margin = 30.0   # e.g. reaches 1.0 at threshold + 30
    rain_margin = 100.0   # e.g. reaches 1.0 at threshold + 100
    water_margin = 0.5    # e.g. reaches 1.0 at threshold + 0.5

    # Component A: Slope risk
    slope_risk = np.clip((slope - slope_threshold) / slope_margin, 0.0, 1.0)

    # Component B: Rainfall risk
    rain_risk = np.clip((rainfall_mm - rainfall_threshold) / rain_margin, 0.0, 1.0)

    # Component C: Water/saturation risk
    water_risk = np.clip((water_depth - water_threshold) / water_margin, 0.0, 1.0)

    # Combined weighted score (Slope: 50%, Rainfall: 30%, Water: 20%)
    total_risk = 0.5 * slope_risk + 0.3 * rain_risk + 0.2 * water_risk

    return total_risk.astype(float)

def classify_landslide_risk(risk: np.ndarray) -> np.ndarray:
    """
    Classifies normalized risk scores into string categories.
    """
    conditions = [
        (risk >= 0.0) & (risk < 0.25),
        (risk >= 0.25) & (risk < 0.50),
        (risk >= 0.50) & (risk < 0.75),
        (risk >= 0.75)
    ]
    choices = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    return np.select(conditions, choices, default="UNKNOWN")

def detect_high_risk_zones(risk: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Returns a boolean mask identifying cells at or above the given risk threshold.
    """
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    return risk >= threshold
