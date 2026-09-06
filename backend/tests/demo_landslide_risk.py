import numpy as np
import sys
import os

# Ensure the app module can be imported when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.simulation.landslide_risk import (
    calculate_landslide_risk,
    classify_landslide_risk,
    detect_high_risk_zones
)

def main():
    # 1. Create a synthetic 3x3 terrain scenario
    # Row 0: Flat areas
    # Row 1: Moderately sloped areas
    # Row 2: Steep / high-slope areas
    slope = np.array([
        [0.0, 5.0, 10.0],
        [15.0, 25.0, 35.0],
        [45.0, 60.0, 75.0]
    ])

    # 2. Corresponding synthetic rainfall and water depth
    # Dry/light rain in flat areas, intense rain in steep areas
    rainfall_mm = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 60.0, 80.0],
        [100.0, 150.0, 200.0]
    ])

    # Low water depth in flat areas, pooling/saturation in steep areas
    water_depth = np.array([
        [0.0, 0.01, 0.02],
        [0.03, 0.06, 0.1],
        [0.2, 0.5, 0.8]
    ])

    # 3. Calculate risks
    # Using default thresholds: slope_threshold=30.0, rainfall_threshold=50.0, water_threshold=0.05
    risk = calculate_landslide_risk(slope, rainfall_mm, water_depth)
    classes = classify_landslide_risk(risk)
    zones = detect_high_risk_zones(risk, threshold=0.5)

    # 4. Print human-readable output grids
    np.set_printoptions(precision=3, suppress=True)
    print("========================================")
    print("      GEOFLOW-RESCUE RISK DEMO")
    print("========================================\n")

    print("--- INPUTS ---")
    print("Slope Grid (degrees):")
    print(slope)
    print("\nRainfall Grid (mm):")
    print(rainfall_mm)
    print("\nWater Depth Grid (m):")
    print(water_depth)

    print("\n--- OUTPUTS ---")
    print("Numeric Risk Grid (0.0 to 1.0):")
    print(risk)
    print("\nRisk Classification Grid:")
    print(classes)
    print("\nHigh-Risk Zone Mask (Threshold >= 0.5):")
    print(zones)

    # 5. Print summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Minimum Risk: {np.min(risk):.3f}")
    print(f"Maximum Risk: {np.max(risk):.3f}")
    print(f"Mean Risk:    {np.mean(risk):.3f}")

    high_count = np.sum(classes == "HIGH")
    critical_count = np.sum(classes == "CRITICAL")
    print(f"Number of HIGH cells:     {high_count}")
    print(f"Number of CRITICAL cells: {critical_count}")
    print("========================================")

if __name__ == "__main__":
    main()
