import unittest
import numpy as np
from app.simulation.landslide_risk import (
    calculate_landslide_risk,
    classify_landslide_risk,
    detect_high_risk_zones
)

class TestLandslideRisk(unittest.TestCase):
    def setUp(self):
        self.slope = np.array([
            [10, 20, 30],
            [40, 50, 60]
        ], dtype=float)

        self.water_depth = np.array([
            [0.0, 0.01, 0.05],
            [0.1, 0.2, 0.5]
        ], dtype=float)

    def test_higher_slope_increases_risk(self):
        risk = calculate_landslide_risk(self.slope, 0.0, self.water_depth)
        # slope increases left to right and top to bottom
        self.assertTrue(risk[0, 2] >= risk[0, 1])
        self.assertTrue(risk[1, 1] > risk[0, 1])

    def test_higher_rainfall_increases_risk(self):
        r1 = calculate_landslide_risk(self.slope, 40.0, self.water_depth)
        r2 = calculate_landslide_risk(self.slope, 100.0, self.water_depth)
        self.assertTrue((r2 > r1).any())

    def test_higher_water_depth_increases_risk(self):
        water2 = self.water_depth + 0.2
        r1 = calculate_landslide_risk(self.slope, 0.0, self.water_depth)
        r2 = calculate_landslide_risk(self.slope, 0.0, water2)
        self.assertTrue((r2 > r1).any())

    def test_risk_stays_between_0_and_1(self):
        # Extreme inputs
        extreme_slope = np.ones((2, 3)) * 90.0
        extreme_water = np.ones((2, 3)) * 10.0
        risk = calculate_landslide_risk(extreme_slope, 500.0, extreme_water)
        self.assertTrue((risk <= 1.0).all())
        self.assertTrue((risk >= 0.0).all())

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            calculate_landslide_risk(self.slope, 10.0, self.water_depth, slope_threshold=-10.0)

    def test_array_dimensions_validated(self):
        bad_slope = np.ones((3, 3))
        with self.assertRaises(ValueError):
            calculate_landslide_risk(bad_slope, 10.0, self.water_depth)

        bad_rain = np.ones((2, 2))
        with self.assertRaises(ValueError):
            calculate_landslide_risk(self.slope, bad_rain, self.water_depth)

    def test_risk_classification(self):
        risk = np.array([0.1, 0.3, 0.6, 0.9])
        classes = classify_landslide_risk(risk)
        self.assertEqual(classes[0], "LOW")
        self.assertEqual(classes[1], "MODERATE")
        self.assertEqual(classes[2], "HIGH")
        self.assertEqual(classes[3], "CRITICAL")

    def test_high_risk_zones(self):
        risk = np.array([0.1, 0.6, 0.9])
        zones = detect_high_risk_zones(risk, 0.5)
        self.assertFalse(zones[0])
        self.assertTrue(zones[1])
        self.assertTrue(zones[2])

    def test_inputs_not_modified(self):
        slope_copy = self.slope.copy()
        water_copy = self.water_depth.copy()

        calculate_landslide_risk(self.slope, 50.0, self.water_depth)

        np.testing.assert_array_equal(self.slope, slope_copy)
        np.testing.assert_array_equal(self.water_depth, water_copy)

if __name__ == '__main__':
    unittest.main()
