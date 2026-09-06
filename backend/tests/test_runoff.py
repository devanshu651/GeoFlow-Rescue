import unittest
import numpy as np
from app.simulation.runoff import calculate_runoff

class TestRunoff(unittest.TestCase):
    def setUp(self):
        self.flow_acc = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=float)

    def test_uniform_rainfall(self):
        # 10mm rainfall -> 0.01m, 20% infiltration -> 0.008m effective runoff generated per cell
        # total runoff = 0.008 * (flow_acc + 1)
        runoff = calculate_runoff(10.0, 0.2, self.flow_acc)
        self.assertAlmostEqual(runoff[0, 0], 0.008)    # 0.008 * 1
        self.assertAlmostEqual(runoff[0, 1], 0.016)   # 0.008 * 2
        self.assertAlmostEqual(runoff[0, 2], 0.024)   # 0.008 * 3

    def test_higher_rainfall_higher_runoff(self):
        r1 = calculate_runoff(10.0, 0.5, self.flow_acc)
        r2 = calculate_runoff(20.0, 0.5, self.flow_acc)
        self.assertTrue((r2 > r1).all())

    def test_higher_infiltration_lower_runoff(self):
        r1 = calculate_runoff(10.0, 0.2, self.flow_acc)
        r2 = calculate_runoff(10.0, 0.8, self.flow_acc)
        self.assertTrue((r2 < r1).all())

    def test_invalid_infiltration_rates(self):
        with self.assertRaises(ValueError):
            calculate_runoff(10.0, -0.1, self.flow_acc)
        with self.assertRaises(ValueError):
            calculate_runoff(10.0, 1.1, self.flow_acc)

    def test_array_rainfall(self):
        rainfall_arr = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30]
        ], dtype=float)
        runoff = calculate_runoff(rainfall_arr, 0.0, self.flow_acc)
        self.assertAlmostEqual(runoff[1, 1], 0.040) # 0.020 * (1 + 1)
        self.assertAlmostEqual(runoff[2, 2], 0.090) # 0.030 * (2 + 1)

    def test_invalid_array_dimensions(self):
        bad_rain = np.array([
            [10, 10],
            [10, 10]
        ], dtype=float)
        with self.assertRaises(ValueError):
            calculate_runoff(bad_rain, 0.0, self.flow_acc)

    def test_inputs_not_modified(self):
        rain = np.array([
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10]
        ], dtype=float)
        rain_copy = rain.copy()

        flow = self.flow_acc.copy()
        flow_copy = flow.copy()

        calculate_runoff(rain, 0.5, flow)

        np.testing.assert_array_equal(rain, rain_copy)
        np.testing.assert_array_equal(flow, flow_copy)

if __name__ == '__main__':
    unittest.main()
