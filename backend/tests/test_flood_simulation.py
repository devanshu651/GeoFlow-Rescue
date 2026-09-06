import unittest
import numpy as np
from app.simulation.flood_simulation import (
    calculate_water_depth,
    detect_flooded_cells,
    calculate_flood_risk,
    detect_first_pooling
)

class TestFloodSimulation(unittest.TestCase):
    def setUp(self):
        self.dem = np.array([
            [10, 10, 10],
            [10, 5, 10],
            [10, 10, 10]
        ], dtype=float)

        # All flow towards center sink
        self.flow_direction = np.array([
            [4, 5, 6],
            [3, 0, 7],
            [2, 1, 8]
        ], dtype=np.uint8)

        # Uniform runoff
        self.runoff = np.ones((3, 3), dtype=float)

    def test_calculate_water_depth(self):
        depth = calculate_water_depth(self.runoff, self.dem, self.flow_direction, iterations=5)

        # Center should accumulate water
        self.assertTrue(depth[1, 1] > 1.0)
        # Surrounding cells should have less water than they started with (since they transferred it)
        self.assertTrue(depth[0, 0] < 1.0)

        # Depth should never be negative
        self.assertTrue((depth >= 0).all())

    def test_water_depth_boundary_safety(self):
        # Create flow pointing out of bounds
        flow_out = np.array([
            [8, 1, 2],
            [7, 0, 3],
            [6, 5, 4]
        ], dtype=np.uint8)

        # Should execute safely without IndexError
        depth = calculate_water_depth(self.runoff, self.dem, flow_out, iterations=2)
        self.assertEqual(depth.shape, (3, 3))

    def test_inputs_not_modified(self):
        runoff_copy = self.runoff.copy()
        dem_copy = self.dem.copy()
        flow_copy = self.flow_direction.copy()

        calculate_water_depth(self.runoff, self.dem, self.flow_direction)

        np.testing.assert_array_equal(self.runoff, runoff_copy)
        np.testing.assert_array_equal(self.dem, dem_copy)
        np.testing.assert_array_equal(self.flow_direction, flow_copy)

    def test_detect_flooded_cells(self):
        depth = np.array([
            [0.0, 0.005],
            [0.02, 0.1]
        ])
        flooded = detect_flooded_cells(depth, threshold=0.01)
        self.assertFalse(flooded[0, 0])
        self.assertFalse(flooded[0, 1])
        self.assertTrue(flooded[1, 0])
        self.assertTrue(flooded[1, 1])

        with self.assertRaises(ValueError):
            detect_flooded_cells(depth, -1)

    def test_calculate_flood_risk(self):
        depth = np.array([
            [0.0, 1.0],
            [2.0, 4.0]
        ])
        risk = calculate_flood_risk(depth)
        self.assertEqual(risk[0, 0], 0.0)
        self.assertEqual(risk[1, 1], 1.0)
        self.assertEqual(risk[1, 0], 0.5)
        self.assertEqual(risk[0, 1], 0.25)

    def test_calculate_flood_risk_zeros(self):
        depth = np.zeros((2, 2))
        risk = calculate_flood_risk(depth)
        self.assertTrue((risk == 0).all())

    def test_detect_first_pooling(self):
        # dem has depression at center
        depth = np.zeros((3, 3))
        depth[1, 1] = 0.5  # Water in depression
        depth[0, 0] = 0.5  # Water not in depression

        pooling = detect_first_pooling(depth, self.dem)
        self.assertTrue(pooling[1, 1])
        self.assertFalse(pooling[0, 0])

if __name__ == '__main__':
    unittest.main()
