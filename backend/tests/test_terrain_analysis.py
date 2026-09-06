import unittest
import numpy as np
from app.simulation.terrain_analysis import calculate_slope, calculate_d8_flow_direction, calculate_flow_accumulation

class TestTerrainAnalysis(unittest.TestCase):

    def test_calculate_slope(self):
        dem = np.array([
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6]
        ], dtype=float)

        # mock transform where cell dx=1, dy=1
        transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        slope = calculate_slope(dem, transform)

        expected_slope = np.degrees(np.arctan(np.sqrt(5)))
        self.assertEqual(slope.shape, (3, 3))
        self.assertAlmostEqual(slope[1, 1], expected_slope, places=4)

    def test_calculate_d8_flow_direction_sink(self):
        dem_sink = np.array([
            [5, 5, 5],
            [5, 1, 5],
            [5, 5, 5]
        ], dtype=float)

        flow_sink = calculate_d8_flow_direction(dem_sink)
        self.assertEqual(flow_sink[0, 1], 5) # N -> S
        self.assertEqual(flow_sink[1, 0], 3) # W -> E
        self.assertEqual(flow_sink[1, 2], 7) # E -> W
        self.assertEqual(flow_sink[2, 1], 1) # S -> N
        self.assertEqual(flow_sink[1, 1], 0) # Center is sink

    def test_calculate_d8_flow_direction_peak(self):
        dem_peak = np.array([
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1]
        ], dtype=float)

        flow_peak = calculate_d8_flow_direction(dem_peak)
        # Center peak should flow to one of the neighbors with max slope.
        # Since drop is 4 to all, it should pick the first in our encoding order (1=N)
        self.assertEqual(flow_peak[1, 1], 1) # First neighbor is N

    def test_calculate_d8_flow_direction_boundary(self):
        # Check that edges don't cause IndexError and flow respects boundaries
        dem = np.array([
            [10, 10, 10],
            [10, 10, 10],
            [10, 10, 10]
        ], dtype=float)
        # For a flat dem, all drops are 0, max_slope is 0, should default to 0
        flow = calculate_d8_flow_direction(dem)
        self.assertTrue((flow == 0).all())

    def test_calculate_flow_accumulation_linear(self):
        # 3x3 flow direction: everyone flows East
        flow_dir = np.array([
            [3, 3, 0],
            [3, 3, 0],
            [3, 3, 0]
        ], dtype=np.uint8)

        acc = calculate_flow_accumulation(flow_dir)

        self.assertEqual(acc[0, 0], 0)
        self.assertEqual(acc[0, 1], 1)
        self.assertEqual(acc[0, 2], 2)

    def test_calculate_flow_accumulation_sink(self):
        # All flow to center
        flow_dir = np.array([
            [4, 5, 6],
            [3, 0, 7],
            [2, 1, 8]
        ], dtype=np.uint8)

        acc = calculate_flow_accumulation(flow_dir)
        # Center cell should have accumulation of 8 (all other 8 cells flow into it)
        self.assertEqual(acc[1, 1], 8)
        # Corner cells should have accumulation of 0
        self.assertEqual(acc[0, 0], 0)

if __name__ == '__main__':
    unittest.main()
