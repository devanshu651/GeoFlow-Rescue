import os
import unittest
import tempfile
import rasterio
from rasterio.transform import from_origin
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

class TestSimulationAPI(unittest.TestCase):
    def setUp(self):
        # Create a temporary GeoTIFF file for testing
        self.tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        self.file_path = self.tmp.name
        self.tmp.close()

        self.width = 10
        self.height = 10
        self.transform = from_origin(0, 0, 1, 1) # West, North, xsize, ysize
        self.crs = 'EPSG:32643'

        # Create elevation data resembling a hill
        self.elevation_data = np.zeros((self.height, self.width), dtype=np.float32)
        for i in range(self.height):
            for j in range(self.width):
                self.elevation_data[i, j] = 100 - (abs(i-5) + abs(j-5)) * 10

        self.nodata_value = -9999.0

        with rasterio.open(
            self.file_path,
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=1,
            dtype=self.elevation_data.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata_value
        ) as dst:
            dst.write(self.elevation_data, 1)

    def tearDown(self):
        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except PermissionError:
                pass

    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_simulate_valid_request(self):
        payload = {
            "rainfall_mm": 100.0,
            "infiltration_rate": 0.2,
            "slope_threshold": 30.0,
            "rainfall_threshold": 50.0,
            "water_threshold": 0.05,
            "dem_file_path": self.file_path
        }
        response = client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["rainfall_mm"], 100.0)
        self.assertIn("max_water_depth", data)
        self.assertIn("flooded_cell_count", data)
        self.assertIn("first_pooling_cell_count", data)
        self.assertIn("flood_risk_max", data)
        self.assertIn("flood_risk_mean", data)
        self.assertIn("landslide_risk_max", data)
        self.assertIn("landslide_risk_mean", data)
        self.assertIn("high_risk_landslide_cell_count", data)

    def test_simulate_missing_dem_file_path(self):
        payload = {
            "rainfall_mm": 100.0,
            "infiltration_rate": 0.2
        }
        response = client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("dem_file_path is required", response.json()["detail"])

    def test_simulate_dem_file_not_found(self):
        payload = {
            "rainfall_mm": 100.0,
            "infiltration_rate": 0.2,
            "dem_file_path": "non_existent_file.tif"
        }
        response = client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 404)
        self.assertIn("DEM file not found", response.json()["detail"])

    def test_simulate_negative_rainfall(self):
        payload = {
            "rainfall_mm": -10.0,
            "infiltration_rate": 0.2,
            "dem_file_path": self.file_path
        }
        response = client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 422) # FastAPI validation error

    def test_simulate_invalid_infiltration_rate(self):
        payload = {
            "rainfall_mm": 100.0,
            "infiltration_rate": 1.5,
            "dem_file_path": self.file_path
        }
        response = client.post("/api/v1/simulate", json=payload)
        self.assertEqual(response.status_code, 422) # FastAPI validation error

        payload_neg = {
            "rainfall_mm": 100.0,
            "infiltration_rate": -0.1,
            "dem_file_path": self.file_path
        }
        response_neg = client.post("/api/v1/simulate", json=payload_neg)
        self.assertEqual(response_neg.status_code, 422)

if __name__ == '__main__':
    unittest.main()
