import os
import unittest
import rasterio
from rasterio.transform import from_origin
import numpy as np
import tempfile
from app.services.dem_service import read_dem

class TestDEMService(unittest.TestCase):

    def setUp(self):
        # Create a temporary GeoTIFF file for testing
        self.tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        self.file_path = self.tmp.name
        self.tmp.close()

        # Define raster properties
        self.width = 10
        self.height = 10
        self.transform = from_origin(0, 0, 1, 1) # West, North, xsize, ysize
        self.crs = 'EPSG:32643'

        # Create dummy elevation data (0 to 99)
        self.elevation_data = np.arange(100, dtype=np.float32).reshape((self.height, self.width))

        # Add a NoData value at the first pixel
        self.nodata_value = -9999.0
        self.elevation_data[0, 0] = self.nodata_value

        # Write to GeoTIFF
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
        # Cleanup
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_read_dem_success(self):
        elevation_data, metadata = read_dem(self.file_path)

        self.assertIsInstance(elevation_data, np.ndarray)
        self.assertEqual(elevation_data.shape, (10, 10))

        self.assertEqual(metadata["width"], 10)
        self.assertEqual(metadata["height"], 10)
        self.assertIn("EPSG:32643", metadata["crs"])
        self.assertEqual(metadata["min_elevation"], 1.0)  # Since 0.0 was replaced by nodata
        self.assertEqual(metadata["max_elevation"], 99.0)

    def test_read_dem_geographic_crs_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            geo_file = tmp.name

        try:
            with rasterio.open(
                geo_file,
                'w',
                driver='GTiff',
                height=self.height,
                width=self.width,
                count=1,
                dtype=self.elevation_data.dtype,
                crs='EPSG:4326',
                transform=self.transform,
                nodata=self.nodata_value
            ) as dst:
                dst.write(self.elevation_data, 1)

            with self.assertRaises(ValueError) as context:
                read_dem(geo_file)
            self.assertIn("Geographic CRS", str(context.exception))
        finally:
            if os.path.exists(geo_file):
                os.remove(geo_file)


    def test_read_dem_non_metre_projected_crs_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            feet_crs_file = tmp.name

        try:
            with rasterio.open(
                feet_crs_file,
                'w',
                driver='GTiff',
                height=self.height,
                width=self.width,
                count=1,
                dtype=self.elevation_data.dtype,
                crs='EPSG:2263',
                transform=self.transform,
                nodata=self.nodata_value
            ) as dst:
                dst.write(self.elevation_data, 1)

            with self.assertRaises(ValueError) as context:
                read_dem(feet_crs_file)
            self.assertIn("metre-based horizontal units", str(context.exception))
        finally:
            if os.path.exists(feet_crs_file):
                os.remove(feet_crs_file)
    def test_read_dem_missing_crs_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            missing_crs_file = tmp.name

        try:
            with rasterio.open(
                missing_crs_file,
                'w',
                driver='GTiff',
                height=self.height,
                width=self.width,
                count=1,
                dtype=self.elevation_data.dtype,
                transform=self.transform,
                nodata=self.nodata_value
            ) as dst:
                dst.write(self.elevation_data, 1)

            with self.assertRaises(ValueError) as context:
                read_dem(missing_crs_file)
            self.assertIn("missing CRS information", str(context.exception))
        finally:
            if os.path.exists(missing_crs_file):
                os.remove(missing_crs_file)

    def test_read_dem_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as context:
            read_dem("non_existent_file.tif")
        self.assertIn("DEM file not found", str(context.exception))

    def test_read_dem_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(b"invalid data")
            invalid_file_path = tmp.name

        try:
            with self.assertRaises(rasterio.errors.RasterioIOError):
                read_dem(invalid_file_path)
        finally:
            if os.path.exists(invalid_file_path):
                os.remove(invalid_file_path)

if __name__ == '__main__':
    unittest.main()
