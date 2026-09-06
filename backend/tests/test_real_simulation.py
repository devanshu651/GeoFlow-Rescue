import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.run_real_simulation import run_real_simulation


class TestRealSimulationScript(unittest.TestCase):
    def test_run_writes_expected_georeferenced_layers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dem_path = Path(temp_dir) / "dem.tif"
            output_dir = Path(temp_dir) / "outputs"
            dem = np.array(
                [[12, 11, 10], [11, 9, 10], [10, 10, 8]],
                dtype=np.int16,
            )
            transform = from_origin(500000, 3000000, 10, 10)
            with rasterio.open(
                dem_path,
                "w",
                driver="GTiff",
                height=3,
                width=3,
                count=1,
                dtype="int16",
                crs="EPSG:32645",
                transform=transform,
                nodata=-32767,
            ) as dst:
                dst.write(dem, 1)

            result = run_real_simulation(dem_path, output_dir)

            expected = {
                "slope.tif",
                "flow_direction.tif",
                "flow_accumulation.tif",
                "runoff.tif",
                "water_depth.tif",
                "flooded_cells.tif",
                "flood_risk.tif",
                "landslide_risk.tif",
            }
            self.assertEqual({path.name for path in output_dir.glob("*.tif")}, expected)
            self.assertEqual(result["dimensions"], (3, 3))
            with rasterio.open(output_dir / "slope.tif") as src:
                self.assertEqual(src.crs.to_epsg(), 32645)
                self.assertEqual(src.transform, transform)
                self.assertEqual((src.width, src.height), (3, 3))


if __name__ == "__main__":
    unittest.main()
