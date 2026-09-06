from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DEM = PROJECT_ROOT / "data" / "dem" / "n26_e086_1arc_v3.tif"
OUTPUT_DEM = PROJECT_ROOT / "data" / "dem" / "n26_e086_aoi_utm45.tif"


# --------------------------------------------------
# Selected GeoFlow-Rescue AOI
# --------------------------------------------------

MIN_LON = 86.5400
MIN_LAT = 26.6650
MAX_LON = 86.6350
MAX_LAT = 26.7350

AOI = box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)


# --------------------------------------------------
# Reproject + crop
# --------------------------------------------------

TARGET_CRS = "EPSG:32645"


with rasterio.open(INPUT_DEM) as src:

    print("Input DEM:")
    print("  CRS:", src.crs)
    print("  Size:", src.width, "x", src.height)
    print("  Resolution:", src.res)

    # Crop original DEM to AOI
    cropped, cropped_transform = mask(
        src,
        [mapping(AOI)],
        crop=True,
        nodata=src.nodata
    )

    cropped_profile = src.profile.copy()
    cropped_profile.update(
        height=cropped.shape[1],
        width=cropped.shape[2],
        transform=cropped_transform
    )

    # Calculate output transform for UTM
    transform, width, height = calculate_default_transform(
        src.crs,
        TARGET_CRS,
        cropped.shape[2],
        cropped.shape[1],
        *AOI.bounds
    )

    output_profile = cropped_profile.copy()

    output_profile.update(
        driver="GTiff",
        crs=TARGET_CRS,
        transform=transform,
        width=width,
        height=height,
        dtype=cropped.dtype,
        compress="deflate",
        nodata=src.nodata
    )

    with rasterio.open(OUTPUT_DEM, "w", **output_profile) as dst:

        reproject(
            source=cropped[0],
            destination=rasterio.band(dst, 1),
            src_transform=cropped_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=src.nodata
        )

print()
print("DEM preparation complete.")
print("Output:", OUTPUT_DEM)

with rasterio.open(OUTPUT_DEM) as src:

    data = src.read(1, masked=True)

    print()
    print("Output DEM:")
    print("  CRS:", src.crs)
    print("  Size:", src.width, "x", src.height)
    print("  Resolution:", src.res)
    print("  Bounds:", src.bounds)
    print("  Min elevation:", data.min())
    print("  Max elevation:", data.max())
    print("  NoData:", src.nodata)