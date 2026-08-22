from __future__ import annotations

import os
from pathlib import Path

import fiona
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.ops import transform


def _require_earth_engine():
    try:
        import ee
        import geemap
    except ImportError as exc:
        raise ImportError(
            "Earth Engine DEM extraction requires `earthengine-api` and `geemap`. "
            "Install them in the active environment before running this stage."
        ) from exc
    return ee, geemap


def initialize_earth_engine(project: str | None = None) -> None:
    ee, _ = _require_earth_engine()
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def remove_polygon_z(geometry):
    """Remove Z coordinates while preserving multipart geometry and holes."""
    return transform(lambda x, y, z=None: (x, y), geometry)


def download_and_clip_dem(
    boundary: str | Path,
    output_tif: str | Path,
    *,
    ee_project: str | None = None,
    dem_asset: str = "USGS/3DEP/10m_collection",
    scale: float = 10,
    temp_tif: str | Path = "temp_dem_rectangle.tif",
) -> Path:
    """Download a rectangular DEM from Earth Engine and clip it to a polygon boundary."""
    ee, geemap = _require_earth_engine()
    initialize_earth_engine(ee_project)

    boundary = Path(boundary)
    output_tif = Path(output_tif)
    temp_tif = Path(temp_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    fiona.drvsupport.supported_drivers["KML"] = "rw"
    if not boundary.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary}")

    gdf = gpd.read_file(boundary)
    if gdf.crs is None:
        raise ValueError(f"Boundary file has no CRS: {boundary}")
    if gdf.has_z.any():
        gdf = gdf.copy()
        gdf.geometry = gdf.geometry.map(remove_polygon_z)

    ee_gdf = gdf.to_crs("EPSG:4326")
    ee_roi = geemap.gdf_to_ee(ee_gdf).geometry()
    ee_bounds = ee_roi.bounds()
    dem = ee.ImageCollection(dem_asset).select("elevation").mosaic()

    geemap.ee_export_image(
        dem,
        filename=str(temp_tif),
        scale=scale,
        region=ee_bounds,
        file_per_band=False,
    )

    with rasterio.open(temp_tif) as src:
        clip_gdf = gdf.to_crs(src.crs)
        out_image, out_transform = mask(
            src,
            clip_gdf.geometry.values,
            crop=True,
            nodata=-9999,
        )
        out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999,
            "compress": "lzw",
        }
    )

    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)

    if temp_tif.exists():
        os.remove(temp_tif)

    return output_tif
