#!/usr/bin/env python
"""Create deterministic inputs for the public synthetic-basin example."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box


EXAMPLE_DIR = Path(__file__).resolve().parent
INPUT_DIR = EXAMPLE_DIR / "inputs"
CRS = "EPSG:26910"
CELL_SIZE_M = 100.0
WIDTH = 15
HEIGHT = 15
WEST = 500_000.0
NORTH = 4_201_500.0


def _write_raster(path: Path, values: np.ndarray, description: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(WEST, NORTH, CELL_SIZE_M, CELL_SIZE_M),
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(values.astype("float32"), 1)
        destination.set_band_description(1, description)


def build_inputs(output_dir: Path = INPUT_DIR) -> list[Path]:
    """Write a small basin, model rasters, recharge, and synthetic pumping."""
    output_dir.mkdir(parents=True, exist_ok=True)

    east = WEST + WIDTH * CELL_SIZE_M
    south = NORTH - HEIGHT * CELL_SIZE_M
    x = WEST + (np.arange(WIDTH) + 0.5) * CELL_SIZE_M
    y = NORTH - (np.arange(HEIGHT) + 0.5) * CELL_SIZE_M
    xx, yy = np.meshgrid(x, y)

    # A south-draining surface with a shallow central valley. All values are
    # synthetic and have no geographic or calibrated interpretation.
    center_x = (WEST + east) / 2.0
    dem = (
        100.0
        + 0.018 * (yy - south)
        + 0.000025 * np.square(xx - center_x)
    )
    transmissivity = np.full_like(dem, 200.0)
    depth_to_bedrock = np.full_like(dem, 12.0)
    specific_yield = np.full_like(dem, 0.10)

    raster_values = {
        "dem.tif": (dem, "synthetic land-surface elevation (m)"),
        "transmissivity_m2day.tif": (
            transmissivity,
            "synthetic transmissivity (m2/day)",
        ),
        "depth_to_bedrock_m.tif": (
            depth_to_bedrock,
            "synthetic aquifer thickness (m)",
        ),
        "specific_yield.tif": (
            specific_yield,
            "synthetic specific yield (dimensionless)",
        ),
    }
    written: list[Path] = []
    for name, (values, description) in raster_values.items():
        path = output_dir / name
        _write_raster(path, values, description)
        written.append(path)

    inset = CELL_SIZE_M
    boundary_path = output_dir / "boundary.geojson"
    gpd.GeoDataFrame(
        {"name": ["synthetic_basin"]},
        geometry=[box(WEST + inset, south + inset, east - inset, NORTH - inset)],
        crs=CRS,
    ).to_file(boundary_path, driver="GeoJSON")
    written.append(boundary_path)

    wells_path = output_dir / "wells.geojson"
    gpd.GeoDataFrame(
        {"APN": ["SYNTHETIC-001"]},
        geometry=[Point(center_x + 100.0, south + 500.0)],
        crs=CRS,
    ).to_file(wells_path, driver="GeoJSON")
    written.append(wells_path)

    pumping_path = output_dir / "pumping_climatology.csv"
    pd.DataFrame(
        {
            "Month": np.arange(1, 13),
            "APN": "SYNTHETIC-001",
            "waterUse_m3Day": [25, 25, 25, 40, 50, 75, 100, 100, 75, 50, 40, 25],
        }
    ).to_csv(pumping_path, index=False)
    written.append(pumping_path)

    dates = pd.date_range("2019-01-01", "2020-12-31", freq="D")
    recharge = np.full(len(dates), 1.0)
    recharge[(np.arange(len(dates)) + 4) % 14 == 0] = 12.0
    recharge_path = output_dir / "recharge.csv"
    pd.DataFrame({"date": dates, "Recharge": recharge}).to_csv(
        recharge_path,
        index=False,
    )
    written.append(recharge_path)

    return written


def main() -> None:
    written = build_inputs()
    print(f"Wrote {len(written)} synthetic input files to {INPUT_DIR}")


if __name__ == "__main__":
    main()
