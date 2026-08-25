#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.hydrogeology import prepare_glhymps2_porosity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GLHYMPS 2.0 hydrogeology alternative using Pelletier "
            "2016 landform thickness when supplied, and compare it with legacy rasters."
        )
    )
    parser.add_argument("--glhymps-zip", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--reference-raster", required=True, type=Path)
    parser.add_argument("--existing-porosity", default=None, type=Path)
    parser.add_argument("--existing-depth", default=None, type=Path)
    parser.add_argument("--existing-transmissivity", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--buffer-m", default=1000.0, type=float)
    parser.add_argument("--target-crs", default="EPSG:26910")
    parser.add_argument("--resolution-m", default=50.0, type=float)
    parser.add_argument(
        "--depth-source",
        default=None,
        help="Official BDTICM URL or a cached clipped GeoTIFF in centimeters.",
    )
    parser.add_argument("--pelletier-regolith", default=None, type=Path)
    parser.add_argument("--pelletier-sediment", default=None, type=Path)
    parser.add_argument("--pelletier-land-cover", default=None, type=Path)
    args = parser.parse_args()

    kwargs = {}
    if args.depth_source is not None:
        kwargs["depth_source"] = args.depth_source
    outputs = prepare_glhymps2_porosity(
        zip_path=args.glhymps_zip,
        boundary_path=args.boundary,
        reference_raster=args.reference_raster,
        existing_porosity_raster=args.existing_porosity,
        existing_depth_raster=args.existing_depth,
        existing_transmissivity_raster=args.existing_transmissivity,
        output_dir=args.output_dir,
        buffer_m=args.buffer_m,
        target_crs=args.target_crs,
        resolution_m=args.resolution_m,
        pelletier_regolith_raster=args.pelletier_regolith,
        pelletier_sediment_raster=args.pelletier_sediment,
        pelletier_land_cover_raster=args.pelletier_land_cover,
        **kwargs,
    )
    for name, path in outputs.items():
        if path is not None:
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
