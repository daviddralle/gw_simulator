#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.dem import download_and_clip_dem


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a 3DEP DEM from Earth Engine and clip it to a KML boundary.")
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--output-tif", required=True, type=Path)
    parser.add_argument("--ee-project", default=None)
    parser.add_argument("--scale", default=10.0, type=float)
    parser.add_argument("--temp-tif", default=Path("temp_dem_rectangle.tif"), type=Path)
    args = parser.parse_args()

    output = download_and_clip_dem(
        args.boundary,
        args.output_tif,
        ee_project=args.ee_project,
        scale=args.scale,
        temp_tif=args.temp_tif,
    )
    print(f"DEM saved to {output}")


if __name__ == "__main__":
    main()
