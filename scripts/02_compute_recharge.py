#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.recharge import build_recharge_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a daily recharge CSV using PML ET, PRISM precipitation, and a storage-deficit bucket.")
    parser.add_argument("--boundary", required=True, type=Path, help="Watershed boundary vector file.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--start-year", default=2000, type=int)
    parser.add_argument("--end-year", default=2024, type=int)
    parser.add_argument("--initial-deficit-mm", default=0.0, type=float)
    parser.add_argument("--ee-project", default=None)
    args = parser.parse_args()

    output = build_recharge_csv(
        args.boundary,
        args.output_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        initial_deficit_mm=args.initial_deficit_mm,
        ee_project=args.ee_project,
    )
    print(f"Recharge CSV saved to {output}")


if __name__ == "__main__":
    main()
