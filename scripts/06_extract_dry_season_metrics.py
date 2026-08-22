#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.dry_season_metrics import DrySeasonMetricConfig, write_dry_season_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract June-October basin and reach dry-season metrics."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=[6, 7, 8, 9, 10],
    )
    parser.add_argument("--rolling-window-days", type=int, nargs="+", default=[7, 30])
    parser.add_argument("--near-zero-flow-m3d", type=float, default=1.0)
    args = parser.parse_args()
    destination = args.destination or args.output_dir / "dry_season_metrics"
    config = DrySeasonMetricConfig(
        months=tuple(args.months),
        rolling_windows_days=tuple(args.rolling_window_days),
        near_zero_flow_m3d=args.near_zero_flow_m3d,
    )
    metadata = write_dry_season_metrics(args.output_dir, destination, config)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
