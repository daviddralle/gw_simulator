#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.reach_plots import (
    ReachVisualizationConfig,
    create_reach_visualizations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create standard total-flow reach maps and routed water-year videos."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument(
        "--water-year",
        action="append",
        type=int,
        default=None,
        help=(
            "Water year to animate; repeat for multiple years. When omitted, the "
            "driest and wettest complete years are selected by modeled recharge."
        ),
    )
    parser.add_argument("--rolling-days", default=30, type=int)
    parser.add_argument("--frame-step-days", default=7, type=int)
    parser.add_argument("--frames-per-second", default=6, type=int)
    args = parser.parse_args()
    if args.rolling_days < 1 or args.frame_step_days < 1 or args.frames_per_second < 1:
        parser.error("Rolling window, frame step, and frame rate must all be positive.")

    metadata = create_reach_visualizations(
        args.output_dir,
        args.boundary,
        water_years=args.water_year,
        config=ReachVisualizationConfig(
            rolling_days=args.rolling_days,
            frame_step_days=args.frame_step_days,
            frames_per_second=args.frames_per_second,
        ),
    )
    names = ", ".join(metadata["files"])
    print(f"Reach visualizations saved to {args.output_dir}: {names}")


if __name__ == "__main__":
    main()
