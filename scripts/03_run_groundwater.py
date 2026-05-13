#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.groundwater import (
    GroundwaterConfig,
    GroundwaterInputs,
    build_groundwater_state,
    build_monthly_pumping_maps,
    run_scenarios,
    save_outputs,
    save_setup_plots,
    spin_up_steady_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dupuit groundwater simulation.")
    parser.add_argument("--dem", default=Path("data/drainage_area_10m_dem_clipped.tif"), type=Path)
    parser.add_argument("--boundary", default=Path("data/comid_8273277.gpkg"), type=Path)
    parser.add_argument("--recharge-csv", default=Path("data/daily_water_balance_full.csv"), type=Path)
    parser.add_argument("--transmissivity", default=Path("data/GLYMPHS/transmissivity_m2d.tif"), type=Path)
    parser.add_argument("--depth-to-bedrock", default=Path("data/GLYMPHS/depthToBedrock_m.tif"), type=Path)
    parser.add_argument("--porosity", default=Path("data/GLYMPHS/storativity.tif"), type=Path)
    parser.add_argument("--wells", default=None, type=Path)
    parser.add_argument("--pumping-schedule", default=None, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--start-date", default="2022-10-01", type=str)
    parser.add_argument("--end-date", default="2023-09-30", type=str)
    parser.add_argument("--target-resolution", default=50.0, type=float)
    parser.add_argument("--target-crs", default="EPSG:26910")
    parser.add_argument("--stream-area-threshold", default=250000.0, type=float)
    parser.add_argument("--courant-coefficient", default=0.5, type=float)
    parser.add_argument("--stream-drain-offset", default=0.0, type=float)
    parser.add_argument("--steady-state-recharge-mm-day", default=1.5, type=float)
    parser.add_argument("--skip-spinup", action="store_true")
    parser.add_argument("--initial-condition", choices=["base", "heads"], default="base")
    parser.add_argument("--progress-interval", default=30, type=int)
    args = parser.parse_args()


    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GroundwaterConfig(
        target_crs=args.target_crs,
        target_resolution=args.target_resolution,
        stream_area_threshold=args.stream_area_threshold,
        courant_coefficient=args.courant_coefficient,
        stream_drain_offset=args.stream_drain_offset,
    )
    inputs = GroundwaterInputs(
        dem_path=args.dem,
        basin_path=args.boundary,
        transmissivity_path=args.transmissivity,
        depth_to_bedrock_path=args.depth_to_bedrock,
        porosity_path=args.porosity,
    )

    state = build_groundwater_state(inputs, config)
    save_setup_plots(state, config, output_dir)
    heads_path = output_dir / "steady_state_heads.npy"
    if not args.skip_spinup:
        spinup = spin_up_steady_state(
            state,
            heads_path,
            config,
            recharge_mm_day=args.steady_state_recharge_mm_day,
        )
        spinup.to_csv(output_dir / "steady_state_spinup.csv", index=False)

    if args.wells and args.pumping_schedule:
        pumping_maps = build_monthly_pumping_maps(
            state,
            wells_path=args.wells,
            pumping_path=args.pumping_schedule,
            basin_path=args.boundary,
            target_crs=args.target_crs,
        )
    else:
        pumping_maps = None

    results, snapshots = run_scenarios(
        state,
        recharge_csv=args.recharge_csv,
        pumping_maps=pumping_maps,
        start_date=args.start_date,
        end_date=args.end_date,
        config=config,
        initial_condition=args.initial_condition,
        heads_path=heads_path,
        progress_interval=args.progress_interval,
    )
    save_outputs(
        results,
        snapshots,
        pumping_maps,
        state,
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        recharge_csv=args.recharge_csv,
    )
    print(f"Groundwater outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
