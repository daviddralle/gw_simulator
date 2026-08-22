#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.groundwater import (
    GroundwaterConfig,
    GroundwaterInputs,
    apply_specific_yield_floor,
    apply_well_aquifer_depth,
    build_depletion_table,
    build_groundwater_state,
    build_monthly_pumping_maps,
    run_scenarios,
    save_outputs,
    save_reach_outputs,
    save_setup_plots,
    spin_up_transient,
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repository: Path) -> dict[str, object]:
    """Return best-effort source revision details without requiring Git at runtime."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
        dirty = None
    return {"git_commit": revision, "git_worktree_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Dupuit groundwater simulation.")
    parser.add_argument("--dem", default=Path("data/drainage_area_10m_dem_clipped.tif"), type=Path)
    parser.add_argument("--boundary", default=Path("data/comid_8273277.gpkg"), type=Path)
    recharge_group = parser.add_mutually_exclusive_group()
    recharge_group.add_argument(
        "--recharge-csv",
        default=None,
        type=Path,
        help="Daily basin-mean CSV with `date` and `Recharge` (mm/day).",
    )
    recharge_group.add_argument(
        "--recharge-raster-manifest",
        default=None,
        type=Path,
        help=(
            "Daily spatial-recharge manifest with date, raster_path, and optional "
            "one-based band and units columns."
        ),
    )
    parser.add_argument("--transmissivity", default=Path("data/GLYMPHS/transmissivity_m2d.tif"), type=Path)
    parser.add_argument("--depth-to-bedrock", default=Path("data/GLYMPHS/depthToBedrock_m.tif"), type=Path)
    parser.add_argument("--porosity", default=Path("data/GLYMPHS/storativity.tif"), type=Path)
    parser.add_argument("--wells", default=None, type=Path)
    parser.add_argument("--pumping-schedule", default=None, type=Path)
    parser.add_argument(
        "--pumping-mode",
        choices=["timeseries", "climatology"],
        default="timeseries",
        help=(
            "Use dated year-month pumping records, with zero pumping outside their coverage, "
            "or repeat the mean calendar-month climatology."
        ),
    )
    parser.add_argument(
        "--pumping-source-mode",
        choices=["well_cell", "topographic"],
        default="well_cell",
        help=(
            "Apply demand at exact mapped cells or allocate it daily within disjoint "
            "D8 channel-reach catchments."
        ),
    )
    parser.add_argument(
        "--pumping-source-area-threshold",
        default=500000.0,
        type=float,
        help=(
            "Minimum D8 drainage area (m2) defining the coarser channel network "
            "used to partition topographic pumping source zones."
        ),
    )
    parser.add_argument("--output-dir", default=Path("outputs"), type=Path)
    parser.add_argument("--start-date", default="2022-10-01", type=str)
    parser.add_argument("--end-date", default="2023-09-30", type=str)
    parser.add_argument("--target-resolution", default=50.0, type=float)
    parser.add_argument("--target-crs", default="EPSG:26910")
    parser.add_argument("--stream-area-threshold", default=250000.0, type=float)
    parser.add_argument("--courant-coefficient", default=0.5, type=float)
    parser.add_argument("--stream-drain-offset", default=0.0, type=float)
    parser.add_argument(
        "--additional-aquifer-depth",
        default=0.0,
        type=float,
        help=(
            "Additional smooth aquifer thickness below the depth database (m); "
            "intended for sensitivity tests and zero by default."
        ),
    )
    parser.add_argument(
        "--deep-aquifer-hydraulics",
        choices=["preserve_transmissivity", "preserve_conductivity"],
        default="preserve_transmissivity",
        help="Whether added depth preserves raster transmissivity or inferred conductivity.",
    )
    parser.add_argument(
        "--well-additional-depth",
        default=0.0,
        type=float,
        help=(
            "Additional depth at exact mapped pumping cells only (m). Link hydraulic "
            "conductivity remains unchanged."
        ),
    )
    parser.add_argument(
        "--specific-yield-floor",
        default=0.0,
        type=float,
        help=(
            "Optional effective specific-yield floor applied to all modeled aquifer "
            "cells while preserving the input raster as a separate field."
        ),
    )
    parser.add_argument(
        "--source-zone-storage-fraction",
        default=0.5,
        type=float,
        help=(
            "Maximum fraction of each source cell's current drainable storage made "
            "available to the daily topographic pumping allocator."
        ),
    )
    parser.add_argument(
        "--pumping-storage-fraction",
        default=0.5,
        type=float,
        help="Maximum fraction of local saturated storage removed per pumping solver chunk.",
    )
    parser.add_argument(
        "--strict-pumping-supply",
        action="store_true",
        help="Fail if mapped shallow cells cannot supply the full pumping sink.",
    )
    parser.add_argument(
        "--stream-loss-mode",
        choices=["unlimited_fixed_head", "routed_volume_limited"],
        default="routed_volume_limited",
        help=(
            "Use conventional two-way fixed-head stream exchange or limit losing "
            "exchange to water routed through each reach during every solver substep."
        ),
    )
    parser.add_argument("--stream-limiter-tolerance-m3", default=1.0e-6, type=float)
    parser.add_argument("--stream-limiter-max-iterations", default=25, type=int)
    parser.add_argument("--spinup-start-date", default=None, type=str, help="Start date for spin-up period.")
    parser.add_argument("--spinup-end-date", default=None, type=str, help="End date for spin-up period.")
    parser.add_argument("--skip-spinup", action="store_true")
    parser.add_argument(
        "--spinup-initial-heads",
        default=None,
        type=Path,
        help="Optional water-table file used to warm-start transient spin-up.",
    )
    parser.add_argument(
        "--pump-during-spinup",
        action="store_true",
        help="Also warm up the pumped branch using the selected pumping forcing.",
    )
    parser.add_argument(
        "--initial-condition",
        choices=["base", "heads"],
        default=None,
        help="Defaults to spin-up heads when spin-up runs, otherwise to the aquifer base.",
    )
    parser.add_argument(
        "--initial-heads-path",
        default=None,
        type=Path,
        help="Existing unimpaired heads used to restart a run when spin-up is skipped.",
    )
    parser.add_argument(
        "--pumped-initial-heads-path",
        default=None,
        type=Path,
        help="Existing pumped heads used to restart a paired run when spin-up is skipped.",
    )
    parser.add_argument(
        "--snapshot-date",
        action="append",
        default=None,
        help=(
            "Save heads on this YYYY-MM-DD date; repeat for multiple dates. "
            "If omitted, the legacy April/September month-end schedule is used."
        ),
    )
    parser.add_argument(
        "--workflow-config",
        default=None,
        type=Path,
        help="Optional YAML config to record in the run provenance.",
    )
    args = parser.parse_args()
    if args.recharge_csv is None and args.recharge_raster_manifest is None:
        args.recharge_csv = Path("data/daily_water_balance_full.csv")


    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GroundwaterConfig(
        target_crs=args.target_crs,
        target_resolution=args.target_resolution,
        stream_area_threshold=args.stream_area_threshold,
        courant_coefficient=args.courant_coefficient,
        stream_drain_offset=args.stream_drain_offset,
        additional_aquifer_depth=args.additional_aquifer_depth,
        deep_aquifer_hydraulics=args.deep_aquifer_hydraulics,
        well_additional_depth=args.well_additional_depth,
        pumping_source_mode=args.pumping_source_mode,
        pumping_source_area_threshold=args.pumping_source_area_threshold,
        specific_yield_floor=args.specific_yield_floor,
        source_zone_storage_fraction=args.source_zone_storage_fraction,
        pumping_storage_fraction=args.pumping_storage_fraction,
        strict_pumping_supply=args.strict_pumping_supply,
        stream_loss_mode=args.stream_loss_mode,
        stream_limiter_tolerance_m3=args.stream_limiter_tolerance_m3,
        stream_limiter_max_iterations=args.stream_limiter_max_iterations,
    )
    inputs = GroundwaterInputs(
        dem_path=args.dem,
        basin_path=args.boundary,
        transmissivity_path=args.transmissivity,
        depth_to_bedrock_path=args.depth_to_bedrock,
        porosity_path=args.porosity,
    )

    state = build_groundwater_state(inputs, config)

    if bool(args.wells) != bool(args.pumping_schedule):
        parser.error("Provide both --wells and --pumping-schedule, or neither.")
    if args.wells and args.pumping_schedule:
        pumping_forcing = build_monthly_pumping_maps(
            state,
            wells_path=args.wells,
            pumping_path=args.pumping_schedule,
            basin_path=args.boundary,
            target_crs=args.target_crs,
            mode=args.pumping_mode,
            source_mode=args.pumping_source_mode,
            source_area_threshold=args.pumping_source_area_threshold,
        )
    else:
        pumping_forcing = None
    changed_specific_yield_nodes = np.array([], dtype=int)
    if args.specific_yield_floor > 0.0:
        changed_specific_yield_nodes = apply_specific_yield_floor(
            state,
            args.specific_yield_floor,
        )
        print(
            f"Applied a specific-yield floor of "
            f"{args.specific_yield_floor:g} at {changed_specific_yield_nodes.size} "
            "modeled aquifer cell(s)."
        )
    if args.well_additional_depth > 0.0:
        if args.pumping_source_mode != "well_cell":
            parser.error(
                "--well-additional-depth is only supported with well-cell pumping sources."
            )
        if pumping_forcing is None or pumping_forcing.well_nodes is None:
            parser.error("--well-additional-depth requires mapped wells and pumping data.")
        applied_nodes = apply_well_aquifer_depth(
            state,
            pumping_forcing.well_nodes,
            args.well_additional_depth,
        )
        print(
            f"Added {args.well_additional_depth:g} m of local aquifer depth at "
            f"{applied_nodes.size} pumping node(s); link conductivity is unchanged."
        )
    save_setup_plots(state, config, output_dir)
    if args.pump_during_spinup and pumping_forcing is None:
        parser.error("--pump-during-spinup requires wells and a pumping schedule.")

    if (args.initial_heads_path or args.pumped_initial_heads_path) and not args.skip_spinup:
        parser.error("External restart heads require --skip-spinup.")
    if args.pumped_initial_heads_path and not args.initial_heads_path:
        parser.error("--pumped-initial-heads-path also requires --initial-heads-path.")
    heads_path = args.initial_heads_path or (output_dir / "spinup_heads_unimpaired.npy")
    pumped_heads_path = args.pumped_initial_heads_path or (
        output_dir / "spinup_heads_pumped.npy"
    )
    spinup_start_date = None
    spinup_end_date = None
    if not args.skip_spinup:
        spinup_start_date = args.spinup_start_date
        spinup_end_date = args.spinup_end_date
        if (spinup_start_date is None) != (spinup_end_date is None):
            parser.error("Provide both --spinup-start-date and --spinup-end-date, or neither.")
        if spinup_start_date is None:
            simulation_start = pd.Timestamp(args.start_date)
            spinup_end = simulation_start - pd.Timedelta(days=1)
            spinup_start = simulation_start - pd.DateOffset(years=2)
            spinup_start_date = spinup_start.strftime("%Y-%m-%d")
            spinup_end_date = spinup_end.strftime("%Y-%m-%d")
            print(f"Auto-inferring spin-up period: {spinup_start_date} to {spinup_end_date}")
        if pd.Timestamp(spinup_end_date) >= pd.Timestamp(args.start_date):
            parser.error("Spin-up must end before the main simulation starts.")

        spinup = spin_up_transient(
            state,
            heads_path,
            args.recharge_csv,
            spinup_start_date,
            spinup_end_date,
            config,
            initial_heads_path=args.spinup_initial_heads,
            recharge_raster_manifest=args.recharge_raster_manifest,
        )
        spinup.to_csv(output_dir / "transient_spinup.csv", index=False)
        if args.pump_during_spinup:
            pumped_spinup = spin_up_transient(
                state,
                pumped_heads_path,
                args.recharge_csv,
                spinup_start_date,
                spinup_end_date,
                config,
                initial_heads_path=args.spinup_initial_heads,
                pumping_forcing=pumping_forcing,
                recharge_raster_manifest=args.recharge_raster_manifest,
            )
            pumped_spinup.to_csv(
                output_dir / "transient_spinup_with_pumping.csv", index=False
            )

    initial_condition = args.initial_condition
    if initial_condition is None:
        initial_condition = (
            "heads" if args.initial_heads_path or not args.skip_spinup else "base"
        )
    if initial_condition == "heads" and not heads_path.exists():
        parser.error(f"Initial head file does not exist: {heads_path}")

    results, snapshots, reach_network, reach_results = run_scenarios(
        state,
        recharge_csv=args.recharge_csv,
        recharge_raster_manifest=args.recharge_raster_manifest,
        pumping_forcing=pumping_forcing,
        start_date=args.start_date,
        end_date=args.end_date,
        config=config,
        initial_condition=initial_condition,
        heads_path=heads_path,
        pumped_heads_path=(
            pumped_heads_path
            if args.pump_during_spinup or args.pumped_initial_heads_path
            else None
        ),
        snapshot_dates=args.snapshot_date,
    )
    save_outputs(
        results,
        snapshots,
        pumping_forcing,
        state,
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    reach_output_metadata = save_reach_outputs(
        state,
        reach_network,
        reach_results,
        results,
        output_dir,
        enforce_stream_availability=(
            config.stream_loss_mode == "routed_volume_limited"
        ),
    )
    pumping_metadata = None
    if pumping_forcing is not None:
        depletion = build_depletion_table(
            results["Unimpaired (Natural)"], results["With Pumping"]
        )
        final_depletion = depletion.iloc[-1]
        pumping_metadata = {
            "mode": pumping_forcing.mode,
            "source_mode": pumping_forcing.source_mode,
            "coverage_start": (
                str(pumping_forcing.coverage_start)
                if pumping_forcing.coverage_start is not None
                else None
            ),
            "coverage_end": (
                str(pumping_forcing.coverage_end)
                if pumping_forcing.coverage_end is not None
                else None
            ),
            "outside_coverage": (
                "zero"
                if pumping_forcing.mode == "timeseries"
                else "not applicable; repeats the calendar-month climatology"
            ),
            "implementation": (
                "daily transmissivity-weighted, storage-capped allocation within D8 "
                "channel-reach catchments"
                if pumping_forcing.source_mode == "topographic"
                else "negative recharge at mapped core nodes"
            ),
            "source_area_threshold_m2": pumping_forcing.source_area_threshold_m2,
            "pumping_node_count": int(pumping_forcing.well_nodes.size),
            "pumping_nodes": pumping_forcing.well_nodes.tolist(),
            "source_zone_count": (
                len(pumping_forcing.source_zones)
                if pumping_forcing.source_zones is not None
                else None
            ),
            "source_node_count": (
                int(pumping_forcing.source_nodes.size)
                if pumping_forcing.source_nodes is not None
                else None
            ),
            "source_zone_area_m2": (
                {
                    "minimum": float(
                        min(
                            np.sum(state.grid.cell_area_at_node[nodes])
                            for nodes in pumping_forcing.source_zones.values()
                        )
                    ),
                    "median": float(
                        np.median(
                            [
                                np.sum(state.grid.cell_area_at_node[nodes])
                                for nodes in pumping_forcing.source_zones.values()
                            ]
                        )
                    ),
                    "maximum": float(
                        max(
                            np.sum(state.grid.cell_area_at_node[nodes])
                            for nodes in pumping_forcing.source_zones.values()
                        )
                    ),
                }
                if pumping_forcing.source_zones
                else None
            ),
            "specific_yield_floor_changed_node_count": int(
                changed_specific_yield_nodes.size
            ),
            "specific_yield_floor_changed_area_m2": float(
                np.sum(state.grid.cell_area_at_node[changed_specific_yield_nodes])
            ),
            "scheduled_volume_m3": float(final_depletion["cumulative_scheduled_pumping_m3"]),
            "allocated_pumping_m3": float(
                final_depletion["cumulative_allocated_pumping_m3"]
            ),
            "modeled_extraction_m3": float(final_depletion["cumulative_modeled_extraction_m3"]),
            "source_capacity_shortfall_m3": float(
                final_depletion["cumulative_source_capacity_shortfall_m3"]
            ),
            "source_allocation_fulfillment_pct": float(
                final_depletion["cumulative_source_allocation_fulfillment_pct"]
            ),
            "pumping_balance_gap_m3": float(final_depletion["cumulative_pumping_balance_gap_m3"]),
            "unmodeled_supply_requirement_m3": float(
                final_depletion["cumulative_unmodeled_supply_requirement_m3"]
            ),
            "schedule_fulfillment_pct": float(final_depletion["cumulative_schedule_fulfillment_pct"]),
            "diagnostic_note": (
                "Source-capacity shortfall is reported demand omitted by a bounded "
                "topographic source zone. Pumping balance gap is a paired water-balance "
                "closure diagnostic and should be near machine precision. Landlab's raw "
                "surface-flux integration estimate is retained separately from the "
                "closed saturation-excess term."
            ),
        }
    input_paths = {
        **{name: Path(path) for name, path in asdict(inputs).items()},
        "recharge_csv": args.recharge_csv,
        "recharge_raster_manifest": args.recharge_raster_manifest,
        "wells": args.wells,
        "pumping_schedule": args.pumping_schedule,
        "workflow_config": args.workflow_config,
    }
    repository = Path(__file__).resolve().parents[1]
    code_paths = {
        "groundwater_module": repository / "gw_simulator/groundwater.py",
        "groundwater_entrypoint": Path(__file__).resolve(),
        "workflow_runner": repository / "scripts/run_workflow.py",
    }
    metadata = {
        "metadata_schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            **git_provenance(repository),
            "code_sha256": {
                name: file_sha256(path) for name, path in code_paths.items()
            },
        },
        "simulation_start": args.start_date,
        "simulation_end": args.end_date,
        "spinup_start": spinup_start_date,
        "spinup_end": spinup_end_date,
        "initial_condition": initial_condition,
        "scenario_initial_heads": (
            str(heads_path.resolve()) if initial_condition == "heads" else None
        ),
        "pumped_scenario_initial_heads": (
            str(pumped_heads_path.resolve())
            if initial_condition == "heads"
            and (args.pump_during_spinup or args.pumped_initial_heads_path)
            else str(heads_path.resolve()) if initial_condition == "heads" else None
        ),
        "spinup_initial_heads": (
            str(args.spinup_initial_heads.resolve()) if args.spinup_initial_heads else None
        ),
        "pump_during_spinup": args.pump_during_spinup,
        "snapshot_dates": args.snapshot_date,
        "configuration": asdict(config),
        "inputs": {name: str(path.resolve()) for name, path in asdict(inputs).items()},
        "input_sha256": {
            name: file_sha256(path)
            for name, path in input_paths.items()
            if path is not None
        },
        "workflow_config": (
            str(args.workflow_config.resolve()) if args.workflow_config else None
        ),
        "wells": str(args.wells.resolve()) if args.wells else None,
        "pumping_schedule": (
            str(args.pumping_schedule.resolve()) if args.pumping_schedule else None
        ),
        "recharge": {
            "source_type": (
                "spatial_raster_manifest"
                if args.recharge_raster_manifest is not None
                else "basin_mean_csv"
            ),
            "input": str(
                (args.recharge_raster_manifest or args.recharge_csv).resolve()
            ),
            "spatial": args.recharge_raster_manifest is not None,
            "spatial_alignment": (
                "area-average reprojection to the groundwater grid; values at active "
                "aquifer cells must be finite and nonnegative"
                if args.recharge_raster_manifest is not None
                else None
            ),
        },
        "pumping": pumping_metadata,
        "grid": {
            "shape": list(state.grid.shape),
            "core_nodes": int(state.grid.core_nodes.size),
            "stream_nodes": int(state.stream_indices.size),
            "modeled_recharge_area_m2": state.basin_area_m2,
            "outlet_node": state.outlet_node,
            "outlet_x": float(state.grid.x_of_node[state.outlet_node]),
            "outlet_y": float(state.grid.y_of_node[state.outlet_node]),
        },
        "reach_outputs": reach_output_metadata,
        "streamflow_definition": (
            "total_streamflow_m3d = groundwater_to_stream_m3d + "
            "saturation_excess_m3d"
        ),
        "saturation_excess_definition": (
            "Solver-integrated local surface-water generation plus any recorded "
            "daily availability-roundoff correction; it is routed immediately "
            "without channel storage, lag, or reinfiltration."
            if config.stream_loss_mode == "routed_volume_limited"
            else "Water-balance remainder after recharge, pumping, aquifer-storage "
            "change, and integrated groundwater exchange with fixed-head stream "
            "nodes; it leaves the watershed immediately without reinfiltration."
        ),
        "stream_loss_limiting": (
            "At every adaptive groundwater substep, potential losing-stream exchange "
            "is capped by routed upstream inflow plus local surface-water generation. "
            "Rejected potential loss is diagnosed but is not counted as streamflow."
            if config.stream_loss_mode == "routed_volume_limited"
            else "Legacy unlimited two-way fixed-head exchange."
        ),
        "landlab_surface_flux_diagnostic": (
            "landlab_saturation_excess_m3d preserves the time-integrated Landlab "
            "surface-discharge field as a numerical diagnostic; it is not the canonical "
            "streamflow term."
        ),
    }
    metadata_path = output_dir / (
        f"simulation_metadata_{args.start_date}_to_{args.end_date}.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Groundwater outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
