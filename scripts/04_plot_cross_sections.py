#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.groundwater import (
    GroundwaterConfig,
    GroundwaterInputs,
    apply_specific_yield_floor,
    apply_well_aquifer_depth,
    build_groundwater_state,
    build_monthly_pumping_maps,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default=Path("data/drainage_area_10m_dem_clipped.tif"), type=Path)
    parser.add_argument("--boundary", default=Path("data/comid_8273277.gpkg"), type=Path)
    parser.add_argument("--transmissivity", default=Path("data/GLYMPHS/transmissivity_m2d.tif"), type=Path)
    parser.add_argument("--depth-to-bedrock", default=Path("data/GLYMPHS/depthToBedrock_m.tif"), type=Path)
    parser.add_argument("--porosity", default=Path("data/GLYMPHS/storativity.tif"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/sim_test_2021_2023"), type=Path)
    parser.add_argument("--date", default="2022-09-30", type=str, help="Date of snapshot to plot")
    parser.add_argument("--target-resolution", default=50.0, type=float)
    parser.add_argument("--target-crs", default="EPSG:26910")
    parser.add_argument("--stream-area-threshold", default=250000.0, type=float)
    parser.add_argument("--stream-drain-offset", default=0.0, type=float)
    parser.add_argument("--additional-aquifer-depth", default=0.0, type=float)
    parser.add_argument(
        "--deep-aquifer-hydraulics",
        choices=["preserve_transmissivity", "preserve_conductivity"],
        default="preserve_transmissivity",
    )
    parser.add_argument("--well-additional-depth", default=0.0, type=float)
    parser.add_argument("--wells", default=None, type=Path)
    parser.add_argument("--pumping-schedule", default=None, type=Path)
    parser.add_argument(
        "--pumping-mode",
        choices=["timeseries", "climatology"],
        default="timeseries",
    )
    parser.add_argument(
        "--pumping-source-mode",
        choices=["well_cell", "topographic"],
        default="well_cell",
    )
    parser.add_argument(
        "--pumping-source-area-threshold",
        default=500000.0,
        type=float,
    )
    parser.add_argument("--specific-yield-floor", default=0.0, type=float)
    args = parser.parse_args()

    output_dir = args.output_dir

    config = GroundwaterConfig(
        target_crs=args.target_crs,
        target_resolution=args.target_resolution,
        stream_area_threshold=args.stream_area_threshold,
        stream_drain_offset=args.stream_drain_offset,
        additional_aquifer_depth=args.additional_aquifer_depth,
        deep_aquifer_hydraulics=args.deep_aquifer_hydraulics,
        well_additional_depth=args.well_additional_depth,
        pumping_source_mode=args.pumping_source_mode,
        pumping_source_area_threshold=args.pumping_source_area_threshold,
        specific_yield_floor=args.specific_yield_floor,
    )
    inputs = GroundwaterInputs(
        dem_path=args.dem,
        basin_path=args.boundary,
        transmissivity_path=args.transmissivity,
        depth_to_bedrock_path=args.depth_to_bedrock,
        porosity_path=args.porosity,
    )

    print("Building groundwater state to get grid and topography...")
    state = build_groundwater_state(inputs, config)
    if args.specific_yield_floor > 0.0:
        apply_specific_yield_floor(state, args.specific_yield_floor)
    if args.well_additional_depth > 0.0:
        if args.wells is None or args.pumping_schedule is None:
            parser.error(
                "Pumping-specific grid settings require --wells and --pumping-schedule."
            )
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
        if args.pumping_source_mode != "well_cell":
            parser.error(
                "--well-additional-depth is only supported with well-cell sources."
            )
        apply_well_aquifer_depth(
            state,
            pumping_forcing.well_nodes,
            args.well_additional_depth,
        )
    grid = state.grid

    # Load snapshots
    wt_unimp_path = output_dir / f"wt_unimpaired_{args.date}.npy"
    wt_pumped_path = output_dir / f"wt_pumped_{args.date}.npy"
    
    if not wt_unimp_path.exists():
        print(f"Error: Unimpaired snapshot for {args.date} not found in {output_dir}")
        return

    wt_unimp = np.load(wt_unimp_path)
    expected_shape = (grid.number_of_nodes,)
    if wt_unimp.shape != expected_shape:
        raise ValueError(
            f"Snapshot has shape {wt_unimp.shape}; expected {expected_shape}. "
            "Use the same grid arguments as the groundwater run."
        )
    pumped_exists = wt_pumped_path.exists()
    
    if pumped_exists:
        wt_pumped = np.load(wt_pumped_path)
        if wt_pumped.shape != expected_shape:
            raise ValueError(
                f"Pumped snapshot has shape {wt_pumped.shape}; expected {expected_shape}."
            )
        depletion = np.full(grid.number_of_nodes, np.nan)
        depletion[grid.core_nodes] = wt_unimp[grid.core_nodes] - wt_pumped[grid.core_nodes]
        max_depletion_node = np.nanargmax(depletion)
        print(f"Max depletion found at ({depletion[max_depletion_node]:.2f} m).")
    else:
        # If no pumping, pick the middle node
        max_depletion_node = int(grid.core_nodes[len(grid.core_nodes) // 2])
        print("No pumped scenario found. Plotting natural cross-section at middle node.")
        wt_pumped = wt_unimp # Dummy for shapes
        
    # Grid properties
    num_cols = grid.number_of_node_columns

    row = max_depletion_node // num_cols
    # Extract east-west transect across the row of maximum depletion
    nodes_in_transect = np.arange(row * num_cols, (row + 1) * num_cols)
    
    topo = grid.at_node["topographic__elevation"][nodes_in_transect]
    base = grid.at_node["aquifer_base__elevation"][nodes_in_transect]
    wt_u = wt_unimp[nodes_in_transect]
    wt_p = wt_pumped[nodes_in_transect]
    modeled = grid.status_at_node[nodes_in_transect] != grid.BC_NODE_IS_CLOSED

    # Filter out inactive nodes for plotting
    x_dist = grid.x_of_node[nodes_in_transect]
    
    # We only want to plot the active part of the basin
    valid_idx = np.where(modeled)[0]
    if len(valid_idx) == 0:
        print("No active nodes in this transect!")
        return
        
    start_idx, end_idx = valid_idx[0], valid_idx[-1]
    
    x_dist = x_dist[start_idx:end_idx+1]
    topo = topo[start_idx:end_idx+1]
    base = base[start_idx:end_idx+1]
    wt_u = wt_u[start_idx:end_idx+1]
    wt_p = wt_p[start_idx:end_idx+1]
    modeled = modeled[start_idx:end_idx+1]
    x_dist = x_dist - x_dist[0]
    topo[~modeled] = np.nan
    base[~modeled] = np.nan
    wt_u[~modeled] = np.nan
    wt_p[~modeled] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_dist, topo, color="black", lw=1.5, label="Topography")
    ax.plot(x_dist, base, color="saddlebrown", lw=1.5, label="Aquifer Base", linestyle="--")
    
    ax.plot(x_dist, wt_u, color="blue", lw=2, label="Unimpaired Water Table")
    
    if pumped_exists:
        ax.plot(x_dist, wt_p, color="red", lw=2, label="Pumped Water Table")
        ax.fill_between(x_dist, wt_p, wt_u, color="red", alpha=0.3, label="Depletion Cone")
        ax.fill_between(x_dist, base, wt_p, color="blue", alpha=0.1, label="Remaining Groundwater")
    else:
        ax.fill_between(x_dist, base, wt_u, color="blue", alpha=0.1, label="Groundwater")

    ax.set_title(f"East-West Groundwater Cross-Section ({args.date})")
    ax.set_xlabel("Distance East (m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_plot = output_dir / f"cross_section_{args.date}.png"
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)
    print(f"Saved cross-section plot to {out_plot}")

if __name__ == "__main__":
    main()
