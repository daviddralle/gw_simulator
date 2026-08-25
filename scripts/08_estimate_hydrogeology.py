#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.groundwater import (
    GroundwaterConfig,
    GroundwaterInputs,
    build_groundwater_state,
)
from gw_simulator.hydrogeology_estimation import (
    RecessionFilter,
    aggregate_streamflow_to_daily,
    block_bootstrap_kirchner_signature,
    convert_streamflow_to_mm_day,
    derive_parameter_triplet,
    estimate_kirchner_signature,
    expand_monthly_pumping_schedule,
    export_local_well_context,
    prepare_recession_pairs,
    rorabaugh_diffusivity_m2_day,
    summarize_well_context,
)
from gw_simulator.workflow import load_workflow_config


DEFAULT_WELL_DIR = Path("data/external/usgs_russian_river_boreholes")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _groundwater_config(workflow) -> GroundwaterConfig:
    values = workflow.values.get("groundwater", {})
    pumping = workflow.values.get("pumping", {})
    return GroundwaterConfig(
        target_crs=values.get("target_crs", "EPSG:26910"),
        target_resolution=float(values.get("target_resolution_m", 50.0)),
        stream_area_threshold=float(values.get("stream_area_threshold_m2", 250000.0)),
        pumping_source_mode=pumping.get("source_mode", "well_cell"),
        pumping_source_area_threshold=float(
            pumping.get("source_area_threshold_m2", 500000.0)
        ),
    )


def _basin_geometry_context(
    workflow,
    *,
    outlet_longitude: float | None = None,
    outlet_latitude: float | None = None,
) -> tuple[dict[str, object], object, np.ndarray | None]:
    inputs = GroundwaterInputs(
        dem_path=workflow.path_value("dem"),
        basin_path=workflow.path_value("boundary"),
        transmissivity_path=workflow.path_value("transmissivity"),
        depth_to_bedrock_path=workflow.path_value("depth_to_bedrock"),
        porosity_path=workflow.path_value("porosity"),
    )
    if any(value is None for value in inputs.__dict__.values()):
        raise ValueError("Workflow lacks a required groundwater raster or boundary.")
    state = build_groundwater_state(inputs, _groundwater_config(workflow))
    grid = state.grid
    stream_nodes = np.asarray(state.stream_indices, dtype=int)
    core_nodes = np.asarray(grid.core_nodes, dtype=int)
    outlet_context = None
    upstream = None
    if outlet_longitude is not None or outlet_latitude is not None:
        if outlet_longitude is None or outlet_latitude is None:
            raise ValueError("Gage longitude and latitude must be provided together.")
        point = gpd.GeoSeries(
            gpd.points_from_xy([outlet_longitude], [outlet_latitude]), crs="EPSG:4326"
        ).to_crs(_groundwater_config(workflow).target_crs).iloc[0]
        stream_tree = cKDTree(
            np.column_stack(
                (grid.x_of_node[stream_nodes], grid.y_of_node[stream_nodes])
            )
        )
        snap_distance, snap_index = stream_tree.query([[point.x, point.y]], k=1)
        outlet_node = int(stream_nodes[int(snap_index[0])])

        receiver = np.asarray(grid.at_node["flow__receiver_node"], dtype=int)
        donors: list[list[int]] = [[] for _ in range(grid.number_of_nodes)]
        for donor, target in enumerate(receiver):
            if donor != target:
                donors[int(target)].append(donor)
        upstream = np.zeros(grid.number_of_nodes, dtype=bool)
        stack = [outlet_node]
        while stack:
            node = stack.pop()
            if upstream[node]:
                continue
            upstream[node] = True
            stack.extend(donors[node])
        core_nodes = core_nodes[upstream[core_nodes]]
        stream_nodes = stream_nodes[upstream[stream_nodes]]
        outlet_context = {
            "longitude": outlet_longitude,
            "latitude": outlet_latitude,
            "snapped_stream_node": outlet_node,
            "snap_distance_m": float(snap_distance[0]),
            "landlab_drainage_area_km2": float(
                grid.at_node["drainage_area"][outlet_node] / 1.0e6
            ),
        }
    if len(core_nodes) == 0 or len(stream_nodes) == 0:
        raise ValueError("No modeled subcatchment was found for the supplied gage.")
    tree = cKDTree(
        np.column_stack((grid.x_of_node[stream_nodes], grid.y_of_node[stream_nodes]))
    )
    nearest_distance = tree.query(
        np.column_stack((grid.x_of_node[core_nodes], grid.y_of_node[core_nodes])), k=1
    )[0]

    receiver = np.asarray(grid.at_node["flow__receiver_node"], dtype=int)
    is_stream = np.zeros(grid.number_of_nodes, dtype=bool)
    is_stream[stream_nodes] = True
    downstream = receiver[stream_nodes]
    internal = (downstream != stream_nodes) & is_stream[downstream]
    source_nodes = stream_nodes[internal]
    target_nodes = downstream[internal]
    stream_length_m = float(
        np.sum(
            np.hypot(
                grid.x_of_node[source_nodes] - grid.x_of_node[target_nodes],
                grid.y_of_node[source_nodes] - grid.y_of_node[target_nodes],
            )
        )
    )
    if outlet_context is None:
        area_m2 = float(state.basin_area_m2)
    else:
        outlet_context["upstream_core_cell_area_km2"] = float(
            len(core_nodes) * grid.dx * grid.dy / 1.0e6
        )
        area_m2 = float(outlet_context["landlab_drainage_area_km2"] * 1.0e6)
    distances = {
        "mean": float(np.mean(nearest_distance)),
        "p25": float(np.quantile(nearest_distance, 0.25)),
        "p50": float(np.quantile(nearest_distance, 0.50)),
        "p75": float(np.quantile(nearest_distance, 0.75)),
        "p90": float(np.quantile(nearest_distance, 0.90)),
        "p95": float(np.quantile(nearest_distance, 0.95)),
    }
    candidate_lengths = {
        "area_over_twice_stream_length": area_m2 / (2.0 * stream_length_m),
        "twice_mean_nearest_stream": 2.0 * distances["mean"],
        "p90_nearest_stream": distances["p90"],
    }
    result = {
        "modeled_active_area_km2": area_m2 / 1.0e6,
        "stream_area_threshold_m2": float(state.stream_area_threshold),
        "stream_node_count": int(len(stream_nodes)),
        "stream_length_km": stream_length_m / 1000.0,
        "nearest_stream_distance_m": distances,
        "candidate_rorabaugh_flow_lengths_m": candidate_lengths,
        "flow_length_note": (
            "These are DEM/network-scale approximations to the transverse "
            "stream-to-divide distance a, not measured groundwater flow paths."
        ),
    }
    if outlet_context is not None:
        result["gage_outlet"] = outlet_context
    return result, state, upstream


def _borehole_ids_in_upstream_mask(
    information_csv: Path,
    *,
    boundary_path: Path,
    state,
    upstream: np.ndarray,
    target_crs: str,
) -> list[object]:
    information = pd.read_csv(information_csv, encoding="cp1252")
    points = gpd.GeoDataFrame(
        information,
        geometry=gpd.points_from_xy(information["Longitude"], information["Latitude"]),
        crs="EPSG:4326",
    )
    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    points = points.loc[
        points.geometry.intersects(boundary.geometry.union_all())
    ].to_crs(target_crs)
    grid = state.grid
    tree = cKDTree(np.column_stack((grid.x_of_node, grid.y_of_node)))
    nodes = tree.query(np.column_stack((points.geometry.x, points.geometry.y)), k=1)[1]
    return points.loc[upstream[nodes], "BoreID"].tolist()


def _boundary_area_m2(boundary_path: Path) -> float:
    boundary = gpd.read_file(boundary_path)
    if boundary.empty or boundary.crs is None:
        raise ValueError("Boundary must contain geometry with a CRS.")
    return float(boundary.to_crs("EPSG:5070").geometry.area.sum())


def _read_streamflow(
    path: Path,
    *,
    date_column: str,
    flow_column: str,
    flow_units: str,
    basin_area_m2: float,
    min_daily_coverage: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path, usecols=[date_column, flow_column])
    daily, quality = aggregate_streamflow_to_daily(
        frame,
        date_column=date_column,
        flow_column=flow_column,
        min_daily_coverage=min_daily_coverage,
    )
    daily["q_mm_day"] = convert_streamflow_to_mm_day(
        daily["flow"], units=flow_units, basin_area_m2=basin_area_m2
    )
    return daily.loc[:, ["date", "q_mm_day"]], quality


def _deficit_storage_constraint(
    forcing: pd.DataFrame,
    *,
    extractable_fraction_min: float,
    extractable_fraction_max: float,
) -> dict[str, object]:
    if not 0.0 < extractable_fraction_min <= extractable_fraction_max <= 1.0:
        raise ValueError(
            "Root-storage fractions must satisfy 0 < minimum <= maximum <= 1."
        )
    if "Deficit" not in forcing:
        raise ValueError("Forcing lacks the Deficit field.")
    work = forcing.loc[:, ["date", "Deficit"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["Deficit"] = pd.to_numeric(work["Deficit"], errors="coerce")
    work = work.dropna().sort_values("date")
    work["water_year"] = work["date"].dt.year + (
        work["date"].dt.month >= 10
    ).astype(int)
    water_year = work.groupby("water_year")["Deficit"].agg(["min", "max"])
    water_year_range = water_year["max"] - water_year["min"]
    dry = work.loc[work["date"].dt.month.isin((5, 6, 7, 8, 9, 10))].copy()
    dry_year = dry.groupby(dry["date"].dt.year)["Deficit"].agg(["first", "last"])
    dry_increase = dry_year["last"] - dry_year["first"]
    dry_quantiles = dry_increase.quantile([0.50, 0.90])
    median_mm = float(dry_quantiles.loc[0.50])
    p90_mm = float(dry_quantiles.loc[0.90])
    maximum_mm = float(work["Deficit"].max())
    return {
        "method": (
            "Basin-mean cumulative P-minus-actual-ET deficit converted to an "
            "equivalent plant-accessible vadose/rock-moisture thickness."
        ),
        "extractable_storage_fraction": [
            extractable_fraction_min,
            extractable_fraction_max,
        ],
        "absolute_maximum_deficit_mm": maximum_mm,
        "water_year_deficit_range_mm": {
            "p50": float(water_year_range.quantile(0.50)),
            "p90": float(water_year_range.quantile(0.90)),
        },
        "dry_season_deficit_increase_mm": {"p50": median_mm, "p90": p90_mm},
        "implied_root_rock_moisture_thickness_m": {
            "central_envelope_from_dry_p50_to_p90": [
                median_mm / 1000.0 / extractable_fraction_max,
                p90_mm / 1000.0 / extractable_fraction_min,
            ],
            "absolute_maximum_deficit_envelope": [
                maximum_mm / 1000.0 / extractable_fraction_max,
                maximum_mm / 1000.0 / extractable_fraction_min,
            ],
        },
        "interpretation": (
            "This is a lower-bound thickness of plant-accessible unsaturated "
            "storage under the assumed extractable fraction. It is not saturated "
            "specific yield, water-table depth, or total aquifer thickness, and "
            "must not be added to Boussinesq storage a second time."
        ),
    }


def _plot_signature(
    bins: pd.DataFrame,
    summary: dict[str, object],
    output: Path,
    source_label: str,
) -> None:
    q = bins["q_mm_day"].to_numpy(dtype=float)
    slope = bins["mean_minus_dqdt_mm_day2"].to_numpy(dtype=float)
    error = bins["se_minus_dqdt_mm_day2"].to_numpy(dtype=float)
    g = bins["g_per_day"].to_numpy(dtype=float)
    fitted_g = bins["fitted_g_per_day"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].errorbar(q, slope, yerr=error, fmt="o", color="#204a87", capsize=2)
    axes[0].plot(q, q * fitted_g, color="#a40000", linewidth=1.5)
    axes[0].set(xscale="log", yscale="log", xlabel="Q (mm/day)", ylabel="-dQ/dt (mm/day²)")

    axes[1].plot(q, g, "o", color="#204a87", label="Binned")
    axes[1].plot(q, fitted_g, color="#a40000", label="Smooth fit")
    axes[1].set(xscale="log", yscale="log", xlabel="Q (mm/day)", ylabel="g(Q) (1/day)")
    axes[1].legend(frameon=False, fontsize=8)

    order = np.argsort(q)
    storage = np.zeros_like(q)
    sorted_q = q[order]
    sorted_g = fitted_g[order]
    storage_sorted = np.concatenate(
        ([0.0], np.cumsum(np.diff(sorted_q) * (1.0 / sorted_g[:-1] + 1.0 / sorted_g[1:]) / 2.0))
    )
    storage[order] = storage_sorted
    axes[2].plot(storage, q, "o-", color="#4e9a06")
    axes[2].set(xlabel="Relative dynamic storage (mm)", ylabel="Q (mm/day)", yscale="log")

    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    for axis in axes[:2]:
        axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        axis.yaxis.set_minor_formatter(NullFormatter())
    axes[2].yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(
        f"Hydrograph signatures: {source_label}\n"
        f"selected wet-season pairs; dynamic storage over fitted range = "
        f"{summary['dynamic_storage_over_fitted_q_range_mm']:.1f} mm",
        fontsize=10,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_well_context(
    geopackage: Path,
    boundary_path: Path,
    output: Path,
) -> None:
    boundary = gpd.read_file(boundary_path).to_crs("EPSG:26910")
    boreholes = gpd.read_file(geopackage, layer="boreholes").to_crs(boundary.crs)
    capacity = gpd.read_file(geopackage, layer="specific_capacity").to_crs(
        boundary.crs
    )
    capacity["specific_capacity"] = pd.to_numeric(
        capacity["Specific Capacity"], errors="coerce"
    )
    capacity = capacity.loc[capacity["specific_capacity"] > 0.0].copy()

    fig, axis = plt.subplots(figsize=(6.5, 6.5))
    boundary.plot(ax=axis, facecolor="#f7f7f7", edgecolor="#222222", linewidth=1.0)
    boreholes.plot(ax=axis, color="#969696", markersize=9, alpha=0.55)
    plotted = axis.scatter(
        capacity.geometry.x,
        capacity.geometry.y,
        c=capacity["specific_capacity"],
        norm=LogNorm(
            vmin=float(capacity["specific_capacity"].min()),
            vmax=float(capacity["specific_capacity"].max()),
        ),
        cmap="viridis",
        s=25,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )
    colorbar = fig.colorbar(plotted, ax=axis, shrink=0.75)
    colorbar.set_label("Specific capacity (gpm/ft; logarithmic scale)")
    axis.set_title(
        f"USGS Russian River boreholes inside model boundary\n"
        f"{len(boreholes)} logs; {len(capacity)} positive specific-capacity records"
    )
    axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Green Valley well evidence and estimate recession, "
            "dynamic-storage, and diffusivity signatures from a supplied hydrograph."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--well-dir", type=Path, default=DEFAULT_WELL_DIR)
    parser.add_argument("--streamflow", type=Path)
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--flow-column", default="total_streamflow_m3d")
    parser.add_argument(
        "--flow-units", choices=("mm_day", "m3_day", "m3_s", "cfs"), default="m3_day"
    )
    parser.add_argument("--source-label", default="supplied hydrograph")
    parser.add_argument(
        "--basin-area-km2",
        type=float,
        help="Contributing area at the gage; defaults to the configured boundary area.",
    )
    parser.add_argument("--gage-longitude", type=float)
    parser.add_argument("--gage-latitude", type=float)
    parser.add_argument("--min-daily-coverage", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=Path("output/hydrogeology_estimation"))
    parser.add_argument("--allowed-months", default="11,12,1,2,3,4")
    parser.add_argument("--antecedent-days", type=int, default=3)
    parser.add_argument("--max-antecedent-precip-mm", type=float, default=0.1)
    parser.add_argument("--max-antecedent-recharge-mm", type=float, default=0.1)
    parser.add_argument("--max-et-mm-day", type=float, default=1.5)
    parser.add_argument("--max-pumping-quantile", type=float, default=0.35)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=731)
    parser.add_argument("--specific-yield-min", type=float)
    parser.add_argument("--specific-yield-max", type=float)
    parser.add_argument("--active-depth-min-m", type=float)
    parser.add_argument("--active-depth-max-m", type=float)
    parser.add_argument("--root-storage-fraction-min", type=float)
    parser.add_argument("--root-storage-fraction-max", type=float)
    args = parser.parse_args()

    workflow = load_workflow_config(args.config)
    boundary_path = workflow.path_value("boundary")
    if boundary_path is None:
        raise ValueError("Workflow lacks a boundary path.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    well_summary = summarize_well_context(
        information_csv=args.well_dir / "RRW_Borehole_Information_v1.1.csv",
        specific_capacity_csv=args.well_dir / "RRW_Borehole_SpecificCapacity.csv",
        texture_csv=args.well_dir / "RRW_Borehole_Texture.csv",
        boundary_path=boundary_path,
    )
    _write_json(output_dir / "well_context_summary.json", well_summary)
    local_well_path = output_dir / "green_valley_well_context.gpkg"
    local_well_layers = export_local_well_context(
        information_csv=args.well_dir / "RRW_Borehole_Information_v1.1.csv",
        specific_capacity_csv=args.well_dir / "RRW_Borehole_SpecificCapacity.csv",
        texture_csv=args.well_dir / "RRW_Borehole_Texture.csv",
        boundary_path=boundary_path,
        output_path=local_well_path,
    )
    _plot_well_context(
        local_well_path,
        boundary_path,
        output_dir / "well_context_map.png",
    )

    geometry, groundwater_state, upstream_mask = _basin_geometry_context(
        workflow,
        outlet_longitude=args.gage_longitude,
        outlet_latitude=args.gage_latitude,
    )
    _write_json(output_dir / "basin_geometry_summary.json", geometry)

    parameter_well_summary = well_summary
    parameter_well_source = "full configured Green Valley model boundary"
    subcatchment_well_path = None
    if upstream_mask is not None:
        borehole_ids = _borehole_ids_in_upstream_mask(
            args.well_dir / "RRW_Borehole_Information_v1.1.csv",
            boundary_path=boundary_path,
            state=groundwater_state,
            upstream=upstream_mask,
            target_crs=_groundwater_config(workflow).target_crs,
        )
        subcatchment_well_summary = summarize_well_context(
            information_csv=args.well_dir / "RRW_Borehole_Information_v1.1.csv",
            specific_capacity_csv=args.well_dir / "RRW_Borehole_SpecificCapacity.csv",
            texture_csv=args.well_dir / "RRW_Borehole_Texture.csv",
            borehole_ids=borehole_ids,
        )
        _write_json(
            output_dir / "subcatchment_well_context_summary.json",
            subcatchment_well_summary,
        )
        subcatchment_well_path = output_dir / "subcatchment_well_context.gpkg"
        export_local_well_context(
            information_csv=args.well_dir / "RRW_Borehole_Information_v1.1.csv",
            specific_capacity_csv=args.well_dir / "RRW_Borehole_SpecificCapacity.csv",
            texture_csv=args.well_dir / "RRW_Borehole_Texture.csv",
            borehole_ids=borehole_ids,
            output_path=subcatchment_well_path,
        )
        if subcatchment_well_summary["local"]["specific_capacity_test_count"] >= 5:
            parameter_well_summary = subcatchment_well_summary
            parameter_well_source = "DEM-delineated gage subcatchment"

    result = {
        "well_context": str(output_dir / "well_context_summary.json"),
        "local_well_data": str(local_well_path),
        "local_well_layers": local_well_layers,
        "local_well_map": str(output_dir / "well_context_map.png"),
        "basin_geometry": str(output_dir / "basin_geometry_summary.json"),
        "subcatchment_well_data": (
            str(subcatchment_well_path) if subcatchment_well_path is not None else None
        ),
        "hydrograph_signature": None,
    }
    if args.streamflow is not None:
        if args.basin_area_km2 is not None:
            basin_area_m2 = args.basin_area_km2 * 1.0e6
            basin_area_source = "command-line contributing area"
        elif args.gage_longitude is not None:
            basin_area_m2 = geometry["modeled_active_area_km2"] * 1.0e6
            basin_area_source = "DEM-delineated modeled active area at gage"
        else:
            basin_area_m2 = _boundary_area_m2(boundary_path)
            basin_area_source = "configured boundary area"
        flow, streamflow_quality = _read_streamflow(
            args.streamflow,
            date_column=args.date_column,
            flow_column=args.flow_column,
            flow_units=args.flow_units,
            basin_area_m2=basin_area_m2,
            min_daily_coverage=args.min_daily_coverage,
        )
        forcing_path = workflow.path_value("recharge_csv")
        pumping_path = workflow.path_value("pumping_schedule")
        if forcing_path is None or pumping_path is None:
            raise ValueError("Workflow lacks recharge forcing or pumping schedule.")
        forcing = pd.read_csv(
            forcing_path, usecols=["date", "ET", "P", "Deficit", "Recharge"]
        )
        pumping = expand_monthly_pumping_schedule(
            pumping_path,
            start_date=flow["date"].min(),
            end_date=flow["date"].max(),
        )
        recession_filter = RecessionFilter(
            allowed_months=tuple(int(value) for value in args.allowed_months.split(",")),
            antecedent_days=args.antecedent_days,
            max_antecedent_precip_mm=args.max_antecedent_precip_mm,
            max_antecedent_recharge_mm=args.max_antecedent_recharge_mm,
            max_et_mm_day=args.max_et_mm_day,
            max_pumping_quantile=args.max_pumping_quantile,
        )
        pairs, selection = prepare_recession_pairs(
            flow, forcing=forcing, pumping=pumping, config=recession_filter
        )
        bins, signature = estimate_kirchner_signature(
            pairs,
            bin_count=recession_filter.bin_count,
            min_bin_count=recession_filter.min_bin_count,
        )
        if args.bootstrap_replicates > 0:
            signature["water_year_block_bootstrap"] = (
                block_bootstrap_kirchner_signature(
                    pairs,
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                    bin_count=recession_filter.bin_count,
                    min_bin_count=recession_filter.min_bin_count,
                )
            )
        flow_lengths = geometry["candidate_rorabaugh_flow_lengths_m"]
        signature["rorabaugh_diffusivity_m2_day"] = {
            name: float(
                rorabaugh_diffusivity_m2_day(
                    signature["low_flow_log_cycle_recession_index_days"], length
                )
            )
            for name, length in flow_lengths.items()
        }
        transmissivity_priors = {
            "p25_low": parameter_well_summary["local"]["screening_transmissivity_low_m2_day"]["p25"],
            "p50_low": parameter_well_summary["local"]["screening_transmissivity_low_m2_day"]["p50"],
            "p50_high": parameter_well_summary["local"]["screening_transmissivity_high_m2_day"]["p50"],
            "p75_high": parameter_well_summary["local"]["screening_transmissivity_high_m2_day"]["p75"],
        }
        combinations = []
        for transmissivity_label, transmissivity in transmissivity_priors.items():
            for diffusivity_label, diffusivity in signature[
                "rorabaugh_diffusivity_m2_day"
            ].items():
                combined = derive_parameter_triplet(
                    transmissivity_m2_day=transmissivity,
                    diffusivity_m2_day=diffusivity,
                    dynamic_storage_mm=signature[
                        "dynamic_storage_over_fitted_q_range_mm"
                    ],
                )
                combinations.append(
                    {
                        "transmissivity_prior": transmissivity_label,
                        "flow_length_method": diffusivity_label,
                        "transmissivity_m2_day": float(
                            combined["transmissivity_m2_day"]
                        ),
                        "diffusivity_m2_day": float(combined["diffusivity_m2_day"]),
                        "specific_yield": float(combined["specific_yield"]),
                        "active_saturated_thickness_change_m": float(
                            combined["effective_depth_m"]
                        ),
                    }
                )
        signature["parameter_combinations"] = {
            "status": (
                "An inference only when the hydrograph source is observed; model-output "
                "results are method verification."
            ),
            "interpretation": (
                "Specific yield is T/(T/Sy). Active saturated-thickness change is "
                "dynamic storage/Sy over the fitted discharge range; it is not total "
                "depth to bedrock."
            ),
            "transmissivity_priors_m2_day": transmissivity_priors,
            "transmissivity_prior_source": parameter_well_source,
            "transmissivity_specific_capacity_test_count": parameter_well_summary[
                "local"
            ]["specific_capacity_test_count"],
            "combinations": combinations,
        }
        if args.specific_yield_min is not None or args.specific_yield_max is not None:
            if args.specific_yield_min is None or args.specific_yield_max is None:
                raise ValueError(
                    "Specific-yield minimum and maximum must be provided together."
                )
            if not 0.0 < args.specific_yield_min <= args.specific_yield_max <= 1.0:
                raise ValueError("Specific-yield bounds must satisfy 0 < min <= max <= 1.")
            if args.active_depth_min_m is not None or args.active_depth_max_m is not None:
                if args.active_depth_min_m is None or args.active_depth_max_m is None:
                    raise ValueError(
                        "Active-depth minimum and maximum must be provided together."
                    )
                active_depth_range = [
                    args.active_depth_min_m,
                    args.active_depth_max_m,
                ]
                active_depth_source = "command-line prior"
            elif upstream_mask is not None:
                depth_summary = parameter_well_summary["local"]["total_depth_m"]
                active_depth_range = [depth_summary["p25"], depth_summary["p90"]]
                active_depth_source = (
                    "p25--p90 total depths of wells in DEM-delineated gage subcatchment"
                )
            else:
                active_depth_range = None
                active_depth_source = None

            prior_combinations = []
            for specific_yield in (
                args.specific_yield_min,
                0.5 * (args.specific_yield_min + args.specific_yield_max),
                args.specific_yield_max,
            ):
                for flow_length_method, diffusivity in signature[
                    "rorabaugh_diffusivity_m2_day"
                ].items():
                    transmissivity = specific_yield * diffusivity
                    item = {
                        "specific_yield": specific_yield,
                        "flow_length_method": flow_length_method,
                        "diffusivity_m2_day": diffusivity,
                        "transmissivity_m2_day": transmissivity,
                        "dynamic_water_table_change_m": signature[
                            "dynamic_storage_over_fitted_q_range_mm"
                        ]
                        / 1000.0
                        / specific_yield,
                    }
                    if active_depth_range is not None:
                        item["bulk_conductivity_m_day"] = [
                            transmissivity / active_depth_range[1],
                            transmissivity / active_depth_range[0],
                        ]
                    prior_combinations.append(item)
            transmissivities = [
                item["transmissivity_m2_day"] for item in prior_combinations
            ]
            water_table_changes = [
                item["dynamic_water_table_change_m"] for item in prior_combinations
            ]
            conductivities = [
                value
                for item in prior_combinations
                for value in item.get("bulk_conductivity_m_day", [])
            ]
            signature["specific_yield_prior_inference"] = {
                "status": "preferred parameterization direction",
                "specific_yield_prior": [
                    args.specific_yield_min,
                    args.specific_yield_max,
                ],
                "active_depth_prior_m": active_depth_range,
                "active_depth_prior_source": active_depth_source,
                "central_transmissivity_envelope_m2_day": [
                    min(transmissivities),
                    max(transmissivities),
                ],
                "central_dynamic_water_table_change_envelope_m": [
                    min(water_table_changes),
                    max(water_table_changes),
                ],
                "central_bulk_conductivity_envelope_m_day": (
                    [min(conductivities), max(conductivities)]
                    if conductivities
                    else None
                ),
                "combinations": prior_combinations,
                "caution": (
                    "Diffusivity remains conditional on Rorabaugh geometry and the "
                    "selected low-flow reference; bootstrap uncertainty is reported "
                    "separately. Well total depth is evidence of active rock, not a "
                    "direct observation of the aquifer base."
                ),
            }
            signature["parameter_combinations"]["status"] = (
                "diagnostic rejected direction: do not infer Sy by combining short "
                "specific-capacity T with recession diffusivity when an independent "
                "specific-yield prior is available"
            )
        signature_payload = {
            "source": str(args.streamflow.resolve()),
            "source_label": args.source_label,
            "observational_status": (
                "Observed only if the supplied source is an observed discharge record; "
                "model output is a method verification, not an estimate."
            ),
            "basin_area_km2_used_for_unit_conversion": basin_area_m2 / 1.0e6,
            "basin_area_source": basin_area_source,
            "streamflow_quality": streamflow_quality,
            "selection": selection,
            "signature": signature,
            "bins": bins.to_dict(orient="records"),
        }
        if (
            args.root_storage_fraction_min is not None
            or args.root_storage_fraction_max is not None
        ):
            if (
                args.root_storage_fraction_min is None
                or args.root_storage_fraction_max is None
            ):
                raise ValueError(
                    "Root-storage fraction minimum and maximum must be supplied together."
                )
            signature_payload["deficit_storage_constraint"] = (
                _deficit_storage_constraint(
                    forcing,
                    extractable_fraction_min=args.root_storage_fraction_min,
                    extractable_fraction_max=args.root_storage_fraction_max,
                )
            )
        signature_path = output_dir / "hydrograph_signature_summary.json"
        _write_json(signature_path, signature_payload)
        _plot_signature(
            bins,
            signature,
            output_dir / "hydrograph_signature.png",
            args.source_label,
        )
        result["hydrograph_signature"] = str(signature_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
