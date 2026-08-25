#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.workflow import (
    load_workflow_config,
    recharge_input_path,
    recharge_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=REPO_ROOT, check=True)


def _path(config, key: str) -> str:
    value = config.path_value(key)
    if value is None:
        raise ValueError(f"Workflow config lacks required path `{key}`.")
    return str(value)


def _append_value(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def prepare_dem(config, *, refresh: bool) -> None:
    dem_path = config.path_value("dem")
    if dem_path is None:
        raise ValueError("Workflow config lacks required path `dem`.")
    if dem_path.exists() and not refresh:
        print(f"DEM already exists: {dem_path}", flush=True)
        return
    earth_engine = config.values.get("earth_engine", {})
    dem = config.values.get("dem_extraction", {})
    command = [
        sys.executable,
        "scripts/01_extract_dem.py",
        "--boundary",
        _path(config, "boundary"),
        "--output-tif",
        str(dem_path),
        "--scale",
        str(dem.get("scale_m", 10.0)),
    ]
    _append_value(command, "--ee-project", earth_engine.get("project"))
    _run(command)


def prepare_hydrogeology(config) -> None:
    model_inputs = [
        config.path_value("transmissivity"),
        config.path_value("depth_to_bedrock"),
        config.path_value("porosity"),
    ]
    if all(path is not None and path.exists() for path in model_inputs):
        paths = ", ".join(str(path) for path in model_inputs)
        print(f"Using existing hydrogeology inputs: {paths}", flush=True)
        return
    command = [
        sys.executable,
        "scripts/00_prepare_hydrogeology.py",
        "--glhymps-zip",
        _path(config, "glhymps_zip"),
        "--boundary",
        _path(config, "boundary"),
        "--reference-raster",
        _path(config, "dem"),
        "--output-dir",
        _path(config, "hydrogeology_output_dir"),
    ]
    _append_value(command, "--existing-porosity", config.path_value("legacy_porosity"))
    _append_value(command, "--existing-depth", config.path_value("legacy_depth_to_bedrock"))
    _append_value(
        command,
        "--existing-transmissivity",
        config.path_value("legacy_transmissivity"),
    )
    _append_value(command, "--depth-source", config.path_value("shangguan_depth_source"))
    _append_value(command, "--pelletier-regolith", config.path_value("pelletier_regolith"))
    _append_value(command, "--pelletier-sediment", config.path_value("pelletier_sediment"))
    _append_value(command, "--pelletier-land-cover", config.path_value("pelletier_land_cover"))
    _run(command)


def prepare_recharge(config, *, refresh: bool) -> None:
    recharge = config.values.get("recharge", {})
    source = recharge_source(config)
    input_path = recharge_input_path(config) if source != "earth_engine_spatial_deficit" else None
    if source in {"csv", "raster_manifest"}:
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(f"Configured recharge input does not exist: {input_path}")
        print(f"Using user-supplied recharge input: {input_path}", flush=True)
        return
    if source == "earth_engine_spatial_deficit":
        raise NotImplementedError(
            "Earth Engine spatial-deficit recharge extraction is not implemented yet. "
            "Provide a `raster_manifest` input for spatial recharge."
        )

    earth_engine = config.values.get("earth_engine", {})
    groundwater = config.values["groundwater"]
    spinup = config.values.get("spinup", {})
    forcing_dates = [
        str(groundwater["start_date"]),
        str(groundwater["end_date"]),
    ]
    if spinup:
        forcing_dates.extend([str(spinup["start_date"]), str(spinup["end_date"])])
    default_start_year = int(min(forcing_dates)[:4])
    default_end_year = int(max(forcing_dates)[:4])
    command = [
        sys.executable,
        "scripts/02_compute_recharge.py",
        "--boundary",
        _path(config, "boundary"),
        "--output-csv",
        _path(config, "recharge_csv"),
        "--start-year",
        str(recharge.get("start_year", default_start_year)),
        "--end-year",
        str(recharge.get("end_year", default_end_year)),
        "--initial-deficit-mm",
        str(recharge.get("initial_deficit_mm", 0.0)),
        "--cache-dir",
        _path(config, "forcing_cache"),
    ]
    _append_value(command, "--ee-project", earth_engine.get("project"))
    if refresh:
        command.append("--refresh")
    _run(command)


def run_groundwater(config) -> None:
    groundwater = config.values["groundwater"]
    spinup = config.values.get("spinup", {})
    pumping = config.values.get("pumping", {})
    command = [
        sys.executable,
        "scripts/03_run_groundwater.py",
        "--dem",
        _path(config, "dem"),
        "--boundary",
        _path(config, "boundary"),
        "--transmissivity",
        _path(config, "transmissivity"),
        "--depth-to-bedrock",
        _path(config, "depth_to_bedrock"),
        "--porosity",
        _path(config, "porosity"),
        "--output-dir",
        _path(config, "output_dir"),
        "--workflow-config",
        str(config.path),
        "--start-date",
        str(groundwater["start_date"]),
        "--end-date",
        str(groundwater["end_date"]),
        "--target-crs",
        str(groundwater.get("target_crs", "EPSG:26910")),
        "--target-resolution",
        str(groundwater.get("target_resolution_m", 50.0)),
        "--stream-area-threshold",
        str(groundwater.get("stream_area_threshold_m2", 250000.0)),
        "--courant-coefficient",
        str(groundwater.get("courant_coefficient", 0.5)),
        "--stream-drain-offset",
        str(groundwater.get("stream_drain_offset_m", 0.0)),
        "--additional-aquifer-depth",
        str(groundwater.get("additional_aquifer_depth_m", 0.0)),
        "--deep-aquifer-hydraulics",
        str(groundwater.get("deep_aquifer_hydraulics", "preserve_transmissivity")),
        "--specific-yield-floor",
        str(groundwater.get("specific_yield_floor", 0.0)),
        "--stream-loss-mode",
        str(groundwater.get("stream_loss_mode", "routed_volume_limited")),
        "--stream-limiter-tolerance-m3",
        str(groundwater.get("stream_limiter_tolerance_m3", 1.0e-6)),
        "--stream-limiter-max-iterations",
        str(groundwater.get("stream_limiter_max_iterations", 25)),
    ]
    recharge_path = recharge_input_path(config)
    if recharge_source(config) == "raster_manifest":
        command.extend(["--recharge-raster-manifest", str(recharge_path)])
    else:
        command.extend(["--recharge-csv", str(recharge_path)])
    if spinup:
        command.extend(
            [
                "--spinup-start-date",
                str(spinup["start_date"]),
                "--spinup-end-date",
                str(spinup["end_date"]),
            ]
        )
        spinup_initial_heads = config.path_value("spinup_initial_heads")
        if spinup_initial_heads is not None:
            command.extend(
                ["--spinup-initial-heads", str(spinup_initial_heads)]
            )
        if spinup.get("pumping", False):
            command.append("--pump-during-spinup")
    else:
        command.append("--skip-spinup")
        initial_heads = config.path_value("initial_heads")
        pumped_initial_heads = config.path_value("pumped_initial_heads")
        if initial_heads is not None:
            command.extend(["--initial-heads-path", str(initial_heads)])
        if pumped_initial_heads is not None:
            command.extend(["--pumped-initial-heads-path", str(pumped_initial_heads)])

    if pumping.get("enabled", False):
        command.extend(
            [
                "--wells",
                _path(config, "wells"),
                "--pumping-schedule",
                _path(config, "pumping_schedule"),
                "--pumping-mode",
                str(pumping.get("mode", "timeseries")),
                "--pumping-source-mode",
                str(pumping.get("source_mode", "well_cell")),
                "--pumping-source-area-threshold",
                str(pumping.get("source_area_threshold_m2", 500000.0)),
                "--source-zone-storage-fraction",
                str(pumping.get("source_zone_storage_fraction", 0.5)),
                "--pumping-storage-fraction",
                str(pumping.get("pumping_storage_fraction", 0.5)),
            ]
        )
        if pumping.get("strict_supply", False):
            command.append("--strict-pumping-supply")

    for date in config.values.get("outputs", {}).get("snapshot_dates", []):
        command.extend(["--snapshot-date", str(date)])
    _run(command)


def plot_cross_sections(config) -> None:
    groundwater = config.values["groundwater"]
    pumping = config.values.get("pumping", {})
    for date in config.values.get("outputs", {}).get("cross_section_dates", []):
        command = [
            sys.executable,
            "scripts/04_plot_cross_sections.py",
            "--dem",
            _path(config, "dem"),
            "--boundary",
            _path(config, "boundary"),
            "--transmissivity",
            _path(config, "transmissivity"),
            "--depth-to-bedrock",
            _path(config, "depth_to_bedrock"),
            "--porosity",
            _path(config, "porosity"),
            "--output-dir",
            _path(config, "output_dir"),
            "--date",
            str(date),
            "--target-resolution",
            str(groundwater.get("target_resolution_m", 50.0)),
            "--target-crs",
            str(groundwater.get("target_crs", "EPSG:26910")),
            "--stream-area-threshold",
            str(groundwater.get("stream_area_threshold_m2", 250000.0)),
            "--stream-drain-offset",
            str(groundwater.get("stream_drain_offset_m", 0.0)),
            "--pumping-source-mode",
            str(pumping.get("source_mode", "well_cell")),
            "--pumping-source-area-threshold",
            str(pumping.get("source_area_threshold_m2", 500000.0)),
        ]
        _run(command)


def plot_reach_depletion(config) -> None:
    outputs = config.values.get("outputs", {})
    command = [
        sys.executable,
        "scripts/05_plot_reach_depletion.py",
        "--output-dir",
        _path(config, "output_dir"),
        "--boundary",
        _path(config, "boundary"),
        "--rolling-days",
        str(outputs.get("reach_fraction_rolling_days", 30)),
        "--frame-step-days",
        str(outputs.get("reach_video_frame_step_days", 7)),
        "--frames-per-second",
        str(outputs.get("reach_video_frames_per_second", 6)),
    ]
    for water_year in outputs.get("reach_video_water_years", []):
        command.extend(["--water-year", str(water_year)])
    _run(command)


def extract_dry_season_metrics(config) -> None:
    _run(
        [
            sys.executable,
            "scripts/06_extract_dry_season_metrics.py",
            "--output-dir",
            _path(config, "output_dir"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured CEFF groundwater workflow.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=[
            "dem",
            "hydrogeology",
            "recharge",
            "preflight",
            "groundwater",
            "plots",
            "metrics",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--refresh-dem", action="store_true")
    parser.add_argument("--refresh-forcing", action="store_true")
    args = parser.parse_args()
    config = load_workflow_config(args.config)
    outputs = config.values.get("outputs", {})

    if args.stage in {"dem", "all"}:
        prepare_dem(config, refresh=args.refresh_dem)
    if args.stage in {"hydrogeology", "all"}:
        prepare_hydrogeology(config)
    if args.stage in {"recharge", "all"}:
        prepare_recharge(config, refresh=args.refresh_forcing)
    if args.stage in {"preflight", "all"}:
        _run([sys.executable, "scripts/00_preflight.py", "--config", str(config.path)])
    if args.stage in {"groundwater", "all"}:
        run_groundwater(config)
    if args.stage in {"plots", "all"}:
        plot_cross_sections(config)
        if (
            config.values.get("pumping", {}).get("enabled", False)
            and outputs.get("reach_visualizations", True)
        ):
            plot_reach_depletion(config)
    if args.stage == "metrics" or (
        args.stage == "all" and outputs.get("dry_season_metrics", True)
    ):
        extract_dry_season_metrics(config)


if __name__ == "__main__":
    main()
