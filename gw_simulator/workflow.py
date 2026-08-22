from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from .recharge import boundary_fingerprint, load_recharge_raster_manifest


RECHARGE_SOURCES = {
    "earth_engine_deficit",
    "csv",
    "raster_manifest",
    "earth_engine_spatial_deficit",
}


PATH_KEYS = {
    "boundary",
    "dem",
    "recharge_csv",
    "recharge_raster_manifest",
    "forcing_cache",
    "transmissivity",
    "depth_to_bedrock",
    "porosity",
    "wells",
    "pumping_schedule",
    "output_dir",
    "initial_heads",
    "pumped_initial_heads",
    "spinup_initial_heads",
    "glhymps_zip",
    "hydrogeology_output_dir",
    "shangguan_depth_source",
    "pelletier_regolith",
    "pelletier_sediment",
    "pelletier_land_cover",
}


@dataclass(frozen=True)
class WorkflowConfig:
    path: Path
    root: Path
    values: dict[str, Any]

    def path_value(self, key: str) -> Path | None:
        value = self.values.get(key)
        if value is None:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    path = Path(path).resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Workflow config must contain a YAML mapping: {path}")
    if values.get("version") != 1:
        raise ValueError("Workflow config must set `version: 1`.")
    for required in ("name", "boundary", "dem", "groundwater"):
        if required not in values:
            raise ValueError(f"Workflow config lacks `{required}`.")
    recharge = values.setdefault("recharge", {})
    if not isinstance(recharge, dict):
        raise ValueError("Workflow `recharge` must be a YAML mapping.")
    source = str(recharge.get("source", "earth_engine_deficit"))
    if source not in RECHARGE_SOURCES:
        allowed = ", ".join(sorted(RECHARGE_SOURCES))
        raise ValueError(f"Unknown recharge source `{source}`; use one of: {allowed}.")
    recharge["source"] = source
    if source == "earth_engine_deficit":
        forcing_root = f"data/forcing/{values['name']}"
        values.setdefault("recharge_csv", f"{forcing_root}/recharge.csv")
        values.setdefault("forcing_cache", f"{forcing_root}/pml_v22a_prism")
    elif source == "csv" and "recharge_csv" not in values:
        raise ValueError("Recharge source `csv` requires top-level `recharge_csv`.")
    elif source == "raster_manifest" and "recharge_raster_manifest" not in values:
        raise ValueError(
            "Recharge source `raster_manifest` requires top-level "
            "`recharge_raster_manifest`."
        )
    root_value = Path(values.get("root", ".")).expanduser()
    root = root_value if root_value.is_absolute() else (path.parent / root_value).resolve()
    return WorkflowConfig(path=path, root=root, values=values)


def recharge_source(config: WorkflowConfig) -> str:
    """Return the configured recharge source, including the basin-mean default."""
    return str(config.values.get("recharge", {}).get("source", "earth_engine_deficit"))


def recharge_input_path(config: WorkflowConfig) -> Path:
    """Return the basin CSV or spatial-manifest path consumed by groundwater."""
    source = recharge_source(config)
    key = "recharge_raster_manifest" if source == "raster_manifest" else "recharge_csv"
    path = config.path_value(key)
    if path is None:
        raise ValueError(f"Recharge source `{source}` lacks required path `{key}`.")
    return path


def _projected_bounds(
    bounds: tuple[float, float, float, float], source_crs, target_crs
) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    min_x, min_y, max_x, max_y = bounds
    points = [
        transformer.transform(x, y)
        for x in np.linspace(min_x, max_x, 11)
        for y in (min_y, max_y)
    ] + [
        transformer.transform(x, y)
        for x in (min_x, max_x)
        for y in np.linspace(min_y, max_y, 11)
    ]
    x_values, y_values = zip(*points)
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _raster_covers_boundary(path: Path, boundary: gpd.GeoDataFrame) -> dict[str, Any]:
    with rasterio.open(path) as source:
        if source.crs is None:
            return {"path": str(path), "covers_boundary_bbox": False, "error": "missing CRS"}
        bounds = boundary.to_crs(source.crs).total_bounds
        covered = (
            bounds[0] >= source.bounds.left
            and bounds[1] >= source.bounds.bottom
            and bounds[2] <= source.bounds.right
            and bounds[3] <= source.bounds.top
        )
        return {
            "path": str(path),
            "covers_boundary_bbox": bool(covered),
            "crs": str(source.crs),
            "shape": [source.height, source.width],
            "resolution": [abs(source.res[0]), abs(source.res[1])],
        }


def _forcing_check(
    path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=["date", "Recharge"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["Recharge"] = pd.to_numeric(frame["Recharge"], errors="coerce")
    duplicate_count = int(frame["date"].duplicated().sum())
    invalid_count = int(frame[["date", "Recharge"]].isna().any(axis=1).sum())
    negative_count = int((frame["Recharge"] < 0.0).sum())
    requested = pd.date_range(start_date, end_date, freq="D")
    available = pd.DatetimeIndex(frame["date"].dropna().unique())
    missing = requested.difference(available)
    return {
        "source": "basin_mean",
        "path": str(path),
        "first_date": frame["date"].min().strftime("%Y-%m-%d"),
        "last_date": frame["date"].max().strftime("%Y-%m-%d"),
        "record_count": int(len(frame)),
        "duplicate_date_count": duplicate_count,
        "invalid_record_count": invalid_count,
        "negative_recharge_count": negative_count,
        "requested_first_date": start_date.strftime("%Y-%m-%d"),
        "requested_last_date": end_date.strftime("%Y-%m-%d"),
        "missing_requested_day_count": int(len(missing)),
        "missing_requested_day_examples": [value.strftime("%Y-%m-%d") for value in missing[:5]],
    }


def _raster_forcing_check(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    boundary: gpd.GeoDataFrame,
) -> dict[str, Any]:
    frame = load_recharge_raster_manifest(
        path,
        start_date,
        end_date,
        inspect_rasters=True,
    )
    unique_paths = list(dict.fromkeys(Path(value) for value in frame["raster_path"]))
    coverage = [_raster_covers_boundary(value, boundary) for value in unique_paths]
    uncovered = [item["path"] for item in coverage if not item["covers_boundary_bbox"]]
    return {
        "source": "spatial_raster_manifest",
        "path": str(path),
        "first_date": frame["date"].min().strftime("%Y-%m-%d"),
        "last_date": frame["date"].max().strftime("%Y-%m-%d"),
        "record_count": int(len(frame)),
        "duplicate_date_count": 0,
        "invalid_record_count": 0,
        "negative_recharge_count": 0,
        "requested_first_date": start_date.strftime("%Y-%m-%d"),
        "requested_last_date": end_date.strftime("%Y-%m-%d"),
        "missing_requested_day_count": 0,
        "missing_requested_day_examples": [],
        "unique_raster_file_count": len(unique_paths),
        "units": sorted(frame["units"].unique().tolist()),
        "raster_files_not_covering_boundary_count": len(uncovered),
        "raster_files_not_covering_boundary_examples": uncovered[:5],
    }


def _snapshot_dates(config: WorkflowConfig) -> list[pd.Timestamp]:
    output = config.values.get("outputs", {})
    return sorted(pd.Timestamp(value) for value in output.get("snapshot_dates", []))


def preflight_workflow(config: WorkflowConfig) -> dict[str, Any]:
    """Check spatial/temporal coverage and estimate groundwater grid costs."""
    values = config.values
    groundwater = values["groundwater"]
    boundary_path = config.path_value("boundary")
    dem_path = config.path_value("dem")
    source = recharge_source(config)
    if source == "earth_engine_spatial_deficit":
        raise NotImplementedError(
            "Earth Engine spatial-deficit recharge is reserved for a future extractor. "
            "Use `earth_engine_deficit`, `csv`, or `raster_manifest` for now."
        )
    recharge_path = recharge_input_path(config)
    if boundary_path is None or dem_path is None:
        raise ValueError("Boundary, DEM, and recharge paths are required.")

    required_paths = {
        "boundary": boundary_path,
        "dem": dem_path,
        (
            "recharge_raster_manifest"
            if source == "raster_manifest"
            else "recharge_csv"
        ): recharge_path,
        "transmissivity": config.path_value("transmissivity"),
        "depth_to_bedrock": config.path_value("depth_to_bedrock"),
        "porosity": config.path_value("porosity"),
    }
    pumping = values.get("pumping", {})
    if pumping.get("enabled", False):
        required_paths["wells"] = config.path_value("wells")
        required_paths["pumping_schedule"] = config.path_value("pumping_schedule")
    missing_paths = [name for name, path in required_paths.items() if path is None or not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing workflow input(s): {', '.join(missing_paths)}")

    boundary = gpd.read_file(boundary_path)
    if boundary.empty or boundary.crs is None:
        raise ValueError("Boundary must contain geometry with a CRS.")
    boundary_area_m2 = float(boundary.to_crs("EPSG:5070").geometry.area.sum())
    target_crs = groundwater.get("target_crs", "EPSG:26910")
    resolution = float(groundwater.get("target_resolution_m", 50.0))
    if resolution <= 0.0:
        raise ValueError("Groundwater target resolution must be positive.")

    with rasterio.open(dem_path) as dem:
        if dem.crs is None:
            raise ValueError("DEM has no CRS.")
        grid_bounds = _projected_bounds(dem.bounds, dem.crs, target_crs)
    width_m = grid_bounds[2] - grid_bounds[0]
    height_m = grid_bounds[3] - grid_bounds[1]
    columns = int(width_m / resolution)
    rows = int(height_m / resolution)
    node_count = rows * columns
    if rows < 3 or columns < 3:
        raise ValueError("Configured resolution produces a groundwater grid smaller than 3 by 3.")
    rectangle_area_m2 = node_count * resolution**2
    overhead_ratio = rectangle_area_m2 / boundary_area_m2

    start_date = pd.Timestamp(groundwater["start_date"])
    end_date = pd.Timestamp(groundwater["end_date"])
    if end_date < start_date:
        raise ValueError("Groundwater end date precedes its start date.")
    spinup = values.get("spinup", {})
    spinup_start = pd.Timestamp(spinup.get("start_date", start_date))
    spinup_end = pd.Timestamp(spinup.get("end_date", start_date - pd.Timedelta(days=1)))
    forcing_start = min(start_date, spinup_start)
    forcing_end = max(end_date, spinup_end)

    raster_checks = {
        name: _raster_covers_boundary(path, boundary)
        for name, path in required_paths.items()
        if name in {"dem", "transmissivity", "depth_to_bedrock", "porosity"}
        and path is not None
    }
    forcing = (
        _raster_forcing_check(recharge_path, forcing_start, forcing_end, boundary)
        if source == "raster_manifest"
        else _forcing_check(recharge_path, forcing_start, forcing_end)
    )
    forcing["configured_source"] = source
    days = int((end_date - start_date).days + 1)
    scenarios = 2 if pumping.get("enabled", False) else 1
    snapshot_dates = _snapshot_dates(config)

    # Landlab component fields vary by version. This is a deliberately conservative
    # planning estimate, not a measurement of peak resident memory.
    estimated_grid_memory_gb = node_count * 2500 / 1024**3
    snapshot_storage_gb = node_count * 8 * len(snapshot_dates) * scenarios / 1024**3
    warnings: list[str] = []
    errors: list[str] = []
    if overhead_ratio > 3.0:
        warnings.append(
            f"The rectangular grid is {overhead_ratio:.1f} times the mapped basin area."
        )
    if estimated_grid_memory_gb > float(values.get("limits", {}).get("warn_memory_gb", 8.0)):
        warnings.append(
            f"Estimated grid working memory is {estimated_grid_memory_gb:.1f} GB."
        )
    if node_count > int(values.get("limits", {}).get("warn_node_count", 2_000_000)):
        warnings.append(f"Groundwater grid has approximately {node_count:,} nodes.")
    for name, check in raster_checks.items():
        if not check["covers_boundary_bbox"]:
            errors.append(f"{name} raster does not cover the boundary bounding box.")
    if forcing["missing_requested_day_count"]:
        errors.append(
            f"Recharge is missing {forcing['missing_requested_day_count']} requested day(s)."
        )
    if forcing["duplicate_date_count"] or forcing["invalid_record_count"] or forcing["negative_recharge_count"]:
        errors.append("Recharge contains duplicate, invalid, or negative records.")
    if forcing.get("raster_files_not_covering_boundary_count", 0):
        errors.append("One or more recharge rasters do not cover the boundary bounding box.")
    if any(date < start_date or date > end_date for date in snapshot_dates):
        errors.append("All configured snapshot dates must fall within the simulation period.")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_name": values["name"],
        "config": str(config.path),
        "boundary": {
            "path": str(boundary_path),
            "fingerprint_sha256": boundary_fingerprint(boundary_path),
            "area_km2": boundary_area_m2 / 1e6,
        },
        "grid": {
            "target_crs": target_crs,
            "target_resolution_m": resolution,
            "estimated_shape": [rows, columns],
            "estimated_node_count": node_count,
            "rectangle_area_km2": rectangle_area_m2 / 1e6,
            "rectangle_to_basin_area_ratio": overhead_ratio,
            "estimated_working_memory_gb": estimated_grid_memory_gb,
        },
        "simulation": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "day_count": days,
            "scenario_count": scenarios,
            "snapshot_dates": [value.strftime("%Y-%m-%d") for value in snapshot_dates],
            "estimated_snapshot_storage_gb": snapshot_storage_gb,
        },
        "forcing": forcing,
        "rasters": raster_checks,
        "warnings": warnings,
        "errors": errors,
        "ready": not errors,
    }


def write_preflight(report: dict[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output
