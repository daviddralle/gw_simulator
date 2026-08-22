#!/usr/bin/env python3
"""Migrate completed reach outputs to total-only local and routed schema v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_simulator.groundwater import (
    REACH_DEFINITION_VERSION,
    REACH_OUTPUT_SCHEMA_VERSION,
    route_reach_daily_table,
)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    daily_path = output_dir / "reach_daily.parquet"
    reaches_path = output_dir / "reaches.gpkg"
    metadata_paths = sorted(output_dir.glob("simulation_metadata_*.json"))
    natural_paths = sorted(output_dir.glob("simulation_unimpaired_*.csv"))
    pumped_paths = sorted(output_dir.glob("simulation_with_pumping_*.csv"))
    if not all((daily_path.exists(), reaches_path.exists())) or not metadata_paths:
        raise FileNotFoundError("Output directory lacks standard reach files or metadata.")
    if not natural_paths or not pumped_paths:
        raise FileNotFoundError("Paired basin simulation CSVs are required for validation.")

    old_daily = pd.read_parquet(daily_path)
    old_daily["date"] = pd.to_datetime(old_daily["date"])
    local_columns = [
        "date",
        "reach_id",
        "unimpaired_local_total_streamflow_m3d",
        "pumped_local_total_streamflow_m3d",
        "local_total_streamflow_depletion_m3d",
    ]
    missing = set(local_columns).difference(old_daily.columns)
    if missing:
        raise ValueError("Reach table lacks total-flow columns: " + ", ".join(sorted(missing)))
    daily = old_daily[local_columns].copy()
    denominator = daily["unimpaired_local_total_streamflow_m3d"].to_numpy()
    fraction = np.full(len(daily), np.nan, dtype=float)
    np.divide(
        100.0 * daily["local_total_streamflow_depletion_m3d"].to_numpy(),
        denominator,
        out=fraction,
        where=denominator > 0.0,
    )
    daily["local_streamflow_depletion_fraction_pct"] = fraction

    old_reaches = gpd.read_file(reaches_path, layer="reaches").sort_values("reach_id")
    downstream = np.asarray(
        [int(value) - 1 if pd.notna(value) else -1 for value in old_reaches["downstream_reach_id"]],
        dtype=int,
    )
    daily = route_reach_daily_table(daily, downstream)

    natural = pd.read_csv(natural_paths[-1], parse_dates=["date"]).sort_values("date")
    pumped = pd.read_csv(pumped_paths[-1], parse_dates=["date"]).sort_values("date")
    outlet_id = int(old_reaches.loc[old_reaches["downstream_reach_id"].isna(), "reach_id"].iloc[0])
    validation: dict[str, float] = {}
    for prefix, basin in (("unimpaired", natural), ("pumped", pumped)):
        local_sum = daily.groupby("date", sort=True)[
            f"{prefix}_local_total_streamflow_m3d"
        ].sum().to_numpy()
        basin_flow = basin["total_streamflow_m3d"].to_numpy()
        routed_outlet = daily.loc[
            daily["reach_id"] == outlet_id,
            f"routed_{prefix}_total_streamflow_m3d",
        ].to_numpy()
        validation[f"maximum_{prefix}_daily_aggregation_error_m3d"] = float(
            np.max(np.abs(local_sum - basin_flow))
        )
        validation[f"maximum_{prefix}_routed_outlet_error_m3d"] = float(
            np.max(np.abs(routed_outlet - basin_flow))
        )
    basin_depletion = (
        natural["total_streamflow_m3d"].to_numpy()
        - pumped["total_streamflow_m3d"].to_numpy()
    )
    local_depletion = daily.groupby("date", sort=True)[
        "local_total_streamflow_depletion_m3d"
    ].sum().to_numpy()
    routed_depletion = daily.loc[
        daily["reach_id"] == outlet_id,
        "routed_total_streamflow_depletion_m3d",
    ].to_numpy()
    validation["maximum_daily_depletion_aggregation_error_m3d"] = float(
        np.max(np.abs(local_depletion - basin_depletion))
    )
    validation["maximum_routed_outlet_depletion_error_m3d"] = float(
        np.max(np.abs(routed_depletion - basin_depletion))
    )
    tolerance = 1.0e-8 * max(float(np.max(np.abs(natural["total_streamflow_m3d"]))), 1.0)
    if max(validation.values()) > tolerance:
        raise RuntimeError(f"Migrated reach table failed basin closure: {validation}")

    definition_sha256 = str(old_reaches["definition_sha256"].iloc[0])
    table = pa.Table.from_pandas(daily, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"reach_output_schema_version": REACH_OUTPUT_SCHEMA_VERSION.encode(),
            b"reach_definition_version": REACH_DEFINITION_VERSION.encode(),
            b"reach_definition_sha256": definition_sha256.encode(),
            b"spatial_semantics": (
                b"Total streamflow only. Local columns exclude upstream inflow. Routed "
                b"columns integrate the local total-flow contribution from the reach and "
                b"all upstream reaches; no channel lag or loss is applied."
            ),
        }
    )
    pq.write_table(table.replace_schema_metadata(metadata), daily_path, compression="zstd")

    static_columns = [
        "reach_id",
        "downstream_reach_id",
        "is_outlet_reach",
        "stream_node_count",
        "reach_length_m",
        "incremental_area_m2",
        "upstream_area_m2",
        "definition_version",
        "definition_sha256",
        "geometry",
    ]
    reaches = old_reaches[static_columns].copy()
    natural_summary = daily.groupby("reach_id", sort=True).agg(
        cumulative_unimpaired_local_streamflow_m3=("unimpaired_local_total_streamflow_m3d", "sum"),
        mean_unimpaired_local_streamflow_m3d=("unimpaired_local_total_streamflow_m3d", "mean"),
    )
    depletion_summary = daily.groupby("reach_id", sort=True).agg(
        cumulative_local_depletion_m3=("local_total_streamflow_depletion_m3d", "sum"),
        cumulative_routed_depletion_m3=("routed_total_streamflow_depletion_m3d", "sum"),
        mean_daily_local_depletion_m3d=("local_total_streamflow_depletion_m3d", "mean"),
        minimum_daily_local_depletion_m3d=("local_total_streamflow_depletion_m3d", "min"),
        maximum_daily_local_depletion_m3d=("local_total_streamflow_depletion_m3d", "max"),
        mean_daily_routed_depletion_m3d=("routed_total_streamflow_depletion_m3d", "mean"),
        maximum_daily_routed_depletion_m3d=("routed_total_streamflow_depletion_m3d", "max"),
    )
    reaches = reaches.merge(natural_summary, on="reach_id", how="left")
    reaches = reaches.merge(depletion_summary, on="reach_id", how="left")
    reaches_path.unlink()
    reaches.to_file(reaches_path, layer="reaches", driver="GPKG", index=False)

    # The depletion CSV is a user-facing total-flow product. Remove the legacy
    # pathway-specific depletion columns while preserving water-balance diagnostics.
    for depletion_path in {
        output_dir / "streamflow_depletion_timeseries.csv",
        *output_dir.glob("streamflow_depletion_*_to_*.csv"),
    }:
        if depletion_path.exists():
            frame = pd.read_csv(depletion_path)
            frame = frame.drop(
                columns=["groundwater_depletion_m3d", "saturation_excess_depletion_m3d"],
                errors="ignore",
            )
            frame.to_csv(depletion_path, index=False)

    metadata_path = metadata_paths[-1]
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_metadata["reach_outputs"] = {
        "schema_version": REACH_OUTPUT_SCHEMA_VERSION,
        "definition_version": REACH_DEFINITION_VERSION,
        "definition_sha256": definition_sha256,
        "reach_count": int(daily["reach_id"].nunique()),
        "daily_row_count": len(daily),
        "date_count": int(daily["date"].nunique()),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (daily_path, reaches_path)
        },
        "validation": validation,
        "spatial_semantics": (
            "Total streamflow only. Local columns exclude upstream inflow. Routed columns "
            "integrate each reach's local total-flow contribution with all upstream reaches, "
            "without channel lag or loss. Neither is well or pumping-zone attribution."
        ),
        "migration": {
            "name": "total-flow-routed-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Migrated {len(daily):,} reach-day rows to schema {REACH_OUTPUT_SCHEMA_VERSION}.")
    print(json.dumps(validation, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    migrate(args.output_dir)


if __name__ == "__main__":
    main()
