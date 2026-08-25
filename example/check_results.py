#!/usr/bin/env python3
"""Check the integrity and internal consistency of the Green Valley results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


EXPECTED_REACH_COUNT = 39
REFERENCE_DIR = Path(__file__).resolve().parent / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(output_dir: Path) -> int:
    manifest = output_dir / "SHA256SUMS"
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        path = output_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing published result: {filename}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {filename}")
        checked += 1
    return checked


def compare_with_reference(output_dir: Path, basin: pd.DataFrame, reaches: pd.DataFrame) -> None:
    if output_dir == REFERENCE_DIR.resolve() or not REFERENCE_DIR.exists():
        return
    reference_basin = pd.read_csv(
        REFERENCE_DIR / "streamflow_depletion_timeseries.csv",
        parse_dates=["date"],
    ).set_index("date")
    candidate_basin = basin.set_index("date")
    reference_basin = reference_basin.loc[candidate_basin.index]
    basin_columns = [
        "unimpaired_total_streamflow_m3d",
        "pumped_total_streamflow_m3d",
        "total_streamflow_depletion_m3d",
        "scheduled_pumping_m3d",
        "allocated_pumping_m3d",
        "pumping_m3d",
        "aquifer_storage_depletion_m3",
    ]
    for column in basin_columns:
        if not np.allclose(
            candidate_basin[column], reference_basin[column], rtol=1e-10, atol=1e-6
        ):
            raise RuntimeError(f"New run differs from the published reference: {column}.")

    reference_reaches = pd.read_parquet(REFERENCE_DIR / "reach_daily.parquet")
    reference_reaches = reference_reaches[
        reference_reaches["date"].isin(reaches["date"])
    ].sort_values(["reach_id", "date"])
    reach_columns = [column for column in reaches if column.endswith("_m3d")]
    for column in reach_columns:
        if not np.allclose(
            reaches[column].to_numpy(),
            reference_reaches[column].to_numpy(),
            rtol=1e-10,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"New reach results differ from the published reference: {column}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=REFERENCE_DIR,
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Check a newly generated run that does not have a SHA256SUMS manifest.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    file_count = None if args.skip_hashes else check_hashes(output_dir)

    metadata_paths = sorted(output_dir.glob("simulation_metadata*.json"))
    if not metadata_paths:
        raise RuntimeError("No simulation metadata file was produced.")
    metadata = json.loads(metadata_paths[-1].read_text())
    start_date = pd.Timestamp(metadata["simulation_start"])
    end_date = pd.Timestamp(metadata["simulation_end"])

    basin = pd.read_csv(
        output_dir / "streamflow_depletion_timeseries.csv",
        parse_dates=["date"],
    ).sort_values("date")
    expected_dates = pd.date_range(start_date, end_date, freq="D")
    if not basin["date"].reset_index(drop=True).equals(pd.Series(expected_dates)):
        raise RuntimeError("Basin depletion table does not contain the expected daily dates.")

    reaches = pd.read_parquet(output_dir / "reach_daily.parquet").sort_values(
        ["reach_id", "date"]
    )
    reaches["date"] = pd.to_datetime(reaches["date"])
    reach_ids = reaches["reach_id"].unique()
    if len(reach_ids) != EXPECTED_REACH_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REACH_COUNT} reaches; found {len(reach_ids)}."
        )
    expected_rows = len(expected_dates) * EXPECTED_REACH_COUNT
    if len(reaches) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} reach-day rows; found {len(reaches)}.")
    counts = reaches.groupby("reach_id")["date"].nunique()
    if not (counts == len(expected_dates)).all():
        raise RuntimeError("At least one reach has incomplete daily coverage.")

    reach_geometry = gpd.read_file(output_dir / "reaches.gpkg")
    outlet_rows = reach_geometry.loc[
        reach_geometry["is_outlet_reach"].astype(bool), "reach_id"
    ]
    if len(outlet_rows) != 1:
        raise RuntimeError(f"Expected one outlet reach; found {len(outlet_rows)}.")
    outlet_id = outlet_rows.iloc[0]
    outlet = reaches.loc[reaches["reach_id"] == outlet_id].sort_values("date")

    comparisons = {
        "unimpaired outlet flow": (
            outlet["routed_unimpaired_total_streamflow_m3d"],
            basin["unimpaired_total_streamflow_m3d"],
        ),
        "pumped outlet flow": (
            outlet["routed_pumped_total_streamflow_m3d"],
            basin["pumped_total_streamflow_m3d"],
        ),
        "outlet depletion": (
            outlet["routed_total_streamflow_depletion_m3d"],
            basin["total_streamflow_depletion_m3d"],
        ),
    }
    for label, (reach_values, basin_values) in comparisons.items():
        if not np.allclose(
            reach_values.to_numpy(),
            basin_values.to_numpy(),
            rtol=1e-9,
            atol=1e-6,
            equal_nan=True,
        ):
            raise RuntimeError(f"Mismatch between reach and basin {label}.")

    compare_with_reference(output_dir, basin, reaches)

    print(
        "Green Valley results verified: "
        f"{f'{file_count} files, ' if file_count is not None else ''}"
        f"{len(expected_dates):,} days, "
        f"{EXPECTED_REACH_COUNT} reaches, outlet reach {outlet_id}."
    )


if __name__ == "__main__":
    main()
