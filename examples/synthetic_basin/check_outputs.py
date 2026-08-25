#!/usr/bin/env python
"""Check numerical and structural outputs from the synthetic example."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2020-12-31")


def main() -> None:
    depletion = pd.read_csv(
        OUTPUT_DIR / "streamflow_depletion_timeseries.csv",
        parse_dates=["date"],
    )
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    if not depletion["date"].equals(pd.Series(expected_dates, name="date")):
        raise RuntimeError("The synthetic depletion table has missing or extra dates.")

    numeric = depletion.select_dtypes(include=["number"])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("The synthetic depletion table contains non-finite values.")

    scheduled = float(depletion["scheduled_pumping_m3d"].sum())
    allocated = float(depletion["allocated_pumping_m3d"].sum())
    cumulative_depletion = float(
        depletion["cumulative_streamflow_depletion_m3"].iloc[-1]
    )
    storage_change = float(
        depletion["aquifer_storage_depletion_change_m3"].iloc[-1]
    )
    modeled_extraction = float(
        depletion["cumulative_modeled_extraction_m3"].iloc[-1]
    )
    if scheduled <= 0.0:
        raise RuntimeError("The synthetic example applied no pumping.")
    if not np.isclose(allocated, scheduled, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("The synthetic example did not allocate its pumping schedule.")
    if cumulative_depletion <= 0.5 * scheduled:
        raise RuntimeError("The synthetic example produced no substantial flow response.")
    if not np.isclose(
        cumulative_depletion + storage_change,
        modeled_extraction,
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise RuntimeError("The paired pumping-response balance does not close.")

    for branch in ("unimpaired", "with_pumping"):
        path = OUTPUT_DIR / f"simulation_{branch}_2020-01-01_to_2020-12-31.csv"
        frame = pd.read_csv(path)
        maximum_error = float(frame["mass_balance_error_m3d"].abs().max())
        if maximum_error > 1.0e-6:
            raise RuntimeError(
                f"The {branch} branch exceeds the mass-balance tolerance: "
                f"{maximum_error:g} m3/day."
            )

    reach_daily = pd.read_parquet(OUTPUT_DIR / "reach_daily.parquet")
    if reach_daily["date"].nunique() != len(expected_dates):
        raise RuntimeError("The reach table does not cover every modeled date.")
    routed_columns = [
        "routed_unimpaired_total_streamflow_m3d",
        "routed_pumped_total_streamflow_m3d",
    ]
    if (reach_daily[routed_columns].to_numpy() < -1.0e-10).any():
        raise RuntimeError("The reach table contains negative routed streamflow.")

    print(
        "Synthetic output check passed: "
        f"{len(expected_dates)} days, "
        f"{reach_daily['reach_id'].nunique()} reaches, "
        f"{scheduled:.0f} m3 scheduled pumping, "
        f"{cumulative_depletion:.0f} m3 cumulative streamflow depletion."
    )


if __name__ == "__main__":
    main()
