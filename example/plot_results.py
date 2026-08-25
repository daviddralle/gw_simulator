#!/usr/bin/env python3
"""Rebuild the figures displayed in the Green Valley example README."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd

from gw_simulator.groundwater import (
    save_depletion_timeseries_figure,
    save_hydrograph_figure,
)
from gw_simulator.reach_plots import save_reach_summary_map


EXAMPLE_DIR = Path(__file__).resolve().parent
DRY_SEASON_YEARS = (2017, 2018, 2021)
DRY_SEASON_LABELS = ("Wet reference", "Median impairment", "Extreme impairment")
GV01_REACH_ID = 36


def _line_segments(reaches: gpd.GeoDataFrame) -> list[np.ndarray]:
    segments = []
    for geometry in reaches.geometry:
        if geometry.geom_type == "LineString":
            segments.append(np.asarray(geometry.coords, dtype=float))
        else:
            longest = max(geometry.geoms, key=lambda part: part.length)
            segments.append(np.asarray(longest.coords, dtype=float))
    return segments


def _set_map_extent(axis, boundary: gpd.GeoDataFrame) -> None:
    west, south, east, north = boundary.total_bounds
    pad_x = 0.035 * (east - west)
    pad_y = 0.035 * (north - south)
    axis.set_xlim(west - pad_x, east + pad_x)
    axis.set_ylim(south - pad_y, north + pad_y)
    axis.set_aspect("equal")
    axis.set_axis_off()


def save_dry_season_contrasts(
    metrics: pd.DataFrame,
    reaches: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot routed depletion for three representative dry seasons."""
    reach_ids = reaches["reach_id"].to_numpy()
    segments = _line_segments(reaches)
    upstream_area = reaches["upstream_area_m2"].to_numpy(dtype=float)
    area_scale = np.sqrt(upstream_area / np.nanmax(upstream_area))
    line_widths = 1.3 + 2.8 * area_scale
    norm = Normalize(vmin=0.0, vmax=100.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.72")

    fig = plt.figure(figsize=(12.4, 6.1))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.065),
        left=0.035,
        right=0.985,
        bottom=0.17,
        top=0.81,
        wspace=0.09,
        hspace=0.13,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    color_axis = fig.add_subplot(grid[1, :])

    outlet_rows = reaches.loc[reaches["is_outlet_reach"].astype(bool), "reach_id"]
    if len(outlet_rows) != 1:
        raise ValueError("Expected exactly one outlet reach.")
    outlet_id = int(outlet_rows.iloc[0])

    for axis, year, label in zip(
        axes, DRY_SEASON_YEARS, DRY_SEASON_LABELS, strict=True
    ):
        selected = metrics.loc[
            (metrics["dry_season_year"] == year) & metrics["complete_season"]
        ].set_index("reach_id")
        values = selected.reindex(reach_ids)[
            "integrated_routed_streamflow_depletion_fraction_pct"
        ].to_numpy(dtype=float)
        outlet_value = float(
            selected.loc[
                outlet_id, "integrated_routed_streamflow_depletion_fraction_pct"
            ]
        )

        boundary.plot(
            ax=axis,
            facecolor="0.98",
            edgecolor="0.50",
            linewidth=0.9,
            zorder=1,
        )
        axis.add_collection(
            LineCollection(
                segments,
                colors="0.79",
                linewidths=line_widths + 0.8,
                zorder=2,
            )
        )
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=line_widths,
            zorder=3,
        )
        collection.set_array(np.ma.masked_invalid(values))
        axis.add_collection(collection)
        axis.set_title(
            f"{label}, {year}\nOutlet depletion: {outlet_value:.1f}%",
            fontsize=11.5,
            pad=11,
        )
        _set_map_extent(axis, boundary)

    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        "Routed streamflow depletion (% of unimpaired June–October volume)",
        labelpad=5,
    )
    fig.suptitle(
        "June–October depletion across the stream network",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.045,
        "All 39 modeled reaches. Gray reaches had ≤1 m³ of modeled unimpaired "
        "June–October flow, so their depletion fractions are undefined. Line width "
        "scales with upstream drainage area.",
        ha="center",
        color="0.35",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def save_response_distribution(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the distribution of June–October routed depletion among reaches."""
    local_depletion = daily.groupby("reach_id", sort=True)[
        "local_total_streamflow_depletion_m3d"
    ].sum()
    data = summary.set_index("reach_id").join(
        local_depletion.rename("cumulative_local_depletion_m3")
    )
    data = data.reset_index()
    data["upstream_area_km2"] = data["upstream_area_m2"] / 1.0e6
    data = data.sort_values(
        "integrated_routed_streamflow_depletion_fraction_pct",
        ascending=False,
    ).reset_index(drop=True)
    data["rank"] = np.arange(1, len(data) + 1)

    color_values = data["cumulative_local_depletion_m3"].to_numpy(dtype=float)
    positive = color_values[color_values > 0.0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    cmap = plt.get_cmap("viridis")
    near_zero_days = np.clip(
        data["total_pumping_created_near_zero_flow_days"].to_numpy(dtype=float),
        0.0,
        None,
    )
    maximum_days = max(float(np.nanmax(near_zero_days)), 1.0)
    point_sizes = 36.0 + 110.0 * np.sqrt(near_zero_days / maximum_days)
    fractions = data[
        "integrated_routed_streamflow_depletion_fraction_pct"
    ].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.25), sharey=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.82,
        bottom=0.28,
        wspace=0.15,
    )
    axes[0].plot(data["rank"], fractions, color="0.65", linewidth=1.0, zorder=1)
    axes[0].scatter(
        data["rank"],
        fractions,
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=36,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    axes[1].scatter(
        data["upstream_area_km2"],
        fractions,
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=point_sizes,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.9,
        zorder=2,
    )

    panel_b_label_offsets = {18: (7, 7), 4: (7, 6), 6: (7, -10)}
    for _, row in data.head(3).iterrows():
        reach_id = int(row["reach_id"])
        fraction = row["integrated_routed_streamflow_depletion_fraction_pct"]
        axes[0].annotate(
            f"R{reach_id}",
            (row["rank"], fraction),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8.5,
        )
        axes[1].annotate(
            f"R{reach_id}",
            (row["upstream_area_km2"], fraction),
            xytext=panel_b_label_offsets[reach_id],
            textcoords="offset points",
            fontsize=8.5,
        )

    outlet_rows = data.loc[data["is_outlet_reach"].astype(bool)]
    if len(outlet_rows) != 1:
        raise ValueError("Expected exactly one outlet reach.")
    special_reaches = (
        (GV01_REACH_ID, "GV01 reach", "D", "#187a3a"),
        (int(outlet_rows.iloc[0]["reach_id"]), "Outlet reach", "*", "black"),
    )
    for reach_id, label, marker, color in special_reaches:
        row = data.loc[data["reach_id"] == reach_id]
        if row.empty:
            raise ValueError(f"Reach {reach_id} is missing from the summary table.")
        row = row.iloc[0]
        for axis, x_value in (
            (axes[0], row["rank"]),
            (axes[1], row["upstream_area_km2"]),
        ):
            axis.scatter(
                [x_value],
                [row["integrated_routed_streamflow_depletion_fraction_pct"]],
                marker=marker,
                s=105 if marker == "D" else 125,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
                label=label if axis is axes[1] else None,
            )

    axes[0].set_title("(a) Ranked depletion fractions", pad=10)
    axes[0].set_xlabel("Reach rank (largest fraction first)")
    axes[0].set_ylabel("Routed June–October depletion (%)")
    axes[0].set_xlim(0, len(data) + 1)
    axes[1].set_title("(b) Depletion and upstream area", pad=10)
    axes[1].set_xlabel("Upstream drainage area (km²; log scale)")
    axes[1].set_xscale("log")
    axes[1].legend(loc="upper right", frameon=False)
    for axis in axes:
        axis.set_ylim(-2.0, 102.0)
        axis.grid(alpha=0.25)

    color_axis = fig.add_axes((0.22, 0.135, 0.62, 0.035))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Cumulative reach-local depletion (m³; log color scale)")
    fig.suptitle(
        "Reach-scale June–October streamflow depletion",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.045,
        "Each point is one reach. Routed fractions include upstream contributions; "
        "reach-local depletion is non-overlapping among reaches. Point size in (b) "
        "scales with pumping-created near-zero-flow days.",
        ha="center",
        color="0.35",
        fontsize=8.8,
        wrap=True,
    )
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def rebuild_figures(inputs_dir: Path, results_dir: Path) -> None:
    natural_path = next(results_dir.glob("simulation_unimpaired_*.csv"))
    pumped_path = next(results_dir.glob("simulation_with_pumping_*.csv"))
    natural = pd.read_csv(natural_path, parse_dates=["date"])
    pumped = pd.read_csv(pumped_path, parse_dates=["date"])
    depletion = pd.read_csv(
        results_dir / "streamflow_depletion_timeseries.csv", parse_dates=["date"]
    )
    start_date = natural["date"].min().strftime("%Y-%m-%d")
    end_date = natural["date"].max().strftime("%Y-%m-%d")
    date_slug = f"{start_date}_to_{end_date}"
    recharge = natural[["date", "recharge_mm_day"]].rename(
        columns={"recharge_mm_day": "Recharge"}
    )

    daily = pd.read_parquet(results_dir / "reach_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    reaches = gpd.read_file(results_dir / "reaches.gpkg", layer="reaches").sort_values(
        "reach_id"
    )
    boundary = gpd.read_file(inputs_dir / "boundary.gpkg").to_crs(reaches.crs)
    metrics = pd.read_csv(results_dir / "reach_dry_season_metrics_by_year.csv")
    summary = pd.read_csv(results_dir / "reach_dry_season_summary.csv")

    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    ):
        save_hydrograph_figure(
            natural,
            recharge,
            results_dir / f"hydrographs_{date_slug}.png",
            start_date=start_date,
            end_date=end_date,
            pumped=pumped,
        )
        save_depletion_timeseries_figure(
            depletion,
            results_dir / f"depletion_timeseries_{date_slug}.png",
            start_date=start_date,
            end_date=end_date,
        )
        save_reach_summary_map(
            daily,
            reaches,
            boundary,
            results_dir / "network_reach_depletion_summary.png",
        )
        save_dry_season_contrasts(
            metrics,
            reaches,
            boundary,
            results_dir / "network_reach_dry_season_contrasts.png",
        )
        save_response_distribution(
            daily,
            summary,
            results_dir / "network_reach_response_distribution.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=EXAMPLE_DIR / "inputs",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXAMPLE_DIR / "results",
    )
    args = parser.parse_args()
    rebuild_figures(args.inputs_dir.resolve(), args.results_dir.resolve())
    print(f"Rebuilt Green Valley README figures in {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
