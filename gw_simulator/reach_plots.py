from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd


REACH_VISUALIZATION_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class ReachVisualizationConfig:
    rolling_days: int = 30
    frame_step_days: int = 7
    frames_per_second: int = 6
    minimum_unimpaired_volume_m3: float = 1.0


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def water_year(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    return dates.year.to_numpy() + (dates.month.to_numpy() >= 10).astype(int)


def add_rolling_depletion_fraction(
    daily: pd.DataFrame,
    config: ReachVisualizationConfig = ReachVisualizationConfig(),
) -> pd.DataFrame:
    """Add an upstream-integrated rolling flow-depletion fraction."""
    required = {
        "date",
        "reach_id",
        "routed_unimpaired_total_streamflow_m3d",
        "routed_total_streamflow_depletion_m3d",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(
            "Reach daily table lacks required columns: " + ", ".join(sorted(missing))
        )
    output = daily.sort_values(["reach_id", "date"]).copy()
    output["date"] = pd.to_datetime(output["date"])
    group = output.groupby("reach_id", sort=False)
    output["rolling_routed_unimpaired_streamflow_m3"] = group[
        "routed_unimpaired_total_streamflow_m3d"
    ].transform(
        lambda values: values.rolling(
            config.rolling_days, min_periods=config.rolling_days
        ).sum()
    )
    output["rolling_routed_depletion_m3"] = group[
        "routed_total_streamflow_depletion_m3d"
    ].transform(
        lambda values: values.rolling(
            config.rolling_days, min_periods=config.rolling_days
        ).sum()
    )
    denominator = output["rolling_routed_unimpaired_streamflow_m3"].to_numpy()
    fraction = np.full(len(output), np.nan, dtype=float)
    np.divide(
        100.0 * output["rolling_routed_depletion_m3"].to_numpy(),
        denominator,
        out=fraction,
        where=denominator > config.minimum_unimpaired_volume_m3,
    )
    output["rolling_routed_depletion_fraction_pct"] = fraction
    output["water_year"] = water_year(output["date"])
    return output.sort_values(["date", "reach_id"]).reset_index(drop=True)


def select_representative_water_years(
    basin_daily: pd.DataFrame,
) -> tuple[list[int], dict[int, float]]:
    """Select the driest and wettest complete water years by modeled recharge."""
    frame = basin_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["water_year"] = water_year(frame["date"])
    summary = frame.groupby("water_year").agg(
        day_count=("date", "size"),
        recharge_m3=("recharge_m3d", "sum"),
    )
    summary = summary[summary["day_count"] >= 365]
    if summary.empty:
        return [], {}
    driest = int(summary["recharge_m3"].idxmin())
    wettest = int(summary["recharge_m3"].idxmax())
    selected = [driest] if driest == wettest else [driest, wettest]
    recharge = {
        int(index): float(value)
        for index, value in summary["recharge_m3"].items()
    }
    return selected, recharge


def _plot_network_values(
    ax,
    boundary: gpd.GeoDataFrame,
    reaches: gpd.GeoDataFrame,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    *,
    cmap_name: str = "RdBu_r",
    percent: bool = False,
    log_scale: bool = False,
) -> dict[str, float | str]:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if log_scale:
        positive = finite[finite > 0.0]
        if not positive.size:
            raise ValueError("Log-scaled reach maps require at least one positive value.")
        lower = float(np.min(positive))
        upper = float(np.max(positive))
        if upper <= lower:
            upper = lower * 10.0
        norm = LogNorm(vmin=lower, vmax=upper)
        mapped_values = np.where(np.asarray(values, dtype=float) > 0.0, values, np.nan)
        display = {"scale": "log", "minimum": lower, "maximum": upper}
    else:
        if finite.size:
            limit = float(np.nanpercentile(np.abs(finite), 98.0))
        else:
            limit = 1.0
        minimum_limit = 1.0 if percent else 1.0e3
        limit = max(limit, minimum_limit)
        if percent:
            limit = min(limit, 100.0)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        mapped_values = np.asarray(values, dtype=float)
        display = {"scale": "linear", "absolute_limit": limit}
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("0.65")
    boundary.plot(ax=ax, facecolor="0.96", edgecolor="0.55", linewidth=0.8)
    reaches.plot(ax=ax, color="0.82", linewidth=5.0, zorder=2)
    mapped = reaches.assign(_value=mapped_values)
    mapped.plot(
        ax=ax,
        column="_value",
        cmap=cmap,
        norm=norm,
        linewidth=3.2,
        zorder=3,
        missing_kwds={"color": "0.65"},
    )
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = ax.figure.colorbar(scalar, ax=ax, fraction=0.04, pad=0.015)
    colorbar.set_label(
        colorbar_label + (" (logarithmic color scale)" if log_scale else "")
    )
    ax.set_title(title)
    ax.set_axis_off()
    return display


def save_reach_summary_map(
    daily: pd.DataFrame,
    reaches: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> dict[str, object]:
    """Save total-flow local and upstream-integrated reach summaries."""
    dry_season = daily[pd.to_datetime(daily["date"]).dt.month.isin([6, 7, 8, 9, 10])]
    summary = daily.groupby("reach_id", sort=True).agg(
        cumulative_local_depletion=("local_total_streamflow_depletion_m3d", "sum"),
        cumulative_routed_depletion=("routed_total_streamflow_depletion_m3d", "sum"),
        cumulative_routed_unimpaired=("routed_unimpaired_total_streamflow_m3d", "sum"),
    )
    dry_summary = dry_season.groupby("reach_id", sort=True).agg(
        dry_depletion=("routed_total_streamflow_depletion_m3d", "sum"),
        dry_unimpaired=("routed_unimpaired_total_streamflow_m3d", "sum"),
    )
    dry_fraction = np.full(len(dry_summary), np.nan, dtype=float)
    denominator = dry_summary["dry_unimpaired"].to_numpy()
    np.divide(
        100.0 * dry_summary["dry_depletion"].to_numpy(),
        denominator,
        out=dry_fraction,
        where=denominator > 1.0,
    )
    dry_summary["dry_fraction"] = dry_fraction
    full_fraction = np.full(len(summary), np.nan, dtype=float)
    full_denominator = summary["cumulative_routed_unimpaired"].to_numpy()
    np.divide(
        100.0 * summary["cumulative_routed_depletion"].to_numpy(),
        full_denominator,
        out=full_fraction,
        where=full_denominator > 1.0,
    )
    summary["full_fraction"] = full_fraction
    mapped = reaches.set_index("reach_id").join(summary).join(dry_summary)
    mapped = mapped.sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    limits = {}
    limits["cumulative_local_depletion_m3"] = _plot_network_values(
        axes[0, 0],
        boundary,
        mapped,
        mapped["cumulative_local_depletion"].to_numpy(),
        "Cumulative reach-local flow depletion",
        "m³ over simulation",
        cmap_name="viridis",
        log_scale=True,
    )
    limits["cumulative_routed_depletion_m3"] = _plot_network_values(
        axes[0, 1],
        boundary,
        mapped,
        mapped["cumulative_routed_depletion"].to_numpy(),
        "Cumulative upstream-integrated flow depletion",
        "m³ over simulation",
        cmap_name="viridis",
        log_scale=True,
    )
    limits["full_period_routed_fraction_pct"] = _plot_network_values(
        axes[1, 0],
        boundary,
        mapped,
        mapped["full_fraction"].to_numpy(),
        "Full-period upstream-integrated depletion fraction",
        "% of unimpaired integrated flow",
        percent=True,
    )
    limits["june_october_fraction_pct"] = _plot_network_values(
        axes[1, 1],
        boundary,
        mapped,
        mapped["dry_fraction"].to_numpy(),
        "June–October upstream-integrated depletion fraction",
        "% of unimpaired integrated flow",
        percent=True,
    )
    fig.suptitle(
        "Total flow depletion by reach",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return {"display_limits": limits}


def _line_segments(reaches: gpd.GeoDataFrame) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for geometry in reaches.geometry:
        if geometry.geom_type == "LineString":
            segments.append(np.asarray(geometry.coords, dtype=float))
        else:
            merged = max(geometry.geoms, key=lambda value: value.length)
            segments.append(np.asarray(merged.coords, dtype=float))
    return segments


def save_reach_fraction_video(
    rolling_daily: pd.DataFrame,
    reaches: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    water_year_value: int,
    output_path: Path,
    config: ReachVisualizationConfig = ReachVisualizationConfig(),
) -> dict[str, object]:
    """Animate upstream-integrated trailing flow-depletion fractions."""
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required to create reach MP4 animations.")
    selected = rolling_daily[rolling_daily["water_year"] == water_year_value].copy()
    if selected["date"].nunique() < 365:
        raise ValueError(f"Water year {water_year_value} is incomplete in reach_daily.parquet.")
    selected = selected.sort_values(["date", "reach_id"])
    all_fractions = selected["rolling_routed_depletion_fraction_pct"].to_numpy()
    finite = np.abs(all_fractions[np.isfinite(all_fractions)])
    color_limit = (
        max(min(float(np.nanpercentile(finite, 98.0)), 100.0), 5.0)
        if finite.size
        else 5.0
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.65")

    dates = pd.DatetimeIndex(selected["date"].drop_duplicates())
    frame_dates = dates[:: config.frame_step_days]
    if frame_dates[-1] != dates[-1]:
        frame_dates = frame_dates.append(pd.DatetimeIndex([dates[-1]]))
    indexed = selected.set_index(["date", "reach_id"])
    reach_ids = reaches["reach_id"].to_numpy()
    segments = _line_segments(reaches)

    outlet_ids = reaches.loc[reaches["downstream_reach_id"].isna(), "reach_id"]
    if len(outlet_ids) != 1:
        raise ValueError("Reach GeoPackage must contain exactly one outlet reach.")
    outlet_id = int(outlet_ids.iloc[0])
    basin_by_date = selected[selected["reach_id"] == outlet_id].set_index("date")

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    boundary.plot(ax=ax, facecolor="0.96", edgecolor="0.55", linewidth=0.9)
    background = LineCollection(segments, colors="0.82", linewidths=6.0, zorder=2)
    ax.add_collection(background)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=4.0,
        zorder=3,
    )
    ax.add_collection(collection)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label(
        f"Trailing {config.rolling_days}-day integrated flow depletion (%)"
    )
    bounds = boundary.total_bounds
    pad_x = 0.04 * (bounds[2] - bounds[0])
    pad_y = 0.04 * (bounds[3] - bounds[1])
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    title = ax.set_title("")
    note = fig.text(
        0.5,
        0.035,
        "Gray: fraction undefined where rolling unimpaired integrated flow is ≤ 1 m³.\n"
        f"Display colors capped at ±{color_limit:.1f}%; source values are not clipped.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.09)

    def draw_frame(date) -> tuple[object, ...]:
        frame = indexed.loc[date].reindex(reach_ids)
        collection.set_array(frame["rolling_routed_depletion_fraction_pct"].to_numpy())
        basin_value = float(
            basin_by_date.loc[date, "rolling_routed_depletion_fraction_pct"]
        )
        basin_text = "undefined" if not np.isfinite(basin_value) else f"{basin_value:.1f}%"
        title.set_text(
            f"Water year {water_year_value}: {pd.Timestamp(date):%Y-%m-%d}\n"
            f"Basin trailing-{config.rolling_days}-day depletion fraction: {basin_text}"
        )
        return collection, title, note

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=config.frames_per_second,
        codec="libx264",
        bitrate=1800,
        metadata={
            "title": f"Upstream-integrated flow depletion, water year {water_year_value}",
            "comment": "Each reach includes its own and all upstream total-flow contributions.",
        },
    )
    with writer.saving(fig, str(output_path), dpi=150):
        for date in frame_dates:
            draw_frame(date)
            writer.grab_frame()
    plt.close(fig)
    return {
        "water_year": water_year_value,
        "frame_count": len(frame_dates),
        "color_limit_pct": color_limit,
        "first_frame": frame_dates[0].strftime("%Y-%m-%d"),
        "last_frame": frame_dates[-1].strftime("%Y-%m-%d"),
    }


def create_reach_visualizations(
    output_dir: Path,
    boundary_path: Path,
    *,
    water_years: list[int] | None = None,
    config: ReachVisualizationConfig = ReachVisualizationConfig(),
) -> dict[str, object]:
    """Create the standard summary map and representative reach animations."""
    output_dir = Path(output_dir)
    daily_path = output_dir / "reach_daily.parquet"
    reaches_path = output_dir / "reaches.gpkg"
    if not daily_path.exists() or not reaches_path.exists():
        raise FileNotFoundError(
            "Reach visualization requires reach_daily.parquet and reaches.gpkg."
        )
    daily = pd.read_parquet(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    reaches = gpd.read_file(reaches_path, layer="reaches").sort_values("reach_id")
    boundary = gpd.read_file(boundary_path).to_crs(reaches.crs)

    rolling = add_rolling_depletion_fraction(daily, config)
    summary_path = output_dir / "reach_depletion_summary.png"
    summary_metadata = save_reach_summary_map(rolling, reaches, boundary, summary_path)

    basin_path_candidates = sorted(output_dir.glob("simulation_unimpaired_*.csv"))
    if not basin_path_candidates:
        raise FileNotFoundError("Could not find the basin unimpaired simulation CSV.")
    basin_daily = pd.read_csv(basin_path_candidates[-1], parse_dates=["date"])
    automatic_years, recharge_by_year = select_representative_water_years(basin_daily)
    selected_years = automatic_years if water_years is None else list(dict.fromkeys(water_years))

    video_metadata = []
    video_paths = []
    for pattern in (
        "reach_depletion_fraction_wy*.mp4",
        "reach_routed_depletion_fraction_wy*.mp4",
    ):
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()
    for year in selected_years:
        video_path = output_dir / f"reach_routed_depletion_fraction_wy{year}.mp4"
        details = save_reach_fraction_video(
            rolling,
            reaches,
            boundary,
            year,
            video_path,
            config,
        )
        details["recharge_m3"] = recharge_by_year.get(year)
        details["selection"] = (
            "automatic driest/wettest complete water year"
            if water_years is None
            else "configured"
        )
        video_metadata.append(details)
        video_paths.append(video_path)

    generated_paths = [summary_path, *video_paths]
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in generated_paths
    }
    metadata = {
        "schema_version": REACH_VISUALIZATION_SCHEMA_VERSION,
        "parameters": {
            "rolling_days": config.rolling_days,
            "frame_step_days": config.frame_step_days,
            "frames_per_second": config.frames_per_second,
            "minimum_unimpaired_volume_m3": config.minimum_unimpaired_volume_m3,
        },
        "fraction_definition": (
            "100 times trailing-window upstream-integrated total-flow depletion "
            "divided by trailing-window upstream-integrated unimpaired total flow. "
            "Each reach includes its own local contribution and all upstream reaches."
        ),
        "source_files": {
            daily_path.name: _file_sha256(daily_path),
            reaches_path.name: _file_sha256(reaches_path),
            Path(boundary_path).name: _file_sha256(Path(boundary_path)),
        },
        "summary_map": summary_metadata,
        "videos": video_metadata,
        "files": files,
    }
    manifest_path = output_dir / "reach_visualization_metadata.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
