from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import calendar

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from landlab import RasterModelGrid
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from tqdm import tqdm


SECONDS_PER_DAY = 86400.0


def _landlab_components():
    try:
        from landlab.components import FlowAccumulator, GroundwaterDupuitPercolator, SinkFillerBarnes
    except ImportError as exc:
        raise ImportError(
            "Could not import required Landlab components. In the current `lab` environment this may be "
            "caused by a statsmodels/scipy compatibility issue while importing `landlab.components`."
        ) from exc
    return FlowAccumulator, GroundwaterDupuitPercolator, SinkFillerBarnes


@dataclass(frozen=True)
class GroundwaterConfig:
    target_crs: str = "EPSG:26910"
    target_resolution: float = 50.0
    stream_area_threshold: float = 250000.0
    courant_coefficient: float = 0.5
    stream_drain_offset: float = 0.0


@dataclass(frozen=True)
class GroundwaterInputs:
    dem_path: Path
    basin_path: Path
    transmissivity_path: Path
    depth_to_bedrock_path: Path
    porosity_path: Path


@dataclass
class GroundwaterState:
    grid: RasterModelGrid
    dem_coarse: object
    dx: float
    dy: float
    origin_x: float
    origin_y: float
    stream_indices: np.ndarray
    active_nodes: np.ndarray
    cell_area: float
    basin_area_m2: float


def _landlab_node_values(values) -> np.ndarray:
    """Convert north-up raster rows to Landlab's lower-left node ordering."""
    return np.flipud(np.asarray(values)).ravel()


def _load_and_align(path: Path, match_da, method=Resampling.average):
    da = rioxarray.open_rasterio(path, masked=True).squeeze()
    return da.rio.reproject_match(match_da, resampling=method)


def build_groundwater_state(
    inputs: GroundwaterInputs,
    config: GroundwaterConfig = GroundwaterConfig(),
) -> GroundwaterState:
    """Build the Landlab grid and static aquifer fields used by all simulations."""
    FlowAccumulator, _, SinkFillerBarnes = _landlab_components()

    dem_raw = rioxarray.open_rasterio(inputs.dem_path, masked=True).squeeze()
    dem_utm = dem_raw.rio.reproject(config.target_crs)
    bounds = dem_utm.rio.bounds()
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    new_width = int(width_m / config.target_resolution)
    new_height = int(height_m / config.target_resolution)
    dem_coarse = dem_utm.rio.reproject(
        config.target_crs,
        shape=(new_height, new_width),
        resampling=Resampling.average,
    )

    dy = abs(dem_coarse.rio.resolution()[1])
    dx = abs(dem_coarse.rio.resolution()[0])
    grid = RasterModelGrid(dem_coarse.shape, xy_spacing=(dx, dy))
    origin_x, origin_y = dem_coarse.rio.bounds()[0], dem_coarse.rio.bounds()[1]

    z = grid.add_zeros("topographic__elevation", at="node")
    dem_vals = _landlab_node_values(dem_coarse.values)
    dem_vals[np.isnan(dem_vals)] = np.nanmin(dem_vals)
    z[:] = dem_vals

    basin_gdf = gpd.read_file(inputs.basin_path)
    if basin_gdf.crs != dem_coarse.rio.crs:
        basin_gdf = basin_gdf.to_crs(dem_coarse.rio.crs)
    is_inside_basin = geometry_mask(
        basin_gdf.geometry,
        out_shape=dem_coarse.shape,
        transform=dem_coarse.rio.transform(),
        invert=True,
    )
    basin_nodes = _landlab_node_values(is_inside_basin).astype(bool)

    # Identify channels on the conditioned DEM before imposing basin closures.
    # Stream nodes later become fixed-head boundaries that drain to the outlet.
    SinkFillerBarnes(grid, method="Steepest", fill_flat=False).run_one_step()
    FlowAccumulator(grid, flow_director="FlowDirectorD8").run_one_step()
    is_stream = grid.at_node["drainage_area"] >= config.stream_area_threshold
    stream_indices = np.where(is_stream & basin_nodes)[0]

    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    grid.status_at_node[~basin_nodes] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[stream_indices] = grid.BC_NODE_IS_FIXED_VALUE

    trans_aligned = _load_and_align(inputs.transmissivity_path, dem_coarse)
    dtb_aligned = _load_and_align(inputs.depth_to_bedrock_path, dem_coarse)
    poro_aligned = _load_and_align(inputs.porosity_path, dem_coarse, method=Resampling.nearest)

    depth_raw = np.nan_to_num(_landlab_node_values(dtb_aligned.values), nan=0.0)
    depth_safe = np.maximum(depth_raw, 0.1)
    transmissivity_sec = np.nan_to_num(_landlab_node_values(trans_aligned.values), nan=0.0) / SECONDS_PER_DAY
    k_nodes = np.minimum(transmissivity_sec / depth_safe, 10.0)
    grid.add_field(
        "hydraulic_conductivity",
        grid.map_mean_of_link_nodes_to_link(k_nodes),
        at="link",
        clobber=True,
    )

    base_elev = z - depth_safe
    drain_elev = z[stream_indices] - config.stream_drain_offset
    base_elev[stream_indices] = np.minimum(base_elev[stream_indices], drain_elev)
    grid.add_field("aquifer_base__elevation", base_elev, at="node", clobber=True)

    porosity = np.nan_to_num(_landlab_node_values(poro_aligned.values), nan=0.2)
    grid.add_field("aquifer__porosity", porosity, at="node", clobber=True)
    grid.add_zeros("recharge_rate", at="node", clobber=True)

    active_nodes = np.zeros(grid.number_of_nodes, dtype=bool)
    active_nodes[grid.core_nodes] = True
    cell_area = dx * dy
    basin_area_m2 = np.sum(grid.cell_area_at_node[grid.core_nodes])

    return GroundwaterState(
        grid=grid,
        dem_coarse=dem_coarse,
        dx=dx,
        dy=dy,
        origin_x=origin_x,
        origin_y=origin_y,
        stream_indices=stream_indices,
        active_nodes=active_nodes,
        cell_area=cell_area,
        basin_area_m2=basin_area_m2,
    )


def initialize_water_table(state: GroundwaterState, mode: str, config: GroundwaterConfig, heads_path: Path | None = None) -> None:
    grid = state.grid
    grid.add_zeros("water_table__elevation", at="node", clobber=True)
    drain_elev = grid.at_node["topographic__elevation"][state.stream_indices] - config.stream_drain_offset
    
    if mode == "base":
        grid.at_node["water_table__elevation"][:] = grid.at_node["aquifer_base__elevation"][:]
    elif mode == "empty":
        grid.at_node["water_table__elevation"][:] = grid.at_node["aquifer_base__elevation"][:] + 0.01
        grid.at_node["water_table__elevation"][state.stream_indices] = drain_elev
    elif mode == "heads":
        if heads_path is None:
            raise ValueError("heads_path is required when mode='heads'")
        grid.at_node["water_table__elevation"][:] = np.load(heads_path)
    else:
        raise ValueError(f"Unknown water table initialization mode: {mode}")
    grid.at_node["water_table__elevation"][state.stream_indices] = drain_elev


def _storage_volume(state: GroundwaterState) -> float:
    grid = state.grid
    thickness = np.maximum(
        grid.at_node["water_table__elevation"] - grid.at_node["aquifer_base__elevation"],
        0.0,
    )
    return float(
        np.sum(
            thickness[grid.core_nodes]
            * grid.at_node["aquifer__porosity"][grid.core_nodes]
            * grid.cell_area_at_node[grid.core_nodes]
        )
    )


def _recharge_volume_for_step(state: GroundwaterState, dt: float) -> float:
    grid = state.grid
    return float(
        np.sum(grid.at_node["recharge_rate"][grid.core_nodes] * grid.cell_area_at_node[grid.core_nodes]) * dt
    )


def _streamflow_to_outlet_m3d(gdp) -> float:
    """Treat flux into fixed stream boundary nodes as instantly routed outlet flow."""
    return float(gdp.calc_gw_flux_out() * SECONDS_PER_DAY)


def make_dupuit_component(state: GroundwaterState, config: GroundwaterConfig):
    _, GroundwaterDupuitPercolator, _ = _landlab_components()
    return GroundwaterDupuitPercolator(
        state.grid,
        hydraulic_conductivity="hydraulic_conductivity",
        recharge_rate="recharge_rate",
        porosity="aquifer__porosity",
        courant_coefficient=config.courant_coefficient,
    )


def spin_up_steady_state(
    state: GroundwaterState,
    output_heads: Path,
    config: GroundwaterConfig = GroundwaterConfig(),
    *,
    recharge_mm_day: float = 20.0,
    max_years: int = 100,
    convergence_threshold: float = 0.01,
) -> pd.DataFrame:
    """Spin up from an almost-empty aquifer until discharge balances uniform recharge."""
    initialize_water_table(state, "empty", config)
    recharge_ms = (recharge_mm_day / 1000.0) / SECONDS_PER_DAY
    state.grid.at_node["recharge_rate"][:] = recharge_ms

    total_recharge_flux_m3s = _recharge_volume_for_step(state, 1.0)
    target_mm_day = (total_recharge_flux_m3s * SECONDS_PER_DAY / state.basin_area_m2) * 1000.0

    gdp = make_dupuit_component(state, config)
    previous_vol = _storage_volume(state)
    records = []

    for day in tqdm(range(1, max_years * 365 + 1), desc="Spin-up"):
        gdp.run_with_adaptive_time_step_solver(SECONDS_PER_DAY)
        current_vol = _storage_volume(state)
        recharge_vol_in = total_recharge_flux_m3s * SECONDS_PER_DAY
        mass_balance_discharge_vol = recharge_vol_in - (current_vol - previous_vol)
        outlet_streamflow_m3d = _streamflow_to_outlet_m3d(gdp)
        discharge_mm_day = (outlet_streamflow_m3d / state.basin_area_m2) * 1000.0
        ratio = discharge_mm_day / target_mm_day if target_mm_day else np.nan
        records.append(
            {
                "day": day,
                "streamflow_m3d": outlet_streamflow_m3d,
                "discharge_mm_day": discharge_mm_day,
                "mass_balance_discharge_m3d": mass_balance_discharge_vol,
                "target_mm_day": target_mm_day,
                "substeps": gdp.number_of_substeps,
            }
        )
        previous_vol = current_vol
        if abs(1.0 - ratio) < convergence_threshold:
            break

    output_heads = Path(output_heads)
    output_heads.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_heads, state.grid.at_node["water_table__elevation"])
    return pd.DataFrame(records)


def _nearest_active_neighbor(grid: RasterModelGrid, node_id: int) -> int:
    neighbors = np.concatenate(
        (grid.adjacent_nodes_at_node[node_id], grid.diagonal_adjacent_nodes_at_node[node_id])
    )
    for neighbor in neighbors[neighbors != -1]:
        if grid.status_at_node[neighbor] == grid.BC_NODE_IS_CORE:
            return int(neighbor)
    return int(node_id)


def build_monthly_pumping_maps(
    state: GroundwaterState,
    wells_path: Path,
    pumping_path: Path,
    basin_path: Path,
    *,
    target_crs: str = "EPSG:26910",
    apn_col: str = "APN",
    year_for_month_lengths: int = 2022,
) -> dict[int, np.ndarray]:
    """Map monthly APN pumping volumes onto Landlab nodes as negative flux fields."""
    basin_gdf = gpd.read_file(basin_path).to_crs(target_crs)
    wells_gdf = gpd.read_file(wells_path).to_crs(target_crs)
    wells_in_catchment = gpd.sjoin(wells_gdf, basin_gdf, how="inner", predicate="intersects")
    df_pump = pd.read_csv(pumping_path)

    well_x = wells_in_catchment.geometry.x.values
    well_y = wells_in_catchment.geometry.y.values
    cols = np.round((well_x - state.origin_x) / state.dx).astype(int)
    rows = np.round((well_y - state.origin_y) / state.dy).astype(int)
    valid_idx = (
        (cols >= 0)
        & (cols < state.grid.number_of_node_columns)
        & (rows >= 0)
        & (rows < state.grid.number_of_node_rows)
    )
    node_ids = rows[valid_idx] * state.grid.number_of_node_columns + cols[valid_idx]
    spatial_map = pd.DataFrame(
        {apn_col: wells_in_catchment[apn_col].values[valid_idx], "NodeID": node_ids}
    )

    non_core_mask = state.grid.status_at_node[spatial_map["NodeID"].values] != state.grid.BC_NODE_IS_CORE
    for idx in spatial_map.index[non_core_mask]:
        spatial_map.at[idx, "NodeID"] = _nearest_active_neighbor(state.grid, int(spatial_map.at[idx, "NodeID"]))

    maps: dict[int, np.ndarray] = {}
    valid_apns = wells_in_catchment[apn_col].unique()
    for month in range(1, 13):
        df_month = df_pump[(df_pump["Month"] == month) & (df_pump[apn_col].isin(valid_apns))].copy()
        if df_month.empty:
            maps[month] = np.zeros(state.grid.number_of_nodes)
            continue
        days = calendar.monthrange(year_for_month_lengths, month)[1]
        if "waterUse_m3Month" in df_month.columns:
            df_month["rate_m3d"] = df_month["waterUse_m3Month"] / days
        elif "waterUse_m3Day" in df_month.columns:
            df_month["rate_m3d"] = df_month["waterUse_m3Day"]
        else:
            raise ValueError("Pumping CSV must contain `waterUse_m3Month` or `waterUse_m3Day`.")
        climatology = df_month.groupby(apn_col)["rate_m3d"].mean().reset_index()
        merged = spatial_map.merge(climatology, on=apn_col, how="inner")
        vol_per_node_m3d = np.bincount(
            merged["NodeID"],
            weights=merged["rate_m3d"],
            minlength=state.grid.number_of_nodes,
        )
        maps[month] = -1.0 * (vol_per_node_m3d / SECONDS_PER_DAY) / state.cell_area
    return maps


def load_recharge_data(recharge_csv: Path, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    df = pd.read_csv(recharge_csv)
    df["date"] = pd.to_datetime(df["date"])
    if "Recharge" not in df.columns:
        raise ValueError("Recharge CSV must contain a `Recharge` column in mm/day.")
    
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].sort_values("date")
    # Resample to fill any missing dates
    df.set_index("date", inplace=True)
    df = df.resample("D").ffill().reset_index()
    
    return df


def run_scenarios(
    state: GroundwaterState,
    recharge_csv: Path,
    pumping_maps: dict[int, np.ndarray] | None = None,
    *,
    start_date: str,
    end_date: str,
    config: GroundwaterConfig = GroundwaterConfig(),
    initial_condition: str = "base",
    heads_path: Path | None = None,
    progress_interval: int = 30,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray]]]:
    """Run daily natural and pumped simulations."""
    recharge = load_recharge_data(recharge_csv, start_date, end_date)
    results: dict[str, pd.DataFrame] = {}
    snapshots: dict[str, dict[str, np.ndarray]] = {"Unimpaired (Natural)": {}}
    scenarios_to_run = ["Unimpaired (Natural)"]
    if pumping_maps is not None:
        snapshots["With Pumping"] = {}
        scenarios_to_run.append("With Pumping")

    for scenario in scenarios_to_run:
        print(f"Starting scenario: {scenario}")
        initialize_water_table(state, "heads" if initial_condition == "heads" else "base", config, heads_path)
        gdp = make_dupuit_component(state, config)
        previous_vol = _storage_volume(state)
        records = []
        days_run = 0

        for row in tqdm(recharge.itertuples(index=False), total=len(recharge), desc=scenario):
            days_run += 1
            recharge_ms = (float(row.Recharge) / 1000.0) / SECONDS_PER_DAY
            if scenario == "With Pumping" and pumping_maps is not None:
                pump_field = pumping_maps[row.date.month]
            else:
                pump_field = 0.0
            state.grid.at_node["recharge_rate"][:] = recharge_ms + pump_field
            gdp.run_with_adaptive_time_step_solver(SECONDS_PER_DAY)

            total_input_vol = _recharge_volume_for_step(state, SECONDS_PER_DAY)
            current_vol = _storage_volume(state)
            mass_balance_discharge_m3d = total_input_vol - (current_vol - previous_vol)
            records.append(
                {
                    "date": row.date,
                    "Q_m3d": mass_balance_discharge_m3d,
                    "storage_m3": current_vol,
                    "substeps": gdp.number_of_substeps,
                }
            )
            if row.date.month in (4, 9) and row.date.is_month_end:
                snapshots[scenario][row.date.strftime("%Y-%m-%d")] = state.grid.at_node["water_table__elevation"].copy()
            previous_vol = current_vol

        results[scenario] = pd.DataFrame(records)

    return results, snapshots


def save_outputs(
    results: dict[str, pd.DataFrame],
    snapshots: dict[str, dict[str, np.ndarray]],
    pumping_maps: dict[int, np.ndarray] | None,
    state: GroundwaterState,
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    recharge_csv: Path,
) -> None:
    """Save simulation result tables and diagnostic figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = f"{start_date}_to_{end_date}"

    natural = results["Unimpaired (Natural)"].copy()
    
    # Clip streamflow to >= 0
    natural["Q_m3d"] = np.maximum(natural["Q_m3d"], 0.0)
    natural.to_csv(output_dir / f"simulation_unimpaired_{date_str}.csv", index=False)

    if "With Pumping" not in results or pumping_maps is None:
        recharge = load_recharge_data(recharge_csv, start_date, end_date)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(natural["date"], natural["Q_m3d"], color="green", label="Unimpaired")
        axes[0].set_ylabel("Basin Discharge ($m^3/day$)")
        axes[0].set_title(f"Hydrograph ({start_date} to {end_date})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].bar(recharge["date"], recharge["Recharge"], color="blue", alpha=0.5, width=1.5)
        axes[1].set_ylabel("Recharge (mm/day)")
        axes[1].set_title("Daily Recharge Forcing")
        axes[1].invert_yaxis()
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"hydrographs_{date_str}.png", dpi=300)
        plt.close(fig)
    else:
        pumped = results["With Pumping"].copy()
        
        # Clip streamflow to >= 0
        pumped["Q_m3d"] = np.maximum(pumped["Q_m3d"], 0.0)
        pumped.to_csv(output_dir / f"simulation_with_pumping_{date_str}.csv", index=False)

        recharge = load_recharge_data(recharge_csv, start_date, end_date)
        depletion_m3d = natural["Q_m3d"].values - pumped["Q_m3d"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            depletion_fraction = (depletion_m3d / natural["Q_m3d"].values) * 100.0
        depletion_fraction[~np.isfinite(depletion_fraction)] = 0.0

        depletion = pd.DataFrame(
            {
                "date": natural["date"],
                "unimpaired_Q_m3d": natural["Q_m3d"].values,
                "pumped_Q_m3d": pumped["Q_m3d"].values,
                "depletion_m3d": depletion_m3d,
                "depletion_fraction_pct": depletion_fraction,
                "cumulative_depletion_m3": np.cumsum(depletion_m3d),
            }
        )
        depletion.to_csv(output_dir / "streamflow_depletion_timeseries.csv", index=False)
        monthly_depl = depletion.groupby(depletion["date"].dt.to_period("M"))["depletion_m3d"].mean()

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(natural["date"], natural["Q_m3d"], color="green", label="Unimpaired")
        axes[0].plot(pumped["date"], pumped["Q_m3d"], color="red", ls="--", label="With Pumping")
        axes[0].set_ylabel("Basin Discharge ($m^3/day$)")
        axes[0].set_title(f"Hydrographs ({start_date} to {end_date})")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].bar(recharge["date"], recharge["Recharge"], color="blue", alpha=0.5, width=1.5)
        axes[1].set_ylabel("Recharge (mm/day)")
        axes[1].set_title("Daily Recharge Forcing")
        axes[1].invert_yaxis()
        axes[1].grid(alpha=0.3)
        axes[2].bar(monthly_depl.index.to_timestamp(), monthly_depl.values, width=20, color="darkred", alpha=0.7)
        axes[2].set_ylabel("Depletion ($m^3/day$)")
        axes[2].set_title("Monthly Streamflow Depletion")
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"hydrographs_{date_str}.png", dpi=300)
        plt.close(fig)

        daily_pumping_m3 = []
        for current_date in natural["date"]:
            flux_map = pumping_maps[current_date.month]
            daily_pumping_m3.append(
                np.abs(np.sum(flux_map[state.grid.core_nodes] * state.grid.cell_area_at_node[state.grid.core_nodes]))
                * SECONDS_PER_DAY
            )
        daily_pumping_m3 = np.asarray(daily_pumping_m3)
        cumulative_pumping_m3 = np.cumsum(daily_pumping_m3)
        cumulative_depletion_m3 = np.cumsum(depletion_m3d)

        fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
        axes[0].plot(natural["date"], depletion_fraction, color="purple", lw=2)
        axes[0].fill_between(natural["date"], 0, depletion_fraction, color="purple", alpha=0.1)
        axes[0].set_ylabel("Depletion Fraction (%)")
        axes[0].set_title(f"Streamflow Capture Fraction ({start_date} to {end_date})")
        axes[0].grid(alpha=0.3)
        axes[1].bar(monthly_depl.index.to_timestamp() + pd.Timedelta(days=15), monthly_depl.values, width=25, color="darkred", alpha=0.7)
        axes[1].set_ylabel("Avg Depletion ($m^3$/day)")
        axes[1].set_title("Monthly Volumetric Streamflow Loss")
        axes[1].grid(alpha=0.3)
        axes[2].plot(natural["date"], cumulative_pumping_m3, color="black", linestyle=":", lw=3, label="Total Pumped")
        axes[2].plot(natural["date"], cumulative_depletion_m3, color="red", lw=2, label="Total Stream Loss")
        axes[2].fill_between(natural["date"], cumulative_depletion_m3, cumulative_pumping_m3, color="blue", alpha=0.15, label="From Aquifer Storage")
        axes[2].fill_between(natural["date"], 0, cumulative_depletion_m3, color="red", alpha=0.15, label="From Streamflow")
        axes[2].set_ylabel("Cumulative Volume ($m^3$)")
        axes[2].set_title("Source of Pumped Water")
        axes[2].legend(loc="upper left")
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"capture_{date_str}.png", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(depletion["date"], depletion["depletion_m3d"], color="darkred", lw=2)
        axes[0].set_ylabel("Depletion ($m^3/day$)")
        axes[0].set_title(f"Daily Streamflow Depletion ({start_date} to {end_date})")
        axes[0].grid(alpha=0.3)
        axes[1].plot(depletion["date"], depletion["cumulative_depletion_m3"], color="red", lw=2)
        axes[1].set_ylabel("Cumulative Depletion ($m^3$)")
        axes[1].set_xlabel("Date")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"depletion_timeseries_{date_str}.png", dpi=300)
        plt.close(fig)

    for scenario, dates_dict in snapshots.items():
        scenario_slug = "unimpaired" if "Unimpaired" in scenario else "pumped"
        for dt_str, wt_elev in dates_dict.items():
            np.save(output_dir / f"wt_{scenario_slug}_{dt_str}.npy", wt_elev)


def save_setup_plots(state: GroundwaterState, config: GroundwaterConfig, output_dir: Path) -> None:
    """Save plots of the initial grid, topography, and boundaries."""
    try:
        from landlab.plot import imshow_grid
    except ImportError:
        print("Warning: Could not import landlab.plot.imshow_grid. Skipping setup plots.")
        return
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = state.grid
    width_m = grid.shape[1] * state.dx
    height_m = grid.shape[0] * state.dy

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Topography & Stream Network
    plt.sca(ax[0])
    imshow_grid(grid, "topographic__elevation", plot_name="Topography & Streams", cmap="terrain")
    y_locs = grid.y_of_node[state.stream_indices]
    x_locs = grid.x_of_node[state.stream_indices]
    ax[0].scatter(x_locs, y_locs, s=1, c="blue", alpha=0.5, label="Streams")
    ax[0].legend()

    # Plot 2: Boundary Conditions
    plt.sca(ax[1])
    status_grid = grid.status_at_node.reshape(grid.shape)
    cmap_bc = plt.get_cmap("viridis", 5)
    im2 = ax[1].imshow(status_grid, cmap=cmap_bc, origin="lower", extent=[0, width_m, 0, height_m])
    ax[1].set_title("Boundary Conditions")
    legend_elements = [
        Patch(facecolor=cmap_bc(0.0), label="Core (Active)"),
        Patch(facecolor=cmap_bc(0.25), label="Fixed (Stream)"),
        Patch(facecolor=cmap_bc(1.0), label="Closed (Inactive)"),
    ]
    ax[1].legend(handles=legend_elements, loc="lower right")

    # Plot 3: Initial Saturated Thickness (Empty Hills)
    plt.sca(ax[2])
    drain_elev = grid.at_node["topographic__elevation"][state.stream_indices] - config.stream_drain_offset
    wt = grid.at_node["aquifer_base__elevation"].copy() + 0.01
    wt[state.stream_indices] = drain_elev
    thickness = wt - grid.at_node["aquifer_base__elevation"]
    thickness[grid.status_at_node == grid.BC_NODE_IS_CLOSED] = np.nan
    imshow_grid(grid, thickness, plot_name="Initial Saturated Thickness (m)", cmap="Blues")
    ax[2].set_title("Initial Thickness (Streams Full, Hills Empty)")

    plt.tight_layout()
    fig.savefig(output_dir / "grid_setup_plots.png", dpi=300)
    plt.close(fig)
