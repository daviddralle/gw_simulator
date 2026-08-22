from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


GALLON_M3 = 0.003785411784
FOOT_M = 0.3048
SECONDS_PER_DAY = 86400.0

# A common screening conversion is T [gallons/day/foot] = 1,500--2,000
# times specific capacity [gallons/minute/foot].  These constants perform the
# unit conversion to square metres/day.  They are deliberately exposed as a
# range because short single-well tests do not justify a single precise factor.
SPECIFIC_CAPACITY_T_LOW = 1500.0 * GALLON_M3 / FOOT_M
SPECIFIC_CAPACITY_T_HIGH = 2000.0 * GALLON_M3 / FOOT_M


@dataclass(frozen=True)
class RecessionFilter:
    """Conservative defaults for isolating low-pumping, low-ET recessions."""

    allowed_months: tuple[int, ...] = (11, 12, 1, 2, 3, 4)
    antecedent_days: int = 3
    max_antecedent_precip_mm: float = 0.1
    max_antecedent_recharge_mm: float = 0.1
    max_et_mm_day: float = 1.5
    max_pumping_quantile: float = 0.35
    min_discharge_mm_day: float = 1.0e-6
    bin_count: int = 12
    min_bin_count: int = 10


def _quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {}
    quantiles = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": float(quantiles.loc[0.10]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.50]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.90]),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def summarize_well_context(
    *,
    information_csv: str | Path,
    specific_capacity_csv: str | Path,
    texture_csv: str | Path,
    boundary_path: str | Path | None = None,
    borehole_ids: Iterable[object] | None = None,
) -> dict[str, object]:
    """Summarize USGS Russian River boreholes intersecting a basin boundary.

    Depth fields in the source tables are feet.  Specific capacity is gallons
    per minute per foot of drawdown.  The source files use Windows-1252.
    """

    information_csv = Path(information_csv)
    specific_capacity_csv = Path(specific_capacity_csv)
    texture_csv = Path(texture_csv)
    information = pd.read_csv(information_csv, encoding="cp1252")
    capacity = pd.read_csv(specific_capacity_csv, encoding="cp1252")
    texture = pd.read_csv(texture_csv, encoding="cp1252")
    requested_ids = set(borehole_ids) if borehole_ids is not None else None
    if requested_ids is not None:
        local_information = information.loc[
            information["BoreID"].isin(requested_ids)
        ].copy()
        selection_source = "supplied borehole IDs"
    else:
        if boundary_path is None:
            raise ValueError("Provide either a boundary or borehole IDs.")
        boundary_path = Path(boundary_path)
        boundary = gpd.read_file(boundary_path)
        if boundary.empty or boundary.crs is None:
            raise ValueError("Boundary must contain geometry with a CRS.")
        boundary_geometry = boundary.to_crs("EPSG:4326").geometry.union_all()
        points = gpd.GeoDataFrame(
            information,
            geometry=gpd.points_from_xy(
                information["Longitude"], information["Latitude"]
            ),
            crs="EPSG:4326",
        )
        inside = points.geometry.intersects(boundary_geometry)
        local_information = information.loc[inside].copy()
        selection_source = str(boundary_path)
    local_ids = set(local_information["BoreID"].tolist())
    local_capacity = capacity.loc[capacity["BoreID"].isin(local_ids)].copy()
    local_texture = texture.loc[texture["BoreID"].isin(local_ids)].copy()

    specific_capacity = _numeric(local_capacity, "Specific Capacity")
    valid_capacity = local_capacity.loc[specific_capacity > 0.0].copy()
    specific_capacity = _numeric(valid_capacity, "Specific Capacity")
    t_low = specific_capacity * SPECIFIC_CAPACITY_T_LOW
    t_high = specific_capacity * SPECIFIC_CAPACITY_T_HIGH

    local_texture["texture_normalized"] = (
        local_texture["Texture"].astype(str).str.strip().str.lower()
    )
    consolidated = {
        "rock",
        "sandstone",
        "siltstone",
        "shale",
        "conglomerate",
        "lava",
    }
    first_consolidated_ft = (
        local_texture.loc[local_texture["texture_normalized"].isin(consolidated)]
        .groupby("BoreID")["Depth1"]
        .min()
    )
    jointed = local_texture["Tex_Q"].astype(str).str.strip().str.lower().eq("joint")
    first_joint_ft = local_texture.loc[jointed].groupby("BoreID")["Depth1"].min()

    duration = _numeric(valid_capacity, "Duration - hours")
    duration_counts = {
        str(float(key)): int(value)
        for key, value in duration.value_counts(dropna=False).sort_index().items()
        if np.isfinite(key)
    }
    total_depth_ft = _numeric(local_information, "Total_depth")
    top_perf_ft = _numeric(valid_capacity, "Top_Perf")
    bottom_perf_ft = _numeric(valid_capacity, "Bottom_Perf")
    static_water_ft = _numeric(valid_capacity, "WaterLevel1 - depth to water feet")
    drawdown_ft = _numeric(valid_capacity, "Drawdown - feet")
    rate_gpm = _numeric(valid_capacity, "Rate - gallons per minute")

    return {
        "source": {
            "information_csv": str(information_csv),
            "specific_capacity_csv": str(specific_capacity_csv),
            "texture_csv": str(texture_csv),
            "selection": selection_source,
            "source_borehole_count": int(information["BoreID"].nunique()),
            "source_specific_capacity_count": int(
                (_numeric(capacity, "Specific Capacity") > 0.0).sum()
            ),
        },
        "local": {
            "borehole_count": int(len(local_ids)),
            "total_depth_count": int(total_depth_ft.notna().sum()),
            "total_depth_ft": _quantiles(total_depth_ft),
            "total_depth_m": _quantiles(total_depth_ft * FOOT_M),
            "specific_capacity_test_count": int(len(valid_capacity)),
            "specific_capacity_gpm_per_ft": _quantiles(specific_capacity),
            "screening_transmissivity_low_m2_day": _quantiles(t_low),
            "screening_transmissivity_high_m2_day": _quantiles(t_high),
            "test_duration_hours": _quantiles(duration),
            "test_duration_counts": duration_counts,
            "pumping_rate_gpm": _quantiles(rate_gpm),
            "drawdown_ft": _quantiles(drawdown_ft),
            "static_water_depth_ft": _quantiles(static_water_ft),
            "top_perforation_ft": _quantiles(top_perf_ft),
            "bottom_perforation_ft": _quantiles(bottom_perf_ft),
            "first_consolidated_interval_count": int(len(first_consolidated_ft)),
            "first_consolidated_depth_ft": _quantiles(first_consolidated_ft),
            "first_consolidated_depth_m": _quantiles(first_consolidated_ft * FOOT_M),
            "first_jointed_interval_count": int(len(first_joint_ft)),
            "first_jointed_depth_ft": _quantiles(first_joint_ft),
            "first_jointed_depth_m": _quantiles(first_joint_ft * FOOT_M),
        },
        "specific_capacity_conversion": {
            "description": (
                "Screening range based on T[gpd/ft] = 1,500--2,000 times "
                "specific capacity[gpm/ft]; not a corrected aquifer-test result."
            ),
            "low_factor_m2_day_per_gpm_ft": SPECIFIC_CAPACITY_T_LOW,
            "high_factor_m2_day_per_gpm_ft": SPECIFIC_CAPACITY_T_HIGH,
        },
    }


def export_local_well_context(
    *,
    information_csv: str | Path,
    specific_capacity_csv: str | Path,
    texture_csv: str | Path,
    boundary_path: str | Path | None = None,
    borehole_ids: Iterable[object] | None = None,
    output_path: str | Path,
) -> dict[str, int]:
    """Export basin-intersecting source records as three GeoPackage layers."""

    information = pd.read_csv(information_csv, encoding="cp1252")
    capacity = pd.read_csv(specific_capacity_csv, encoding="cp1252")
    texture = pd.read_csv(texture_csv, encoding="cp1252")

    def to_points(frame: pd.DataFrame) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            frame,
            geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
            crs="EPSG:4326",
        )

    boreholes = to_points(information)
    if borehole_ids is not None:
        boreholes = boreholes.loc[boreholes["BoreID"].isin(set(borehole_ids))].copy()
    else:
        if boundary_path is None:
            raise ValueError("Provide either a boundary or borehole IDs.")
        boundary = gpd.read_file(boundary_path)
        if boundary.empty or boundary.crs is None:
            raise ValueError("Boundary must contain geometry with a CRS.")
        boundary_geometry = boundary.to_crs("EPSG:4326").geometry.union_all()
        boreholes = boreholes.loc[
            boreholes.geometry.intersects(boundary_geometry)
        ].copy()
    local_ids = set(boreholes["BoreID"])
    capacity_points = to_points(capacity.loc[capacity["BoreID"].isin(local_ids)].copy())
    texture_points = to_points(texture.loc[texture["BoreID"].isin(local_ids)].copy())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # GeoPackage ``mode='w'`` replaces one layer but may preserve other layers.
    # This file is a generated product, so remove it to make reruns idempotent.
    output_path.unlink(missing_ok=True)
    boreholes.to_file(output_path, layer="boreholes", driver="GPKG", mode="w")
    capacity_points.to_file(
        output_path, layer="specific_capacity", driver="GPKG", mode="a"
    )
    texture_points.to_file(output_path, layer="texture_intervals", driver="GPKG", mode="a")
    return {
        "boreholes": int(len(boreholes)),
        "specific_capacity_records": int(len(capacity_points)),
        "texture_intervals": int(len(texture_points)),
    }


def convert_streamflow_to_mm_day(
    values: pd.Series | np.ndarray,
    *,
    units: str,
    basin_area_m2: float,
) -> np.ndarray:
    """Convert a streamflow series to catchment-average millimetres/day."""

    flow = np.asarray(values, dtype=float)
    if basin_area_m2 <= 0.0:
        raise ValueError("Basin area must be positive.")
    normalized = units.lower().replace("/", "_")
    if normalized in {"mm_day", "mm_d"}:
        return flow
    if normalized in {"m3_day", "m3_d"}:
        volume_m3_day = flow
    elif normalized in {"m3_s", "cms"}:
        volume_m3_day = flow * SECONDS_PER_DAY
    elif normalized in {"cfs", "ft3_s"}:
        volume_m3_day = flow * (FOOT_M**3) * SECONDS_PER_DAY
    else:
        raise ValueError(f"Unsupported streamflow units: {units}")
    return volume_m3_day / basin_area_m2 * 1000.0


def aggregate_streamflow_to_daily(
    frame: pd.DataFrame,
    *,
    date_column: str,
    flow_column: str,
    min_daily_coverage: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Aggregate daily or subdaily discharge without modifying source records."""

    if not 0.0 < min_daily_coverage <= 1.0:
        raise ValueError("Minimum daily coverage must be in (0, 1].")
    if date_column not in frame or flow_column not in frame:
        raise ValueError("Streamflow frame lacks the requested date or flow column.")

    work = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "flow": pd.to_numeric(frame[flow_column], errors="coerce"),
        }
    ).dropna(subset=["date"])
    work = work.sort_values("date")
    input_negative_count = int((work["flow"] < 0.0).sum())
    input_zero_count = int((work["flow"] == 0.0).sum())
    work.loc[work["flow"] < 0.0, "flow"] = np.nan
    duplicate_timestamp_count = int(work["date"].duplicated().sum())
    work = work.groupby("date", as_index=False)["flow"].mean()
    if len(work) < 2:
        raise ValueError("At least two streamflow timestamps are required.")

    positive_minutes = (
        work["date"].diff().dt.total_seconds().div(60.0).dropna()
    )
    positive_minutes = positive_minutes.loc[positive_minutes > 0.0]
    if positive_minutes.empty:
        raise ValueError("Streamflow timestamps do not advance in time.")
    modal_interval_minutes = float(positive_minutes.mode().iloc[0])
    subdaily = modal_interval_minutes < 720.0 or work["date"].dt.normalize().duplicated().any()
    expected_observations = (
        max(1, int(round(1440.0 / modal_interval_minutes))) if subdaily else 1
    )
    minimum_observations = int(np.ceil(expected_observations * min_daily_coverage))

    work = work.set_index("date")
    daily = work.resample("D").agg(
        flow=("flow", "mean"),
        valid_observations=("flow", "count"),
    )
    complete = daily["valid_observations"] >= minimum_observations
    daily.loc[~complete, "flow"] = np.nan
    daily = daily.reset_index()
    summary = {
        "input_row_count": int(len(frame)),
        "parsed_timestamp_count": int(len(work)),
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "input_negative_flow_count_excluded": input_negative_count,
        "input_zero_flow_count": input_zero_count,
        "first_timestamp": work.index.min().isoformat(),
        "last_timestamp": work.index.max().isoformat(),
        "modal_sampling_interval_minutes": modal_interval_minutes,
        "expected_observations_per_day": expected_observations,
        "minimum_daily_coverage": min_daily_coverage,
        "minimum_valid_observations_per_day": minimum_observations,
        "days_with_any_valid_observation": int(
            (daily["valid_observations"] > 0).sum()
        ),
        "complete_daily_mean_count": int(daily["flow"].notna().sum()),
    }
    return daily, summary


def expand_monthly_pumping_schedule(
    schedule_csv: str | Path,
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate parcel-level monthly rates and expand them to daily totals."""

    schedule = pd.read_csv(
        schedule_csv,
        usecols=["Date", "waterUse_m3Day"],
        parse_dates=["Date"],
    )
    schedule["waterUse_m3Day"] = pd.to_numeric(
        schedule["waterUse_m3Day"], errors="coerce"
    )
    if schedule[["Date", "waterUse_m3Day"]].isna().any(axis=None):
        raise ValueError("Pumping schedule contains invalid dates or rates.")
    monthly = schedule.groupby("Date", as_index=True)["waterUse_m3Day"].sum().sort_index()
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    month_starts = dates.to_period("M").to_timestamp()
    daily = monthly.reindex(month_starts).to_numpy(dtype=float)
    return pd.DataFrame({"date": dates, "pumping_m3_day": daily})


def prepare_recession_pairs(
    streamflow: pd.DataFrame,
    *,
    forcing: pd.DataFrame,
    pumping: pd.DataFrame | None = None,
    config: RecessionFilter = RecessionFilter(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select consecutive daily pairs under conservative recession conditions.

    ``streamflow`` must contain ``date`` and ``q_mm_day``.  ``forcing`` must
    contain ``date``, ``P``, ``ET``, and ``Recharge`` in mm/day.  Slightly
    rising pairs are retained because excluding one sign of measurement noise
    biases binned mean recession rates upward.
    """

    if config.antecedent_days < 1:
        raise ValueError("Antecedent window must be at least one day.")
    required_flow = {"date", "q_mm_day"}
    required_forcing = {"date", "P", "ET", "Recharge"}
    if missing := required_flow.difference(streamflow.columns):
        raise ValueError(f"Streamflow is missing columns: {sorted(missing)}")
    if missing := required_forcing.difference(forcing.columns):
        raise ValueError(f"Forcing is missing columns: {sorted(missing)}")

    flow = streamflow.loc[:, ["date", "q_mm_day"]].copy()
    met = forcing.loc[:, ["date", "P", "ET", "Recharge"]].copy()
    flow["date"] = pd.to_datetime(flow["date"], errors="coerce")
    met["date"] = pd.to_datetime(met["date"], errors="coerce")
    for column in ("q_mm_day",):
        flow[column] = pd.to_numeric(flow[column], errors="coerce")
    for column in ("P", "ET", "Recharge"):
        met[column] = pd.to_numeric(met[column], errors="coerce")
    if flow["date"].duplicated().any() or met["date"].duplicated().any():
        raise ValueError("Streamflow and forcing dates must be unique.")

    merged = flow.merge(met, on="date", how="left", validate="one_to_one")
    if pumping is not None:
        pump = pumping.loc[:, ["date", "pumping_m3_day"]].copy()
        pump["date"] = pd.to_datetime(pump["date"], errors="coerce")
        pump["pumping_m3_day"] = pd.to_numeric(
            pump["pumping_m3_day"], errors="coerce"
        )
        merged = merged.merge(pump, on="date", how="left", validate="one_to_one")
    else:
        merged["pumping_m3_day"] = 0.0
    merged = merged.sort_values("date").reset_index(drop=True)

    merged["antecedent_precip_mm"] = merged["P"].rolling(
        config.antecedent_days, min_periods=config.antecedent_days
    ).sum()
    merged["antecedent_recharge_mm"] = merged["Recharge"].rolling(
        config.antecedent_days, min_periods=config.antecedent_days
    ).sum()
    merged["previous_date"] = merged["date"].shift(1)
    merged["previous_q_mm_day"] = merged["q_mm_day"].shift(1)
    merged["consecutive"] = (
        merged["date"] - merged["previous_date"] == pd.Timedelta(days=1)
    )
    merged["q_mid_mm_day"] = (
        merged["q_mm_day"] + merged["previous_q_mm_day"]
    ) / 2.0
    merged["minus_dqdt_mm_day2"] = (
        merged["previous_q_mm_day"] - merged["q_mm_day"]
    )

    pumping_threshold = float(
        merged["pumping_m3_day"].dropna().quantile(config.max_pumping_quantile)
    )
    eligible = (
        merged["date"].dt.month.isin(config.allowed_months)
        & merged["consecutive"]
        & (merged["antecedent_precip_mm"] <= config.max_antecedent_precip_mm)
        & (merged["antecedent_recharge_mm"] <= config.max_antecedent_recharge_mm)
        & (merged["ET"] <= config.max_et_mm_day)
        & (merged["pumping_m3_day"] <= pumping_threshold)
        & (merged["previous_q_mm_day"] > config.min_discharge_mm_day)
        & (merged["q_mm_day"] > config.min_discharge_mm_day)
        & (merged["q_mid_mm_day"] > config.min_discharge_mm_day)
    )
    selected = merged.loc[eligible].copy()
    selected["receding"] = selected["minus_dqdt_mm_day2"] > 0.0
    summary = {
        "filter": asdict(config),
        "input_day_count": int(len(merged)),
        "eligible_pair_count": int(len(selected)),
        "eligible_receding_pair_count": int(selected["receding"].sum()),
        "pumping_threshold_m3_day": pumping_threshold,
        "first_eligible_date": (
            selected["date"].min().strftime("%Y-%m-%d") if len(selected) else None
        ),
        "last_eligible_date": (
            selected["date"].max().strftime("%Y-%m-%d") if len(selected) else None
        ),
    }
    return selected, summary


def estimate_kirchner_signature(
    pairs: pd.DataFrame,
    *,
    bin_count: int = 12,
    min_bin_count: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Estimate Kirchner's g(Q), dynamic storage, and recession exponent."""

    required = {"q_mid_mm_day", "minus_dqdt_mm_day2"}
    if missing := required.difference(pairs.columns):
        raise ValueError(f"Recession pairs are missing columns: {sorted(missing)}")
    work = pairs.loc[:, sorted(required)].replace([np.inf, -np.inf], np.nan).dropna()
    work = work.loc[work["q_mid_mm_day"] > 0.0].copy()
    if len(work) < max(4 * min_bin_count, 20):
        raise ValueError("Too few eligible recession pairs for stable binning.")

    effective_bins = min(bin_count, max(4, len(work) // min_bin_count))
    work["bin"] = pd.qcut(
        np.log(work["q_mid_mm_day"]), q=effective_bins, duplicates="drop"
    )
    grouped = work.groupby("bin", observed=True)
    bins = grouped.agg(
        q_mm_day=("q_mid_mm_day", "mean"),
        mean_minus_dqdt_mm_day2=("minus_dqdt_mm_day2", "mean"),
        sd_minus_dqdt_mm_day2=("minus_dqdt_mm_day2", "std"),
        count=("minus_dqdt_mm_day2", "size"),
        receding_fraction=("minus_dqdt_mm_day2", lambda x: float((x > 0.0).mean())),
    ).reset_index(drop=True)
    bins["se_minus_dqdt_mm_day2"] = (
        bins["sd_minus_dqdt_mm_day2"] / np.sqrt(bins["count"])
    )
    bins = bins.loc[
        (bins["count"] >= min_bin_count)
        & (bins["mean_minus_dqdt_mm_day2"] > 0.0)
        & (bins["q_mm_day"] > 0.0)
    ].copy()
    if len(bins) < 4:
        raise ValueError("Fewer than four bins have positive mean recession rates.")

    bins["g_per_day"] = bins["mean_minus_dqdt_mm_day2"] / bins["q_mm_day"]
    bins["tau_days"] = 1.0 / bins["g_per_day"]
    relative_error = (
        bins["se_minus_dqdt_mm_day2"] / bins["mean_minus_dqdt_mm_day2"]
    ).clip(lower=0.05)
    weights = 1.0 / relative_error.to_numpy(dtype=float)
    log_q = np.log(bins["q_mm_day"].to_numpy(dtype=float))
    log_g = np.log(bins["g_per_day"].to_numpy(dtype=float))

    quadratic = np.polyfit(log_q, log_g, deg=2, w=weights)
    linear = np.polyfit(log_q, log_g, deg=1, w=weights)
    fitted_linear = np.polyval(linear, log_q)
    weighted_mean = np.average(log_g, weights=weights**2)
    ss_res = float(np.sum(weights**2 * (log_g - fitted_linear) ** 2))
    ss_tot = float(np.sum(weights**2 * (log_g - weighted_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    integration_q = np.exp(np.linspace(float(log_q.min()), float(log_q.max()), 2000))
    integration_g = np.exp(np.polyval(quadratic, np.log(integration_q)))
    dynamic_storage_mm = float(np.trapezoid(1.0 / integration_g, integration_q))

    low_log_q = float(np.quantile(log_q, 0.20))
    low_q = float(np.exp(low_log_q))
    low_g = float(np.exp(np.polyval(quadratic, low_log_q)))
    recession_index_days = float(np.log(10.0) / low_g)

    bins["fitted_g_per_day"] = np.exp(np.polyval(quadratic, log_q))
    bins["local_b"] = 1.0 + quadratic[1] + 2.0 * quadratic[0] * log_q
    summary = {
        "eligible_pair_count": int(len(work)),
        "retained_bin_count": int(len(bins)),
        "q_range_mm_day": [float(bins["q_mm_day"].min()), float(bins["q_mm_day"].max())],
        "dynamic_storage_over_fitted_q_range_mm": dynamic_storage_mm,
        "power_law_b": float(1.0 + linear[0]),
        "power_law_log_g_intercept": float(linear[1]),
        "power_law_r_squared": r_squared,
        "quadratic_log_g_coefficients": [float(value) for value in quadratic],
        "local_b_range": [float(bins["local_b"].min()), float(bins["local_b"].max())],
        "low_flow_reference_q_mm_day": low_q,
        "low_flow_g_per_day": low_g,
        "low_flow_log_cycle_recession_index_days": recession_index_days,
    }
    return bins, summary


def block_bootstrap_kirchner_signature(
    pairs: pd.DataFrame,
    *,
    replicates: int = 500,
    seed: int = 731,
    min_pairs_per_water_year: int = 10,
    bin_count: int = 12,
    min_bin_count: int = 10,
) -> dict[str, object]:
    """Estimate sampling spread by resampling whole water years."""

    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    if "date" not in pairs:
        raise ValueError("Recession pairs must include dates for block bootstrap.")
    work = pairs.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["water_year"] = work["date"].dt.year + (
        work["date"].dt.month >= 10
    ).astype(int)
    counts = work.groupby("water_year").size()
    years = counts.loc[counts >= min_pairs_per_water_year].index.to_numpy(dtype=int)
    if len(years) < 3:
        raise ValueError("Fewer than three water years have enough recession pairs.")

    random = np.random.default_rng(seed)
    estimates: list[list[float]] = []
    for _ in range(replicates):
        samples = []
        for occurrence, water_year in enumerate(
            random.choice(years, size=len(years), replace=True)
        ):
            sample = work.loc[
                work["water_year"].eq(water_year),
                ["q_mid_mm_day", "minus_dqdt_mm_day2"],
            ].copy()
            sample.index = np.arange(len(sample)) + occurrence * 1_000_000
            samples.append(sample)
        try:
            _, signature = estimate_kirchner_signature(
                pd.concat(samples),
                bin_count=bin_count,
                min_bin_count=min_bin_count,
            )
        except ValueError:
            continue
        estimates.append(
            [
                signature["power_law_b"],
                signature["low_flow_log_cycle_recession_index_days"],
                signature["dynamic_storage_over_fitted_q_range_mm"],
            ]
        )
    if len(estimates) < max(10, replicates // 2):
        raise ValueError("Too few successful water-year bootstrap fits.")

    values = np.asarray(estimates, dtype=float)

    def interval(column: int) -> dict[str, float]:
        quantiles = np.quantile(values[:, column], [0.05, 0.50, 0.95])
        return {
            "p05": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p95": float(quantiles[2]),
        }

    return {
        "method": "nonparametric resampling of complete water-year blocks",
        "requested_replicates": replicates,
        "successful_replicates": int(len(estimates)),
        "seed": seed,
        "minimum_pairs_per_water_year": min_pairs_per_water_year,
        "eligible_water_year_pair_counts": {
            str(year): int(counts.loc[year]) for year in years
        },
        "power_law_b": interval(0),
        "low_flow_log_cycle_recession_index_days": interval(1),
        "dynamic_storage_over_fitted_q_range_mm": interval(2),
        "caution": (
            "These intervals describe sensitivity to the sampled water years, not "
            "all rating-curve, geometry, pumping, ET, or model-structural uncertainty."
        ),
    }


def rorabaugh_diffusivity_m2_day(
    recession_index_days: float,
    flow_length_m: float | Iterable[float],
) -> np.ndarray:
    """Return T/Sy from the Rorabaugh late-recession approximation."""

    if recession_index_days <= 0.0:
        raise ValueError("Recession index must be positive.")
    length = np.asarray(flow_length_m, dtype=float)
    if np.any(~np.isfinite(length)) or np.any(length <= 0.0):
        raise ValueError("Flow lengths must be finite and positive.")
    return 0.933 * length**2 / recession_index_days


def derive_parameter_triplet(
    *,
    transmissivity_m2_day: float | Iterable[float],
    diffusivity_m2_day: float | Iterable[float],
    dynamic_storage_mm: float,
    storage_multiplier: float = 1.0,
) -> dict[str, np.ndarray]:
    """Combine T, T/Sy, and dynamic-storage estimates.

    The returned depth is the saturated-thickness change represented by the
    analyzed discharge range.  It is not, by itself, total depth to bedrock.
    """

    transmissivity = np.asarray(transmissivity_m2_day, dtype=float)
    diffusivity = np.asarray(diffusivity_m2_day, dtype=float)
    if np.any(transmissivity <= 0.0) or np.any(diffusivity <= 0.0):
        raise ValueError("Transmissivity and diffusivity must be positive.")
    if dynamic_storage_mm <= 0.0 or storage_multiplier <= 0.0:
        raise ValueError("Dynamic storage and its multiplier must be positive.")
    specific_yield = transmissivity / diffusivity
    storage_capacity_m = dynamic_storage_mm / 1000.0 * storage_multiplier
    effective_depth_m = storage_capacity_m / specific_yield
    return {
        "transmissivity_m2_day": transmissivity,
        "diffusivity_m2_day": diffusivity,
        "specific_yield": specific_yield,
        "storage_capacity_m": np.asarray(storage_capacity_m),
        "effective_depth_m": effective_depth_m,
    }
