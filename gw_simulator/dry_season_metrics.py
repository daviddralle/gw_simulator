from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DRY_SEASON_METRICS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class DrySeasonMetricConfig:
    months: tuple[int, ...] = (6, 7, 8, 9, 10)
    rolling_windows_days: tuple[int, ...] = (7, 30)
    near_zero_flow_m3d: float = 1.0
    minimum_fraction_denominator_m3: float = 1.0

    def __post_init__(self) -> None:
        if not self.months or any(month < 1 or month > 12 for month in self.months):
            raise ValueError("Dry-season months must be integers from 1 through 12.")
        if len(set(self.months)) != len(self.months):
            raise ValueError("Dry-season months must be unique.")
        if not self.rolling_windows_days or any(
            window <= 0 for window in self.rolling_windows_days
        ):
            raise ValueError("Rolling windows must contain positive day counts.")
        if self.near_zero_flow_m3d < 0.0:
            raise ValueError("Near-zero flow threshold cannot be negative.")
        if self.minimum_fraction_denominator_m3 <= 0.0:
            raise ValueError("Minimum fraction denominator must be positive.")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {', '.join(missing)}")


def _prepare_dates(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    if output["date"].isna().any():
        raise ValueError(f"{label} contains invalid dates.")
    return output


def _expected_dates(year: int, months: tuple[int, ...]) -> pd.DatetimeIndex:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return dates[dates.month.isin(months)]


def _season_frame(
    frame: pd.DataFrame,
    year: int,
    config: DrySeasonMetricConfig,
) -> tuple[pd.DataFrame, int, bool]:
    expected = _expected_dates(year, config.months)
    selected = frame[frame["date"].dt.year == year].copy()
    if selected["date"].duplicated().any():
        raise ValueError(f"Dry-season input contains duplicate dates in {year}.")
    selected = selected.set_index("date").sort_index()
    observed = int(selected.index.isin(expected).sum())
    selected = selected.reindex(expected)
    complete = observed == len(expected) and not selected.isna().all(axis=1).any()
    return selected, observed, complete


def _integrated_fraction(
    numerator: float,
    denominator: float,
    config: DrySeasonMetricConfig,
) -> float:
    if denominator <= config.minimum_fraction_denominator_m3:
        return np.nan
    return 100.0 * numerator / denominator


def _flow_statistics(
    series: pd.Series,
    prefix: str,
    config: DrySeasonMetricConfig,
) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    metrics = {
        f"{prefix}_mean_m3d": float(finite.mean()),
        f"{prefix}_median_m3d": float(finite.median()),
        f"{prefix}_minimum_m3d": float(finite.min()),
        f"{prefix}_p05_m3d": float(finite.quantile(0.05)),
    }
    for window in config.rolling_windows_days:
        rolling = values.rolling(window, min_periods=window).mean()
        metrics[f"{prefix}_minimum_{window}day_mean_m3d"] = float(rolling.min())
    return metrics


def extract_basin_dry_season_metrics(
    depletion: pd.DataFrame,
    config: DrySeasonMetricConfig = DrySeasonMetricConfig(),
) -> pd.DataFrame:
    required = {
        "date",
        "unimpaired_total_streamflow_m3d",
        "pumped_total_streamflow_m3d",
        "total_streamflow_depletion_m3d",
        "scheduled_pumping_m3d",
        "allocated_pumping_m3d",
        "modeled_extraction_m3d",
        "source_capacity_shortfall_m3d",
        "daily_aquifer_storage_depletion_m3",
    }
    _require_columns(depletion, required, "Basin depletion table")
    frame = _prepare_dates(depletion, "Basin depletion table")
    frame = frame[frame["date"].dt.month.isin(config.months)]
    records: list[dict[str, object]] = []
    for year in sorted(frame["date"].dt.year.unique()):
        season, observed_days, complete = _season_frame(frame, int(year), config)
        natural = season["unimpaired_total_streamflow_m3d"]
        pumped = season["pumped_total_streamflow_m3d"]
        depletion_rate = season["total_streamflow_depletion_m3d"]
        natural_volume = float(natural.sum())
        pumped_volume = float(pumped.sum())
        depletion_volume = float(depletion_rate.sum())
        defined = natural > config.near_zero_flow_m3d
        record: dict[str, object] = {
            "dry_season_year": int(year),
            "start_date": _expected_dates(int(year), config.months).min(),
            "end_date": _expected_dates(int(year), config.months).max(),
            "observed_days": observed_days,
            "expected_days": len(_expected_dates(int(year), config.months)),
            "complete_season": complete,
            "unimpaired_streamflow_volume_m3": natural_volume,
            "pumped_streamflow_volume_m3": pumped_volume,
            "streamflow_depletion_volume_m3": depletion_volume,
            "integrated_impaired_streamflow_fraction_pct": _integrated_fraction(
                pumped_volume, natural_volume, config
            ),
            "integrated_streamflow_depletion_fraction_pct": _integrated_fraction(
                depletion_volume, natural_volume, config
            ),
            "streamflow_depletion_mean_m3d": float(depletion_rate.mean()),
            "streamflow_depletion_median_m3d": float(depletion_rate.median()),
            "streamflow_depletion_maximum_m3d": float(depletion_rate.max()),
            "exact_zero_pumped_flow_days": int((pumped == 0.0).sum()),
            "near_zero_pumped_flow_days": int(
                (pumped <= config.near_zero_flow_m3d).sum()
            ),
            "pumping_created_near_zero_flow_days": int(
                (defined & (pumped <= config.near_zero_flow_m3d)).sum()
            ),
            "fraction_defined_days": int(defined.sum()),
            "days_with_at_least_50pct_depletion": int(
                (defined & (pumped <= 0.5 * natural)).sum()
            ),
            "days_with_at_least_90pct_depletion": int(
                (defined & (pumped <= 0.1 * natural)).sum()
            ),
            "scheduled_pumping_volume_m3": float(
                season["scheduled_pumping_m3d"].sum()
            ),
            "allocated_pumping_volume_m3": float(
                season["allocated_pumping_m3d"].sum()
            ),
            "modeled_extraction_volume_m3": float(
                season["modeled_extraction_m3d"].sum()
            ),
            "source_capacity_shortfall_volume_m3": float(
                season["source_capacity_shortfall_m3d"].sum()
            ),
            "aquifer_storage_depletion_change_m3": float(
                season["daily_aquifer_storage_depletion_m3"].sum()
            ),
        }
        record.update(_flow_statistics(natural, "unimpaired_streamflow", config))
        record.update(_flow_statistics(pumped, "pumped_streamflow", config))
        records.append(record)

    metrics = pd.DataFrame.from_records(records).sort_values("dry_season_year")
    metrics["depletion_fraction_rank_complete_seasons"] = pd.Series(
        pd.NA, index=metrics.index, dtype="Int64"
    )
    eligible = metrics[metrics["complete_season"]]
    metrics.loc[
        eligible.index, "depletion_fraction_rank_complete_seasons"
    ] = eligible["integrated_streamflow_depletion_fraction_pct"].rank(
        ascending=False, method="min"
    ).astype("Int64")
    return metrics.reset_index(drop=True)


def extract_reach_dry_season_metrics(
    reach_daily: pd.DataFrame,
    reach_attributes: pd.DataFrame | None = None,
    config: DrySeasonMetricConfig = DrySeasonMetricConfig(),
) -> pd.DataFrame:
    required = {
        "date",
        "reach_id",
        "unimpaired_local_total_streamflow_m3d",
        "local_total_streamflow_depletion_m3d",
        "routed_unimpaired_total_streamflow_m3d",
        "routed_pumped_total_streamflow_m3d",
        "routed_total_streamflow_depletion_m3d",
    }
    _require_columns(reach_daily, required, "Reach daily table")
    frame = _prepare_dates(reach_daily, "Reach daily table")
    frame = frame[frame["date"].dt.month.isin(config.months)]
    records: list[dict[str, object]] = []
    for (reach_id, year), group in frame.groupby(
        ["reach_id", frame["date"].dt.year], sort=True
    ):
        season, observed_days, complete = _season_frame(group, int(year), config)
        natural = season["routed_unimpaired_total_streamflow_m3d"]
        pumped = season["routed_pumped_total_streamflow_m3d"]
        depletion_rate = season["routed_total_streamflow_depletion_m3d"]
        natural_volume = float(natural.sum())
        pumped_volume = float(pumped.sum())
        depletion_volume = float(depletion_rate.sum())
        local_natural_volume = float(
            season["unimpaired_local_total_streamflow_m3d"].sum()
        )
        local_depletion_volume = float(
            season["local_total_streamflow_depletion_m3d"].sum()
        )
        defined = natural > config.near_zero_flow_m3d
        record: dict[str, object] = {
            "reach_id": int(reach_id),
            "dry_season_year": int(year),
            "observed_days": observed_days,
            "expected_days": len(_expected_dates(int(year), config.months)),
            "complete_season": complete,
            "routed_unimpaired_streamflow_volume_m3": natural_volume,
            "routed_pumped_streamflow_volume_m3": pumped_volume,
            "routed_streamflow_depletion_volume_m3": depletion_volume,
            "integrated_routed_impaired_streamflow_fraction_pct": (
                _integrated_fraction(pumped_volume, natural_volume, config)
            ),
            "integrated_routed_streamflow_depletion_fraction_pct": (
                _integrated_fraction(depletion_volume, natural_volume, config)
            ),
            "local_unimpaired_streamflow_volume_m3": local_natural_volume,
            "local_streamflow_depletion_volume_m3": local_depletion_volume,
            "integrated_local_streamflow_depletion_fraction_pct": (
                _integrated_fraction(
                    local_depletion_volume, local_natural_volume, config
                )
            ),
            "routed_streamflow_depletion_mean_m3d": float(depletion_rate.mean()),
            "routed_streamflow_depletion_maximum_m3d": float(depletion_rate.max()),
            "routed_natural_zero_flow_days": int((natural == 0.0).sum()),
            "routed_pumped_zero_flow_days": int((pumped == 0.0).sum()),
            "routed_pumped_near_zero_flow_days": int(
                (pumped <= config.near_zero_flow_m3d).sum()
            ),
            "pumping_created_near_zero_flow_days": int(
                (defined & (pumped <= config.near_zero_flow_m3d)).sum()
            ),
            "fraction_defined_days": int(defined.sum()),
            "days_with_at_least_50pct_depletion": int(
                (defined & (pumped <= 0.5 * natural)).sum()
            ),
            "days_with_at_least_90pct_depletion": int(
                (defined & (pumped <= 0.1 * natural)).sum()
            ),
        }
        record.update(_flow_statistics(natural, "routed_unimpaired_streamflow", config))
        record.update(_flow_statistics(pumped, "routed_pumped_streamflow", config))
        records.append(record)

    metrics = pd.DataFrame.from_records(records)
    complete = metrics[metrics["complete_season"]]
    metrics["depletion_volume_rank_within_season"] = pd.Series(
        pd.NA, index=metrics.index, dtype="Int64"
    )
    metrics["depletion_fraction_rank_within_season"] = pd.Series(
        pd.NA, index=metrics.index, dtype="Int64"
    )
    for column, rank_column in (
        ("routed_streamflow_depletion_volume_m3", "depletion_volume_rank_within_season"),
        (
            "integrated_routed_streamflow_depletion_fraction_pct",
            "depletion_fraction_rank_within_season",
        ),
    ):
        ranks = complete.groupby("dry_season_year")[column].rank(
            ascending=False, method="min"
        )
        metrics.loc[complete.index, rank_column] = ranks.astype("Int64")

    if reach_attributes is not None:
        attributes = reach_attributes.copy()
        _require_columns(attributes, {"reach_id"}, "Reach attributes")
        attributes = attributes.drop_duplicates("reach_id")
        metrics = metrics.merge(attributes, on="reach_id", how="left", validate="many_to_one")
    return metrics.sort_values(["dry_season_year", "reach_id"]).reset_index(drop=True)


def summarize_reach_dry_season_metrics(reach_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "reach_id",
        "dry_season_year",
        "complete_season",
        "routed_unimpaired_streamflow_volume_m3",
        "routed_pumped_streamflow_volume_m3",
        "routed_streamflow_depletion_volume_m3",
        "integrated_routed_streamflow_depletion_fraction_pct",
        "routed_pumped_near_zero_flow_days",
        "pumping_created_near_zero_flow_days",
        "days_with_at_least_50pct_depletion",
        "days_with_at_least_90pct_depletion",
        "routed_pumped_streamflow_minimum_7day_mean_m3d",
        "routed_pumped_streamflow_minimum_30day_mean_m3d",
    }
    _require_columns(reach_metrics, required, "Reach dry-season metrics")
    complete = reach_metrics[reach_metrics["complete_season"]].copy()
    if complete.empty:
        raise ValueError("No complete dry seasons are available for reach summaries.")

    records: list[dict[str, object]] = []
    metric_columns = set(required)
    attribute_columns = [
        column
        for column in complete.columns
        if column not in metric_columns
        and column
        not in {
            "observed_days",
            "expected_days",
            "fraction_defined_days",
            "depletion_volume_rank_within_season",
            "depletion_fraction_rank_within_season",
            "routed_natural_zero_flow_days",
            "routed_pumped_zero_flow_days",
            "routed_streamflow_depletion_mean_m3d",
            "routed_streamflow_depletion_maximum_m3d",
            "local_unimpaired_streamflow_volume_m3",
            "local_streamflow_depletion_volume_m3",
            "integrated_local_streamflow_depletion_fraction_pct",
        }
        and not column.startswith("routed_unimpaired_streamflow_")
        and not column.startswith("routed_pumped_streamflow_")
        and column != "integrated_routed_impaired_streamflow_fraction_pct"
    ]
    for reach_id, group in complete.groupby("reach_id", sort=True):
        natural_volume = float(group["routed_unimpaired_streamflow_volume_m3"].sum())
        pumped_volume = float(group["routed_pumped_streamflow_volume_m3"].sum())
        depletion_volume = float(group["routed_streamflow_depletion_volume_m3"].sum())
        worst_index = group[
            "integrated_routed_streamflow_depletion_fraction_pct"
        ].idxmax()
        record: dict[str, object] = {
            "reach_id": int(reach_id),
            "complete_season_count": int(len(group)),
            "first_complete_dry_season_year": int(group["dry_season_year"].min()),
            "last_complete_dry_season_year": int(group["dry_season_year"].max()),
            "routed_unimpaired_streamflow_volume_m3": natural_volume,
            "routed_pumped_streamflow_volume_m3": pumped_volume,
            "routed_streamflow_depletion_volume_m3": depletion_volume,
            "integrated_routed_streamflow_depletion_fraction_pct": (
                100.0 * depletion_volume / natural_volume
                if natural_volume > 0.0
                else np.nan
            ),
            "median_annual_routed_depletion_fraction_pct": float(
                group["integrated_routed_streamflow_depletion_fraction_pct"].median()
            ),
            "maximum_annual_routed_depletion_fraction_pct": float(
                group["integrated_routed_streamflow_depletion_fraction_pct"].max()
            ),
            "worst_dry_season_year_by_fraction": int(
                reach_metrics.loc[worst_index, "dry_season_year"]
            ),
            "total_routed_pumped_near_zero_flow_days": int(
                group["routed_pumped_near_zero_flow_days"].sum()
            ),
            "total_pumping_created_near_zero_flow_days": int(
                group["pumping_created_near_zero_flow_days"].sum()
            ),
            "total_days_with_at_least_50pct_depletion": int(
                group["days_with_at_least_50pct_depletion"].sum()
            ),
            "total_days_with_at_least_90pct_depletion": int(
                group["days_with_at_least_90pct_depletion"].sum()
            ),
            "minimum_annual_7day_pumped_flow_m3d": float(
                group["routed_pumped_streamflow_minimum_7day_mean_m3d"].min()
            ),
            "minimum_annual_30day_pumped_flow_m3d": float(
                group["routed_pumped_streamflow_minimum_30day_mean_m3d"].min()
            ),
        }
        for column in attribute_columns:
            record[column] = group.iloc[0][column]
        records.append(record)

    summary = pd.DataFrame.from_records(records)
    summary["overall_depletion_volume_rank"] = summary[
        "routed_streamflow_depletion_volume_m3"
    ].rank(ascending=False, method="min").astype("Int64")
    summary["overall_depletion_fraction_rank"] = summary[
        "integrated_routed_streamflow_depletion_fraction_pct"
    ].rank(ascending=False, method="min").astype("Int64")
    return summary.sort_values("reach_id").reset_index(drop=True)


def write_dry_season_metrics(
    output_dir: Path,
    destination: Path,
    config: DrySeasonMetricConfig = DrySeasonMetricConfig(),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    destination = Path(destination)
    depletion_candidates = sorted(output_dir.glob("streamflow_depletion_*_to_*.csv"))
    if len(depletion_candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one dated streamflow depletion CSV; found "
            f"{len(depletion_candidates)}."
        )
    depletion_path = depletion_candidates[0]
    reach_daily_path = output_dir / "reach_daily.parquet"
    reaches_path = output_dir / "reaches.gpkg"
    for path in (reach_daily_path, reaches_path):
        if not path.exists():
            raise FileNotFoundError(path)

    depletion = pd.read_csv(depletion_path, parse_dates=["date"])
    reach_daily = pd.read_parquet(reach_daily_path)
    reach_attributes = pd.read_parquet(reaches_path) if reaches_path.suffix == ".parquet" else None
    if reach_attributes is None:
        import geopandas as gpd

        reaches = gpd.read_file(reaches_path, layer="reaches")
        keep = [
            "reach_id",
            "downstream_reach_id",
            "is_outlet_reach",
            "stream_node_count",
            "reach_length_m",
            "incremental_area_m2",
            "upstream_area_m2",
        ]
        reach_attributes = pd.DataFrame(reaches[keep])

    basin = extract_basin_dry_season_metrics(depletion, config)
    reach = extract_reach_dry_season_metrics(reach_daily, reach_attributes, config)
    reach_summary = summarize_reach_dry_season_metrics(reach)

    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "basin": destination / "basin_dry_season_metrics_by_year.csv",
        "reach_by_year": destination / "reach_dry_season_metrics_by_year.csv",
        "reach_summary": destination / "reach_dry_season_summary.csv",
    }
    basin.to_csv(artifacts["basin"], index=False)
    reach.to_csv(artifacts["reach_by_year"], index=False)
    reach_summary.to_csv(artifacts["reach_summary"], index=False)

    metadata: dict[str, object] = {
        "schema_version": DRY_SEASON_METRICS_SCHEMA_VERSION,
        "configuration": asdict(config),
        "definitions": {
            "dry_season_year": (
                "Calendar year containing the configured dry-season months; this avoids "
                "splitting June-October at the October 1 water-year boundary."
            ),
            "complete_season": (
                "True only when every expected calendar day in the configured months is present."
            ),
            "integrated_fraction": (
                "100 times the seasonal volume numerator divided by seasonal unimpaired "
                "streamflow volume; daily percentages are not averaged."
            ),
            "low_flow_windows": (
                "Minimum consecutive daily mean within the dry season. Missing days break "
                "a rolling window."
            ),
            "comparative_rankings": "Incomplete edge seasons are excluded.",
            "reach_summary": "Uses complete dry seasons only.",
        },
        "source_files": {
            depletion_path.name: _sha256(depletion_path),
            reach_daily_path.name: _sha256(reach_daily_path),
            reaches_path.name: _sha256(reaches_path),
        },
        "artifacts": {},
        "complete_dry_season_years": basin.loc[
            basin["complete_season"], "dry_season_year"
        ].astype(int).tolist(),
        "incomplete_dry_season_years": basin.loc[
            ~basin["complete_season"], "dry_season_year"
        ].astype(int).tolist(),
    }
    metadata["artifacts"] = {
        path.name: {
            "rows": int(
                {"basin": basin, "reach_by_year": reach, "reach_summary": reach_summary}[key].shape[0]
            ),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for key, path in artifacts.items()
    }
    metadata_path = destination / "dry_season_metrics_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
