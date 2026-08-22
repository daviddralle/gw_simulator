from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


PML_COLLECTION = "projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a"
PML_SCALE_FACTOR = 0.01
PML_SCALE_METERS = 500
PML_COMPOSITE_DAYS = 8
PRISM_COLLECTION = "OREGONSTATE/PRISM/ANd"
PRISM_SCALE_METERS = 4000

RECHARGE_RASTER_UNIT_TO_MM_DAY = {
    "mm/day": 1.0,
    "mm/d": 1.0,
    "m/day": 1000.0,
    "m/d": 1000.0,
    "m/s": 1000.0 * 86400.0,
}


def _require_earth_engine():
    try:
        import ee
        import geemap
    except ImportError as exc:
        raise ImportError(
            "Recharge extraction requires `earthengine-api` and `geemap`. "
            "Install them in the active environment before running this stage."
        ) from exc
    return ee, geemap


def initialize_earth_engine(project: str | None = None) -> None:
    """Initialize Earth Engine using credentials already present on the machine."""
    ee, _ = _require_earth_engine()
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate once with "
            "`earthengine authenticate` in the active environment, then rerun."
        ) from exc


def _read_boundary(boundary_path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError(f"Boundary contains no features: {boundary_path}")
    if gdf.crs is None:
        raise ValueError(f"Boundary has no CRS: {boundary_path}")
    return gdf


def boundary_fingerprint(boundary_path: str | Path) -> str:
    """Return a geometry-based identifier that is stable across vector formats."""
    gdf = _read_boundary(boundary_path).to_crs("EPSG:4326")
    return hashlib.sha256(gdf.geometry.union_all().wkb).hexdigest()


def boundary_to_ee(boundary_path: str | Path):
    _, geemap = _require_earth_engine()
    gdf = _read_boundary(boundary_path).to_crs("EPSG:4326")
    return geemap.gdf_to_ee(gdf).geometry()


def _normalize_recharge_raster_unit(value: object) -> str:
    unit = str(value).strip().lower().replace(" ", "")
    aliases = {
        "mmday-1": "mm/day",
        "mmd-1": "mm/d",
        "mday-1": "m/day",
        "md-1": "m/d",
        "ms-1": "m/s",
    }
    unit = aliases.get(unit, unit)
    if unit not in RECHARGE_RASTER_UNIT_TO_MM_DAY:
        allowed = ", ".join(sorted(RECHARGE_RASTER_UNIT_TO_MM_DAY))
        raise ValueError(
            f"Unsupported recharge raster unit `{value}`; use one of: {allowed}."
        )
    return unit


def load_recharge_raster_manifest(
    manifest_path: str | Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    inspect_rasters: bool = False,
) -> pd.DataFrame:
    """Load a complete daily manifest for spatial recharge rasters.

    The CSV requires ``date`` and ``raster_path``. Optional ``band`` values are
    one-based and default to 1; this permits one multiband GeoTIFF to hold the
    complete series. Optional ``units`` default to ``mm/day`` and may instead be
    ``m/day`` or ``m/s`` (including the short aliases listed above). Relative
    raster paths are resolved from the manifest directory.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    start_dt = pd.Timestamp(start_date).normalize()
    end_dt = pd.Timestamp(end_date).normalize()
    if end_dt < start_dt:
        raise ValueError("Recharge end date must be on or after the start date.")

    frame = pd.read_csv(manifest_path)
    required = {"date", "raster_path"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Recharge raster manifest lacks column(s): "
            + ", ".join(sorted(missing_columns))
        )
    if frame.empty:
        raise ValueError("Recharge raster manifest is empty.")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Recharge raster manifest contains invalid dates.")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.strftime(
            "%Y-%m-%d"
        )
        raise ValueError(
            "Recharge raster manifest contains duplicate dates: "
            + ", ".join(duplicates[:5])
        )

    if "band" not in frame:
        frame["band"] = 1
    band_values = pd.to_numeric(frame["band"], errors="coerce")
    invalid_bands = (
        band_values.isna()
        | (band_values < 1)
        | (band_values != np.floor(band_values))
    )
    if invalid_bands.any():
        raise ValueError("Recharge raster bands must be positive one-based integers.")
    frame["band"] = band_values.astype(int)

    if "units" not in frame:
        frame["units"] = "mm/day"
    frame["units"] = frame["units"].map(_normalize_recharge_raster_unit)
    if frame["raster_path"].isna().any():
        raise ValueError("Recharge raster manifest contains empty raster paths.")

    manifest_parent = manifest_path.parent

    def resolve_raster(value: object) -> Path:
        raster_path = Path(str(value)).expanduser()
        if not raster_path.is_absolute():
            raster_path = manifest_parent / raster_path
        return raster_path.resolve()

    frame["raster_path"] = frame["raster_path"].map(resolve_raster)
    frame = frame.loc[
        (frame["date"] >= start_dt) & (frame["date"] <= end_dt)
    ].sort_values("date")
    if frame.empty:
        raise ValueError(
            f"No recharge rasters found for period {start_dt.date()} to {end_dt.date()}."
        )
    expected = pd.date_range(start_dt, end_dt, freq="D")
    missing_dates = expected.difference(frame["date"])
    if len(missing_dates):
        preview = ", ".join(value.strftime("%Y-%m-%d") for value in missing_dates[:5])
        raise ValueError(
            f"Recharge raster forcing is missing {len(missing_dates)} requested day(s), "
            f"including {preview}."
        )

    missing_paths = sorted(
        {path for path in frame["raster_path"] if not Path(path).exists()},
        key=str,
    )
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(
            f"Recharge raster manifest references {len(missing_paths)} missing file(s): "
            f"{preview}"
        )

    if inspect_rasters:
        for raster_path, rows in frame.groupby("raster_path", sort=False):
            with rasterio.open(raster_path) as source:
                if source.crs is None:
                    raise ValueError(f"Recharge raster has no CRS: {raster_path}")
                maximum_band = int(rows["band"].max())
                if maximum_band > source.count:
                    raise ValueError(
                        f"Recharge manifest requests band {maximum_band} from "
                        f"{raster_path}, which has {source.count} band(s)."
                    )

    return frame.reset_index(drop=True)


def _records_from_ee(data: dict[str, Any], value_name: str) -> pd.DataFrame:
    records = [
        {"date": key[-8:], value_name: value}
        for key, value in data.items()
        if value is not None
    ]
    if not records:
        return pd.DataFrame(columns=[value_name], index=pd.DatetimeIndex([], name="date"))
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").sort_index()
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()
        raise ValueError(f"Earth Engine returned duplicate dates: {duplicates[:5].tolist()}")
    return frame


def _chunk_path(
    cache_dir: Path | None,
    dataset: str,
    fingerprint: str,
    start_year: int,
    end_year: int,
) -> Path | None:
    if cache_dir is None:
        return None
    path = cache_dir / "chunks" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{fingerprint[:12]}_{start_year}_{end_year}.csv"


def _load_chunk(path: Path | None, value_name: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if value_name not in frame:
        raise ValueError(f"Cached forcing chunk lacks `{value_name}`: {path}")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Cached forcing chunk has invalid dates: {path}")
    return frame[[value_name]]


def _save_chunk(frame: pd.DataFrame, path: Path | None) -> None:
    if path is not None:
        frame.rename_axis("date").reset_index().to_csv(path, index=False)


def extract_pml_et_composites(
    boundary_path: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    chunk_years: int = 5,
    ee_project: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Extract watershed-mean PML V2.2a eight-day mean daily ET rates."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    if chunk_years < 1:
        raise ValueError("chunk_years must be at least one")

    cache_path = Path(cache_dir) if cache_dir is not None else None
    fingerprint = boundary_fingerprint(boundary_path)
    roi = None
    collection = None
    chunks: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1, chunk_years):
        chunk_end = min(year + chunk_years - 1, end_year)
        path = _chunk_path(cache_path, "pml_v22a", fingerprint, year, chunk_end)
        cached = None if refresh else _load_chunk(path, "ET_mm_day")
        if cached is not None:
            chunks.append(cached)
            continue

        if collection is None:
            ee, _ = _require_earth_engine()
            initialize_earth_engine(ee_project)
            roi = boundary_to_ee(boundary_path)
            collection = ee.ImageCollection(PML_COLLECTION)

        def prepare_image(img):
            return img.select("ET").multiply(PML_SCALE_FACTOR).rename(
                img.date().format("YYYYMMdd")
            )

        filtered = collection.filterDate(
            f"{year}-01-01", f"{chunk_end + 1}-01-01"
        ).map(prepare_image)
        data = filtered.toBands().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=PML_SCALE_METERS,
            maxPixels=1e9,
            tileScale=8,
        ).getInfo()
        frame = _records_from_ee(data, "ET_mm_day")
        if frame.empty:
            raise ValueError(f"Earth Engine returned no PML ET for {year}-{chunk_end}.")
        _save_chunk(frame, path)
        chunks.append(frame)

    result = pd.concat(chunks).sort_index()
    if result.index.has_duplicates:
        raise ValueError("PML extraction produced duplicate composite dates.")
    return result


def expand_pml_composites_to_daily(
    composites: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Expand PML eight-day mean daily rates as piecewise-constant daily rates.

    PML timestamps identify the start of each eight-day averaging interval. Holding
    the documented mean rate over that interval preserves the composite water
    volume. The last value may cover at most eight days including its timestamp.
    """
    if "ET_mm_day" not in composites:
        raise ValueError("PML composites must contain an `ET_mm_day` column.")
    frame = composites[["ET_mm_day"]].copy().sort_index()
    if frame.empty:
        raise ValueError("PML composites are empty.")
    if frame.index.has_duplicates:
        raise ValueError("PML composites contain duplicate dates.")
    if frame["ET_mm_day"].isna().any() or (frame["ET_mm_day"] < 0.0).any():
        raise ValueError("PML ET must be complete and nonnegative.")

    gaps = frame.index.to_series().diff().dropna().dt.days
    bad_gaps = gaps[gaps > PML_COMPOSITE_DAYS]
    if not bad_gaps.empty:
        when = bad_gaps.index[0].strftime("%Y-%m-%d")
        raise ValueError(f"PML composites have a gap longer than 8 days at {when}.")

    first = frame.index.min()
    supported_end = frame.index.max() + pd.Timedelta(days=PML_COMPOSITE_DAYS - 1)
    requested_start = pd.Timestamp(start_date) if start_date is not None else first
    requested_end = pd.Timestamp(end_date) if end_date is not None else supported_end
    actual_start = max(requested_start, first)
    if requested_end > supported_end:
        raise ValueError(
            f"Requested ET through {requested_end.date()}, but the last PML composite "
            f"only supports daily expansion through {supported_end.date()}."
        )
    if requested_end < actual_start:
        raise ValueError("Requested dates do not overlap the available PML composites.")

    daily_index = pd.date_range(actual_start, requested_end, freq="D")
    daily = frame.reindex(frame.index.union(daily_index)).sort_index().ffill().loc[daily_index]
    daily.index.name = "date"
    return daily.rename(columns={"ET_mm_day": "ET"})


def extract_pml_et(
    boundary_path: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    chunk_years: int = 5,
    ee_project: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Backward-compatible PML extraction returning daily ET."""
    composites = extract_pml_et_composites(
        boundary_path,
        start_year=start_year,
        end_year=end_year,
        chunk_years=chunk_years,
        ee_project=ee_project,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    daily = expand_pml_composites_to_daily(
        composites,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )
    return daily.rename(columns={"ET": "ET_interpolated"}).assign(
        ET_mm_day=lambda value: value["ET_interpolated"]
    )[["ET_mm_day", "ET_interpolated"]]


def extract_prism_ppt(
    boundary_path: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    ee_project: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Extract watershed-average daily PRISM precipitation with yearly caching."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    cache_path = Path(cache_dir) if cache_dir is not None else None
    fingerprint = boundary_fingerprint(boundary_path)
    roi = None
    collection = None
    chunks: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        path = _chunk_path(cache_path, "prism_and", fingerprint, year, year)
        cached = None if refresh else _load_chunk(path, "P")
        if cached is not None:
            chunks.append(cached)
            continue

        if collection is None:
            ee, _ = _require_earth_engine()
            initialize_earth_engine(ee_project)
            roi = boundary_to_ee(boundary_path)
            collection = ee.ImageCollection(PRISM_COLLECTION).select("ppt")

        def prepare_image(img):
            return img.rename(img.date().format("YYYYMMdd"))

        filtered = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").map(
            prepare_image
        )
        data = filtered.toBands().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=PRISM_SCALE_METERS,
            maxPixels=1e9,
            tileScale=8,
        ).getInfo()
        frame = _records_from_ee(data, "P")
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        missing = expected.difference(frame.index)
        if len(missing):
            preview = ", ".join(value.strftime("%Y-%m-%d") for value in missing[:5])
            raise ValueError(
                f"PRISM extraction for {year} is missing {len(missing)} day(s), "
                f"including {preview}."
            )
        frame = frame.reindex(expected).rename_axis("date")
        _save_chunk(frame, path)
        chunks.append(frame)

    result = pd.concat(chunks).sort_index()
    if result.index.has_duplicates:
        raise ValueError("PRISM extraction produced duplicate dates.")
    return result.rename(columns={"P": "ppt_mm"})


def compute_storage_deficit_recharge(
    et_daily: pd.DataFrame,
    ppt_daily: pd.DataFrame,
    *,
    initial_deficit_mm: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute recharge as precipitation surplus after filling storage deficit."""
    if initial_deficit_mm < 0.0:
        raise ValueError("initial_deficit_mm must be nonnegative")
    et_col = "ET_interpolated" if "ET_interpolated" in et_daily.columns else "ET"
    if et_col not in et_daily and "ET_mm_day" in et_daily:
        et_col = "ET_mm_day"
    ppt_col = "ppt_mm" if "ppt_mm" in ppt_daily.columns else "P"

    df = et_daily[[et_col]].join(ppt_daily[[ppt_col]], how="inner")
    df = df.rename(columns={et_col: "ET", ppt_col: "P"})
    if start_date or end_date:
        df = df.loc[start_date:end_date].copy()

    if df.empty:
        raise ValueError("ET and precipitation have no overlapping dates.")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = expected.difference(df.index)
    if len(missing):
        raise ValueError(f"Water-balance forcing is missing {len(missing)} daily date(s).")
    if df[["ET", "P"]].isna().any().any():
        raise ValueError("Water-balance forcing contains missing ET or precipitation values.")
    if (df[["ET", "P"]] < 0.0).any().any():
        raise ValueError("ET and precipitation must be nonnegative.")

    deficits = np.empty(len(df), dtype=float)
    recharge = np.empty(len(df), dtype=float)
    current_deficit = float(initial_deficit_mm)
    for index, (et_value, precipitation) in enumerate(df[["ET", "P"]].itertuples(index=False)):
        balance = current_deficit + et_value - precipitation
        current_deficit = max(balance, 0.0)
        deficits[index] = current_deficit
        recharge[index] = max(-balance, 0.0)

    df["Deficit"] = deficits
    df["Recharge"] = recharge
    return df.reset_index(names="date")


def validate_water_balance(
    frame: pd.DataFrame, initial_deficit_mm: float = 0.0
) -> dict[str, float]:
    """Validate and summarize P - ET = Recharge - change in deficit."""
    required = {"date", "ET", "P", "Deficit", "Recharge"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Recharge table lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Recharge table is empty.")
    previous = frame["Deficit"].shift(fill_value=float(initial_deficit_mm))
    residual = frame["P"] - frame["ET"] - frame["Recharge"] + frame["Deficit"] - previous
    max_error = float(residual.abs().max())
    if max_error > 1e-9:
        raise ValueError(f"Storage-deficit water balance does not close: {max_error:.3g} mm")
    return {
        "precipitation_mm": float(frame["P"].sum()),
        "et_mm": float(frame["ET"].sum()),
        "recharge_mm": float(frame["Recharge"].sum()),
        "initial_deficit_mm": float(initial_deficit_mm),
        "final_deficit_mm": float(frame["Deficit"].iloc[-1]),
        "max_daily_balance_error_mm": max_error,
    }


def build_recharge_csv(
    boundary_path: str | Path,
    output_csv: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    initial_deficit_mm: float = 0.0,
    ee_project: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> Path:
    """Fetch cached forcing, compute recharge locally, and write provenance files."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir is not None else output_csv.parent / "forcing_cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    composites = extract_pml_et_composites(
        boundary_path,
        start_year=start_year,
        end_year=end_year,
        ee_project=ee_project,
        cache_dir=cache_path,
        refresh=refresh,
    )
    precipitation = extract_prism_ppt(
        boundary_path,
        start_year=start_year,
        end_year=end_year,
        ee_project=ee_project,
        cache_dir=cache_path,
        refresh=refresh,
    )
    et_daily = expand_pml_composites_to_daily(
        composites,
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
    )
    water_balance = compute_storage_deficit_recharge(
        et_daily, precipitation, initial_deficit_mm=initial_deficit_mm
    )
    summary = validate_water_balance(water_balance, initial_deficit_mm)

    composites.rename_axis("date").reset_index().to_csv(
        cache_path / "pml_et_8day.csv", index=False
    )
    precipitation.rename_axis("date").reset_index().to_csv(
        cache_path / "prism_ppt_daily.csv", index=False
    )
    water_balance[["date", "ET", "P"]].to_csv(
        cache_path / "forcing_daily.csv", index=False
    )
    water_balance.to_csv(output_csv, index=False)

    gdf = _read_boundary(boundary_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": str(Path(boundary_path).resolve()),
        "boundary_fingerprint_sha256": boundary_fingerprint(boundary_path),
        "boundary_area_km2": float(gdf.to_crs("EPSG:5070").geometry.area.sum() / 1e6),
        "requested_period": [f"{start_year}-01-01", f"{end_year}-12-31"],
        "actual_period": [
            water_balance["date"].min().strftime("%Y-%m-%d"),
            water_balance["date"].max().strftime("%Y-%m-%d"),
        ],
        "pml": {
            "collection": PML_COLLECTION,
            "band": "ET",
            "scale_factor": PML_SCALE_FACTOR,
            "reduction_scale_m": PML_SCALE_METERS,
            "temporal_interpretation": "eight-day mean daily rate held over each composite interval",
            "first_composite": composites.index.min().strftime("%Y-%m-%d"),
            "last_composite": composites.index.max().strftime("%Y-%m-%d"),
        },
        "prism": {
            "collection": PRISM_COLLECTION,
            "band": "ppt",
            "reduction_scale_m": PRISM_SCALE_METERS,
        },
        "storage_deficit": {
            "method": "unbounded cumulative deficit; surplus becomes recharge",
            **summary,
        },
        "artifacts": {
            "pml_composites": str((cache_path / "pml_et_8day.csv").resolve()),
            "prism_daily": str((cache_path / "prism_ppt_daily.csv").resolve()),
            "forcing_daily": str((cache_path / "forcing_daily.csv").resolve()),
            "recharge_daily": str(output_csv.resolve()),
        },
    }
    output_csv.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_csv
