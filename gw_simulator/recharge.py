from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


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
    ee, _ = _require_earth_engine()
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def boundary_to_ee(boundary_path: str | Path):
    _, geemap = _require_earth_engine()
    gdf = gpd.read_file(boundary_path)
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return geemap.gdf_to_ee(gdf)


def extract_pml_et(
    boundary_path: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    chunk_years: int = 5,
    ee_project: str | None = None,
) -> pd.DataFrame:
    """Extract watershed-average PML V2 total ET and interpolate to daily values."""
    ee, _ = _require_earth_engine()
    initialize_earth_engine(ee_project)
    roi = boundary_to_ee(boundary_path)
    collection = ee.ImageCollection("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a")

    def prepare_image(img):
        total_et = img.select("ET").multiply(0.01)
        return total_et.rename(img.date().format("YYYYMMdd"))

    def get_data_chunk(start: int, end: int):
        filtered = collection.filterDate(f"{start}-01-01", f"{end}-12-31").map(prepare_image)
        stack = filtered.toBands()
        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=1e9,
            tileScale=8,
        )
        return stats.getInfo()

    full_data: dict[str, float] = {}
    for year in range(start_year, end_year + 1, chunk_years):
        chunk_end = min(year + chunk_years - 1, end_year)
        full_data.update(get_data_chunk(year, chunk_end))

    records = [{"date": key[-8:], "ET_mm_day": value} for key, value in full_data.items()]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    daily = df.resample("D").mean()
    daily["ET_interpolated"] = daily["ET_mm_day"].interpolate(method="linear")
    return daily


def extract_prism_ppt(
    boundary_path: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    ee_project: str | None = None,
) -> pd.DataFrame:
    """Extract watershed-average daily PRISM precipitation."""
    ee, _ = _require_earth_engine()
    initialize_earth_engine(ee_project)
    roi = boundary_to_ee(boundary_path)
    collection = ee.ImageCollection("OREGONSTATE/PRISM/ANd").select("ppt")

    def prepare_image(img):
        return img.rename(img.date().format("YYYYMMdd"))

    def get_prism_chunk(year: int):
        filtered = collection.filterDate(f"{year}-01-01", f"{year}-12-31").map(prepare_image)
        stack = filtered.toBands()
        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=4000,
            maxPixels=1e9,
            tileScale=8,
        )
        return stats.getInfo()

    full_data: dict[str, float] = {}
    for year in range(start_year, end_year + 1):
        full_data.update(get_prism_chunk(year))

    records = [{"date": key[-8:], "ppt_mm": value} for key, value in full_data.items()]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").sort_index()


def compute_storage_deficit_recharge(
    et_daily: pd.DataFrame,
    ppt_daily: pd.DataFrame,
    *,
    initial_deficit_mm: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute recharge as surplus after filling a root-zone storage deficit."""
    et_col = "ET_interpolated" if "ET_interpolated" in et_daily.columns else "ET_mm_day"
    ppt_col = "ppt_mm" if "ppt_mm" in ppt_daily.columns else "P"

    df = et_daily[[et_col]].join(ppt_daily[[ppt_col]], how="inner")
    df = df.rename(columns={et_col: "ET", ppt_col: "P"})
    df["P"] = df["P"].fillna(0.0)
    df["ET"] = df["ET"].ffill().bfill()

    if start_date or end_date:
        df = df.loc[start_date:end_date].copy()

    deficit = []
    recharge = []
    current_deficit = float(initial_deficit_mm)

    for _, row in df.iterrows():
        balance = current_deficit + (row["ET"] - row["P"])
        if balance > 0:
            current_deficit = balance
            current_recharge = 0.0
        else:
            current_deficit = 0.0
            current_recharge = abs(balance)
        deficit.append(current_deficit)
        recharge.append(current_recharge)

    df["Deficit"] = np.asarray(deficit)
    df["Recharge"] = np.asarray(recharge)
    return df.reset_index()


def build_recharge_csv(
    boundary_path: str | Path,
    output_csv: str | Path,
    *,
    start_year: int = 2000,
    end_year: int = 2024,
    initial_deficit_mm: float = 0.0,
    ee_project: str | None = None,
) -> Path:
    et = extract_pml_et(boundary_path, start_year=start_year, end_year=end_year, ee_project=ee_project)
    ppt = extract_prism_ppt(boundary_path, start_year=start_year, end_year=end_year, ee_project=ee_project)
    water_balance = compute_storage_deficit_recharge(
        et,
        ppt,
        initial_deficit_mm=initial_deficit_mm,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    water_balance.to_csv(output_csv, index=False)
    return output_csv
