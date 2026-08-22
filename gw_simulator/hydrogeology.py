from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import reproject
from rasterio.transform import from_origin


GLHYMPS2_DOI = "https://doi.org/10.5683/SP2/TTJNIU"
GLHYMPS2_DATAFILE_ID = 71909
GLHYMPS2_RELEASE = "1.0 (2018-10-25)"
SHANGGUAN_DTB_DOI = "https://doi.org/10.1002/2016MS000686"
SHANGGUAN_DTB_URL = (
    "https://files.isric.org/soilgrids/former/2017-03-10/data/"
    "BDTICM_M_250m_ll.tif"
)
PELLETIER2016_DOI = "https://doi.org/10.3334/ORNLDAAC/1304"
PELLETIER2016_PUBLICATION_DOI = "https://doi.org/10.1002/2015MS000526"
PELLETIER2016_RELEASE = "2016-02-03"
GLHYMPS_HYDRAULIC_CONDUCTIVITY_FACTOR = 1.0e7
SECONDS_PER_DAY = 86400.0
GLHYMPS2_FIELDS = [
    "OBJECTID_1",
    "IDENTITY_",
    "logK_Ferr_",
    "Porosity_x",
    "K_stdev_x1",
    "GUM_K",
    "Prmfrst",
]


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_shapefile_path(zip_path: str | Path) -> str:
    return f"/vsizip/{Path(zip_path).resolve()}/GLHYMPS.shp"


def _read_boundary(path: str | Path) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(path)
    if boundary.empty or boundary.crs is None:
        raise ValueError(f"Boundary must contain geometry with a CRS: {path}")
    boundary = boundary.loc[~boundary.geometry.is_empty & boundary.geometry.notna()].copy()
    if boundary.empty:
        raise ValueError(f"Boundary contains no usable geometry: {path}")
    return boundary


def read_glhymps2_clip(
    zip_path: str | Path,
    boundary_path: str | Path,
    *,
    buffer_m: float = 1000.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Read only GLHYMPS features near a boundary from the global ZIP archive."""
    if buffer_m < 0.0:
        raise ValueError("buffer_m cannot be negative")
    source_path = _zip_shapefile_path(zip_path)
    info = pyogrio.read_info(source_path)
    source_crs = CRS.from_user_input(info["crs"])
    boundary = _read_boundary(boundary_path).to_crs(source_crs)
    query = boundary.geometry.union_all().buffer(buffer_m)
    features = pyogrio.read_dataframe(
        source_path,
        bbox=query.bounds,
        columns=GLHYMPS2_FIELDS,
    )
    if features.empty:
        raise ValueError("No GLHYMPS 2.0 polygons overlap the requested boundary.")
    features = features.loc[features.intersects(query)].copy()
    if features.empty:
        raise ValueError("No GLHYMPS 2.0 polygons intersect the buffered boundary.")
    features["porosity_fraction"] = features["Porosity_x"].astype(float) / 100.0
    features["log10_permeability_m2"] = features["logK_Ferr_"].astype(float) / 100.0
    if (
        features["porosity_fraction"].isna().any()
        or (features["porosity_fraction"] <= 0.0).any()
        or (features["porosity_fraction"] > 1.0).any()
    ):
        raise ValueError("GLHYMPS clip contains invalid porosity values.")
    return features, boundary


def _projected_grid(
    boundary: gpd.GeoDataFrame,
    *,
    target_crs: str,
    resolution_m: float,
    buffer_m: float,
) -> tuple[gpd.GeoDataFrame, tuple[int, int], rasterio.Affine, tuple[float, float, float, float]]:
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive")
    projected = boundary.to_crs(target_crs)
    min_x, min_y, max_x, max_y = projected.geometry.union_all().buffer(buffer_m).bounds
    left = np.floor(min_x / resolution_m) * resolution_m
    bottom = np.floor(min_y / resolution_m) * resolution_m
    right = np.ceil(max_x / resolution_m) * resolution_m
    top = np.ceil(max_y / resolution_m) * resolution_m
    width = int(round((right - left) / resolution_m))
    height = int(round((top - bottom) / resolution_m))
    transform = from_origin(left, top, resolution_m, resolution_m)
    return projected, (height, width), transform, (left, right, bottom, top)


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    crs,
    transform,
    description: str,
    tags: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(values.astype("float32"), 1)
        destination.set_band_description(1, description)
        destination.update_tags(**(tags or {}))


def _read_remote_depth_to_bedrock(
    source_url: str,
    *,
    shape: tuple[int, int],
    transform,
    crs,
) -> tuple[np.ndarray, dict[str, str]]:
    """Read a spatial window of the official absolute depth-to-bedrock raster."""
    source_candidate = Path(source_url)
    if source_candidate.exists():
        source_path = str(source_candidate.resolve())
    else:
        source_path = source_url if source_url.startswith("/vsi") else f"/vsicurl/{source_url}"
    depth_cm = np.full(shape, np.nan, dtype="float32")
    with rasterio.Env(
        GDAL_HTTP_MULTIRANGE="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    ):
        with rasterio.open(source_path) as source:
            source_tags = source.tags()
            if source_tags.get("ATTRIBUTE_LABEL") != "BDTICM_M":
                raise ValueError(
                    "Depth source is not the expected SoilGrids BDTICM absolute-depth layer."
                )
            if source_tags.get("ATTRIBUTE_UNITS_OF_MEASURE") != "cm":
                raise ValueError("SoilGrids absolute depth-to-bedrock must be in cm.")
            if (
                source.shape == shape
                and source.crs == crs
                and source.transform.almost_equals(transform)
            ):
                depth_cm[:] = source.read(1, masked=True).filled(np.nan)
            else:
                reproject(
                    rasterio.band(source, 1),
                    depth_cm,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=transform,
                    dst_crs=crs,
                    dst_nodata=np.nan,
                    resampling=rasterio.enums.Resampling.bilinear,
                )
    depth_m = depth_cm.astype(float) / 100.0
    return depth_m, source_tags


def _read_aligned_raster(
    path: str | Path,
    *,
    shape: tuple[int, int],
    transform,
    crs,
    resampling=rasterio.enums.Resampling.bilinear,
) -> np.ndarray:
    output = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as source:
        reproject(
            rasterio.band(source, 1),
            output,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return output


def _pelletier_landform_thickness(
    *,
    regolith_path: str | Path,
    sediment_path: str | Path,
    land_cover_path: str | Path,
    shape: tuple[int, int],
    transform,
    crs,
    inside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the landform-aware Pelletier depth to unweathered bedrock."""
    land_cover = _read_aligned_raster(
        land_cover_path,
        shape=shape,
        transform=transform,
        crs=crs,
        resampling=rasterio.enums.Resampling.nearest,
    )
    regolith = _read_aligned_raster(
        regolith_path,
        shape=shape,
        transform=transform,
        crs=crs,
    )
    sediment = _read_aligned_raster(
        sediment_path,
        shape=shape,
        transform=transform,
        crs=crs,
    )

    rounded_land_cover = np.rint(land_cover)
    if np.any(~np.isfinite(rounded_land_cover[inside])):
        raise ValueError("Pelletier land-cover mask has uncovered watershed cells.")
    basin_classes = set(rounded_land_cover[inside].astype(int))
    unsupported = basin_classes.difference({1, 2})
    if unsupported:
        raise ValueError(
            "Pelletier thickness cannot define aquifer depth for land-cover class(es) "
            f"{sorted(unsupported)}; mask or handle those water/ice cells explicitly."
        )

    thickness = np.full(shape, np.nan, dtype=float)
    upland = rounded_land_cover == 1
    lowland = rounded_land_cover == 2
    thickness[upland] = regolith[upland]
    thickness[lowland] = sediment[lowland]
    if np.any(~np.isfinite(thickness[inside])) or np.any(thickness[inside] <= 0.0):
        raise ValueError("Pelletier landform thickness is invalid inside the watershed.")
    return thickness, rounded_land_cover, regolith, sediment


def _field_summary(values: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    selected = values[inside & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("Hydrogeology field has no finite values inside the watershed.")
    return {
        "minimum": float(selected.min()),
        "median": float(np.median(selected)),
        "area_weighted_mean": float(selected.mean()),
        "maximum": float(selected.max()),
    }


def _paired_field_summary(
    official: np.ndarray,
    legacy: np.ndarray,
    inside: np.ndarray,
) -> dict[str, object]:
    valid = inside & np.isfinite(official) & np.isfinite(legacy)
    if not np.any(valid):
        raise ValueError("Official and legacy hydrogeology fields do not overlap.")
    difference = official[valid] - legacy[valid]
    ratio = np.divide(
        official[valid],
        legacy[valid],
        out=np.full(valid.sum(), np.nan),
        where=legacy[valid] != 0.0,
    )
    finite_ratio = ratio[np.isfinite(ratio)]
    return {
        "official": _field_summary(official, valid),
        "legacy": _field_summary(legacy, valid),
        "official_minus_legacy": {
            "minimum": float(difference.min()),
            "median": float(np.median(difference)),
            "area_weighted_mean": float(difference.mean()),
            "maximum": float(difference.max()),
            "mean_absolute_difference": float(np.abs(difference).mean()),
        },
        "official_divided_by_legacy": {
            "minimum": float(finite_ratio.min()),
            "median": float(np.median(finite_ratio)),
            "area_weighted_mean": float(finite_ratio.mean()),
            "maximum": float(finite_ratio.max()),
        },
    }


def _comparison_summary(
    official: np.ndarray,
    existing: np.ndarray,
    inside: np.ndarray,
    cell_area_m2: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    valid = inside & np.isfinite(official) & np.isfinite(existing)
    if not np.any(valid):
        raise ValueError("Existing and official porosity rasters do not overlap in the basin.")
    pairs = pd.DataFrame(
        {
            "existing_specific_yield": existing[valid],
            "glhymps_total_porosity": official[valid],
        }
    )
    pair_table = (
        pairs.groupby(["existing_specific_yield", "glhymps_total_porosity"], dropna=False)
        .size()
        .rename("cell_count")
        .reset_index()
    )
    pair_table["area_km2"] = pair_table["cell_count"] * cell_area_m2 / 1e6
    difference = official[valid] - existing[valid]
    summary = {
        "compared_cell_count": int(valid.sum()),
        "compared_area_km2": float(valid.sum() * cell_area_m2 / 1e6),
        "existing": {
            "minimum": float(existing[valid].min()),
            "median": float(np.median(existing[valid])),
            "area_weighted_mean": float(existing[valid].mean()),
            "maximum": float(existing[valid].max()),
        },
        "glhymps_total_porosity": {
            "minimum": float(official[valid].min()),
            "median": float(np.median(official[valid])),
            "area_weighted_mean": float(official[valid].mean()),
            "maximum": float(official[valid].max()),
        },
        "glhymps_minus_existing": {
            "minimum": float(difference.min()),
            "median": float(np.median(difference)),
            "area_weighted_mean": float(difference.mean()),
            "maximum": float(difference.max()),
            "mean_absolute_difference": float(np.abs(difference).mean()),
        },
    }
    return pair_table.sort_values("area_km2", ascending=False), summary


def _save_comparison_plot(
    path: Path,
    official: np.ndarray,
    existing: np.ndarray,
    inside: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    official_masked = np.where(inside, official, np.nan)
    existing_masked = np.where(inside, existing, np.nan)
    difference = official_masked - existing_masked
    finite = inside & np.isfinite(official) & np.isfinite(existing)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    shared_min = float(np.nanmin([official_masked, existing_masked]))
    shared_max = float(np.nanmax([official_masked, existing_masked]))
    for axis, values, title in [
        (axes[0, 0], existing_masked, "Existing input raster"),
        (axes[0, 1], official_masked, "GLHYMPS 2.0 total porosity"),
    ]:
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            cmap="viridis",
            vmin=shared_min,
            vmax=shared_max,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
    fig.colorbar(image, ax=[axes[0, 0], axes[0, 1]], label="Fraction", shrink=0.8)

    limit = float(np.nanmax(np.abs(difference)))
    diff_image = axes[1, 0].imshow(
        difference,
        extent=extent,
        origin="upper",
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
    )
    axes[1, 0].set_title("GLHYMPS minus existing")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(diff_image, ax=axes[1, 0], label="Fraction", shrink=0.8)

    x = existing[finite]
    y = official[finite]
    axes[1, 1].hexbin(x, y, gridsize=35, mincnt=1, bins="log", cmap="magma")
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    axes[1, 1].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[1, 1].set(xlabel="Existing input", ylabel="GLHYMPS total porosity")
    axes[1, 1].set_title("Cell comparison")
    for axis in axes.flat[:3]:
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
    fig.suptitle("Green Valley hydrogeology provenance check", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_hydrogeology_plot(
    path: Path,
    *,
    official_fields: dict[str, np.ndarray],
    legacy_fields: dict[str, np.ndarray],
    inside: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    rows = [
        ("Specific-yield proxy", "porosity", "Fraction", False),
        ("Absolute depth to bedrock", "depth", "m", False),
        ("Transmissivity", "transmissivity", "m2/day", True),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    for row, (title, key, units, logarithmic) in enumerate(rows):
        official = np.where(inside, official_fields[key], np.nan)
        legacy = np.where(inside, legacy_fields[key], np.nan)
        if logarithmic:
            official_plot = np.log10(official)
            legacy_plot = np.log10(legacy)
            label = f"log10({units})"
        else:
            official_plot = official
            legacy_plot = legacy
            label = units
        vmin = float(np.nanmin([official_plot, legacy_plot]))
        vmax = float(np.nanmax([official_plot, legacy_plot]))
        for column, (values, subtitle) in enumerate(
            [(legacy_plot, "Legacy"), (official_plot, "Official")]
        ):
            image = axes[row, column].imshow(
                values,
                extent=extent,
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, column].set_title(f"{title}: {subtitle}")
            axes[row, column].set_aspect("equal")
        fig.colorbar(image, ax=[axes[row, 0], axes[row, 1]], label=label, shrink=0.75)

        ratio = np.divide(
            official,
            legacy,
            out=np.full_like(official, np.nan, dtype=float),
            where=np.isfinite(legacy) & (legacy > 0.0),
        )
        ratio_image = axes[row, 2].imshow(
            np.log10(ratio),
            extent=extent,
            origin="upper",
            cmap="RdBu",
        )
        axes[row, 2].set_title(f"{title}: log10(official / legacy)")
        axes[row, 2].set_aspect("equal")
        fig.colorbar(ratio_image, ax=axes[row, 2], label="log10 ratio", shrink=0.75)
    for axis in axes.flat:
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
    fig.suptitle("Official hydrogeology baseline and legacy comparison", fontsize=16)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def prepare_glhymps2_porosity(
    *,
    zip_path: str | Path,
    boundary_path: str | Path,
    reference_raster: str | Path,
    existing_porosity_raster: str | Path | None = None,
    existing_depth_raster: str | Path | None = None,
    existing_transmissivity_raster: str | Path | None = None,
    output_dir: str | Path,
    buffer_m: float = 1000.0,
    target_crs: str = "EPSG:26910",
    resolution_m: float = 50.0,
    depth_source: str | Path | None = None,
    pelletier_regolith_raster: str | Path | None = None,
    pelletier_sediment_raster: str | Path | None = None,
    pelletier_land_cover_raster: str | Path | None = None,
) -> dict[str, Path | None]:
    """Create a coherent official hydrogeology package and comparison artifacts.

    When all three Pelletier rasters are supplied, the primary aquifer thickness is
    regolith on upland cells and sediment on lowland cells. Shangguan soil depth and
    legacy comparison rasters are optional and never affect primary transmissivity.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    features, boundary_source = read_glhymps2_clip(
        zip_path, boundary_path, buffer_m=buffer_m
    )
    clip_path = output_dir / "glhymps2_source_clip.gpkg"
    features.to_file(clip_path, layer="glhymps2", driver="GPKG")

    if not Path(reference_raster).exists():
        raise FileNotFoundError(reference_raster)
    boundary, shape, transform, output_bounds = _projected_grid(
        boundary_source,
        target_crs=target_crs,
        resolution_m=resolution_m,
        buffer_m=buffer_m,
    )
    crs = boundary.crs

    features_reference = features.to_crs(crs)
    official = rasterize(
        (
            (geometry, value)
            for geometry, value in zip(
                features_reference.geometry,
                features_reference["porosity_fraction"],
            )
        ),
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=False,
    )
    inside = geometry_mask(
        boundary.geometry,
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    if np.any(~np.isfinite(official[inside])):
        raise ValueError("GLHYMPS raster has uncovered cells inside the watershed.")

    log10_permeability = rasterize(
        (
            (geometry, value)
            for geometry, value in zip(
                features_reference.geometry,
                features_reference["log10_permeability_m2"],
            )
        ),
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=False,
    ).astype(float)
    if np.any(~np.isfinite(log10_permeability[inside])):
        raise ValueError("GLHYMPS permeability has uncovered cells inside the watershed.")

    pelletier_sources = (
        pelletier_regolith_raster,
        pelletier_sediment_raster,
        pelletier_land_cover_raster,
    )
    if any(value is not None for value in pelletier_sources) and not all(
        value is not None for value in pelletier_sources
    ):
        raise ValueError(
            "Provide all three Pelletier rasters (regolith, sediment, and land cover)."
        )
    use_pelletier = all(value is not None for value in pelletier_sources)
    if depth_source is None and not use_pelletier:
        depth_source = SHANGGUAN_DTB_URL
    shangguan_depth_m = None
    depth_source_tags: dict[str, str] = {}
    if depth_source is not None:
        shangguan_depth_m, depth_source_tags = _read_remote_depth_to_bedrock(
            str(depth_source),
            shape=shape,
            transform=transform,
            crs=crs,
        )
        if np.any(~np.isfinite(shangguan_depth_m[inside])) or np.any(
            shangguan_depth_m[inside] <= 0.0
        ):
            raise ValueError("Official depth-to-bedrock is invalid inside the watershed.")
    pelletier_land_cover = None
    pelletier_regolith = None
    pelletier_sediment = None
    if use_pelletier:
        for value in pelletier_sources:
            if not Path(value).exists():
                raise FileNotFoundError(value)
        (
            depth_m,
            pelletier_land_cover,
            pelletier_regolith,
            pelletier_sediment,
        ) = _pelletier_landform_thickness(
            regolith_path=pelletier_regolith_raster,
            sediment_path=pelletier_sediment_raster,
            land_cover_path=pelletier_land_cover_raster,
            shape=shape,
            transform=transform,
            crs=crs,
            inside=inside,
        )
    else:
        if shangguan_depth_m is None:
            raise ValueError("A Shangguan depth source is required without Pelletier data.")
        depth_m = shangguan_depth_m

    permeability_m2 = np.power(10.0, log10_permeability)
    hydraulic_conductivity_m_s = (
        permeability_m2 * GLHYMPS_HYDRAULIC_CONDUCTIVITY_FACTOR
    )
    hydraulic_conductivity_m_day = hydraulic_conductivity_m_s * SECONDS_PER_DAY
    transmissivity_m2_day = hydraulic_conductivity_m_day * depth_m

    porosity_path = output_dir / "specific_yield_glhymps2_total_porosity.tif"
    _write_raster(
        porosity_path,
        official,
        crs=crs,
        transform=transform,
        description="glhymps2_total_porosity_fraction",
        tags={
            "source": "GLHYMPS 2.0",
            "source_doi": GLHYMPS2_DOI,
            "interpretation": "total porosity used as a specific-yield proxy",
            "conversion": "Porosity_x / 100",
        },
    )

    shangguan_depth_path = None
    if shangguan_depth_m is not None:
        shangguan_depth_path = output_dir / "depth_to_bedrock_shangguan2017_m.tif"
        _write_raster(
            shangguan_depth_path,
            shangguan_depth_m,
            crs=crs,
            transform=transform,
            description="absolute_depth_to_bedrock_m",
            tags={
                "source": "SoilGrids250m BDTICM_M (2017-03-10)",
                "source_doi": SHANGGUAN_DTB_DOI,
                "source_url": str(depth_source),
                "conversion": "BDTICM_M centimeters / 100",
            },
        )
    if use_pelletier:
        depth_path = output_dir / "depth_to_unweathered_bedrock_pelletier2016_m.tif"
        _write_raster(
            depth_path,
            depth_m,
            crs=crs,
            transform=transform,
            description="pelletier2016_landform_thickness_m",
            tags={
                "source": "Pelletier et al. (2016), ORNL DAAC 1304",
                "source_doi": PELLETIER2016_DOI,
                "interpretation": (
                    "upland regolith thickness for land-cover class 1; sedimentary "
                    "deposit thickness for land-cover class 2"
                ),
                "resampling": "bilinear to analysis grid; land-cover class nearest-neighbor",
            },
        )
        land_cover_path = output_dir / "pelletier2016_land_cover_class.tif"
        _write_raster(
            land_cover_path,
            pelletier_land_cover,
            crs=crs,
            transform=transform,
            description="pelletier2016_land_cover_class",
            tags={
                "source": "Pelletier et al. (2016), ORNL DAAC 1304",
                "classes": "0 ocean; 1 upland; 2 lowland; 3 lake; 4 perennial ice",
            },
        )
    else:
        depth_path = shangguan_depth_path
        land_cover_path = None
    conductivity_path = output_dir / "hydraulic_conductivity_glhymps2_mday.tif"
    _write_raster(
        conductivity_path,
        hydraulic_conductivity_m_day,
        crs=crs,
        transform=transform,
        description="glhymps2_hydraulic_conductivity_m_day",
        tags={
            "source": "GLHYMPS 2.0",
            "source_doi": GLHYMPS2_DOI,
            "conversion": "10^(logK_Ferr_/100) * 1e7 m/s * 86400 s/day",
        },
    )
    transmissivity_path = output_dir / (
        "transmissivity_glhymps2_pelletier2016_m2day.tif"
        if use_pelletier
        else "transmissivity_glhymps2_shangguan_m2day.tif"
    )
    _write_raster(
        transmissivity_path,
        transmissivity_m2_day,
        crs=crs,
        transform=transform,
        description="glhymps2_transmissivity_m2_day",
        tags={
            "source": (
                "GLHYMPS 2.0 permeability x Pelletier 2016 landform thickness"
                if use_pelletier
                else "GLHYMPS 2.0 permeability x Shangguan 2017 depth"
            ),
            "source_doi": (
                f"{GLHYMPS2_DOI}; {PELLETIER2016_DOI}"
                if use_pelletier
                else f"{GLHYMPS2_DOI}; {SHANGGUAN_DTB_DOI}"
            ),
            "conversion": "hydraulic_conductivity_m_day * modeled_aquifer_thickness_m",
        },
    )

    cell_area_m2 = abs(transform.a * transform.e)
    hydrogeology_comparison: dict[str, object] = {
        "official_specific_yield_proxy": _field_summary(official, inside),
        "official_hydraulic_conductivity_m_day": _field_summary(
            hydraulic_conductivity_m_day, inside
        ),
    }
    existing = None
    comparison_path = None
    plot_path = None
    summary = None
    if existing_porosity_raster is not None:
        existing = _read_aligned_raster(
            existing_porosity_raster,
            shape=shape,
            transform=transform,
            crs=crs,
            resampling=rasterio.enums.Resampling.nearest,
        )
        comparison, summary = _comparison_summary(
            official, existing, inside, cell_area_m2
        )
        comparison_path = output_dir / "porosity_value_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        plot_path = output_dir / "porosity_comparison.png"
        _save_comparison_plot(plot_path, official, existing, inside, output_bounds)
        hydrogeology_comparison["specific_yield_proxy_comparison"] = (
            _paired_field_summary(official, existing, inside)
        )
    if use_pelletier:
        class_values, class_counts = np.unique(
            pelletier_land_cover[inside].astype(int), return_counts=True
        )
        hydrogeology_comparison["pelletier2016"] = {
            "modeled_thickness_m": _field_summary(depth_m, inside),
            "shangguan_depth_m": (
                _field_summary(shangguan_depth_m, inside)
                if shangguan_depth_m is not None
                else None
            ),
            "modeled_divided_by_shangguan": (
                _paired_field_summary(depth_m, shangguan_depth_m, inside)[
                    "official_divided_by_legacy"
                ]
                if shangguan_depth_m is not None
                else None
            ),
            "land_cover_area_km2": {
                str(int(value)): float(count * cell_area_m2 / 1e6)
                for value, count in zip(class_values, class_counts)
            },
            "land_cover_classes": {
                "1": "upland: use regolith thickness",
                "2": "lowland: use sedimentary-deposit thickness",
            },
        }
    hydro_plot_path = output_dir / "hydrogeology_comparison.png"
    legacy_depth = None
    if existing_depth_raster is not None:
        legacy_depth = _read_aligned_raster(
            existing_depth_raster,
            shape=shape,
            transform=transform,
            crs=crs,
        )
        hydrogeology_comparison["depth_to_bedrock_m"] = _paired_field_summary(
            depth_m, legacy_depth, inside
        )
    legacy_transmissivity = None
    if existing_transmissivity_raster is not None:
        legacy_transmissivity = _read_aligned_raster(
            existing_transmissivity_raster,
            shape=shape,
            transform=transform,
            crs=crs,
        )
        hydrogeology_comparison["transmissivity_m2_day"] = _paired_field_summary(
            transmissivity_m2_day, legacy_transmissivity, inside
        )
    if (
        existing is not None
        and legacy_depth is not None
        and legacy_transmissivity is not None
    ):
        _save_hydrogeology_plot(
            hydro_plot_path,
            official_fields={
                "porosity": official,
                "depth": depth_m,
                "transmissivity": transmissivity_m2_day,
            },
            legacy_fields={
                "porosity": existing,
                "depth": legacy_depth,
                "transmissivity": legacy_transmissivity,
            },
            inside=inside,
            extent=output_bounds,
        )

    class_area = (
        pd.Series(official[inside], name="porosity_fraction")
        .value_counts()
        .rename_axis("porosity_fraction")
        .rename("cell_count")
        .reset_index()
    )
    class_area["area_km2"] = class_area["cell_count"] * cell_area_m2 / 1e6
    class_path = output_dir / "glhymps2_porosity_classes.csv"
    class_area.to_csv(class_path, index=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "GLHYMPS 2.0",
            "doi": GLHYMPS2_DOI,
            "release": GLHYMPS2_RELEASE,
            "borealis_datafile_id": GLHYMPS2_DATAFILE_ID,
            "archive": str(zip_path.resolve()),
            "archive_sha256": file_sha256(zip_path),
            "porosity_field": "Porosity_x",
            "conversion": "porosity_fraction = Porosity_x / 100",
            "permeability_field": "logK_Ferr_",
            "permeability_conversion": "log10_permeability_m2 = logK_Ferr_ / 100",
            "hydraulic_conductivity_conversion": (
                "K_m_s = 10^(logK_Ferr_/100) * 1e7, following the GLHYMPS readme"
            ),
        },
        "depth_to_bedrock_source": {
            "primary": (
                {
                    "name": "Pelletier 2016 landform-specific permeable-layer thickness",
                    "doi": PELLETIER2016_DOI,
                    "publication_doi": PELLETIER2016_PUBLICATION_DOI,
                    "release": PELLETIER2016_RELEASE,
                    "regolith_raster": str(Path(pelletier_regolith_raster).resolve()),
                    "regolith_sha256": file_sha256(pelletier_regolith_raster),
                    "sediment_raster": str(Path(pelletier_sediment_raster).resolve()),
                    "sediment_sha256": file_sha256(pelletier_sediment_raster),
                    "land_cover_raster": str(Path(pelletier_land_cover_raster).resolve()),
                    "land_cover_sha256": file_sha256(pelletier_land_cover_raster),
                    "native_resolution": "30 arc-second (~1 km)",
                    "mosaic_rule": (
                        "land-cover class 1 uses upland regolith thickness; class 2 "
                        "uses valley-bottom/lowland sedimentary-deposit thickness"
                    ),
                    "warning": (
                        "The upland regolith layer is an experimental product with a "
                        "high degree of uncertainty, according to the source documentation."
                    ),
                }
                if use_pelletier
                else {
                    "name": "SoilGrids250m BDTICM_M absolute depth to bedrock",
                    "doi": SHANGGUAN_DTB_DOI,
                    "url": str(depth_source) if depth_source is not None else None,
                }
            ),
            "alternate_shangguan2017": (
                {
                    "name": "SoilGrids250m BDTICM_M absolute depth to bedrock",
                    "doi": SHANGGUAN_DTB_DOI,
                    "url": str(depth_source),
                    "local_source_sha256": (
                        file_sha256(depth_source)
                        if Path(depth_source).exists()
                        else None
                    ),
                    "publication_date": depth_source_tags.get("PUBLICATION_DATE"),
                    "attribute_title": depth_source_tags.get("ATTRIBUTE_TITLE"),
                    "source_units": depth_source_tags.get("ATTRIBUTE_UNITS_OF_MEASURE"),
                    "conversion": "centimeters / 100 = meters",
                }
                if depth_source is not None
                else None
            ),
        },
        "interpretation": (
            "GLHYMPS total porosity is used directly as an effective specific-yield "
            "proxy for this screening-level model; it is not a measured drainable porosity."
        ),
        "processing": {
            "buffer_m": buffer_m,
            "reference_raster": str(Path(reference_raster).resolve()),
            "existing_porosity_raster": (
                str(Path(existing_porosity_raster).resolve())
                if existing_porosity_raster is not None
                else None
            ),
            "existing_depth_raster": (
                str(Path(existing_depth_raster).resolve())
                if existing_depth_raster is not None
                else None
            ),
            "existing_transmissivity_raster": (
                str(Path(existing_transmissivity_raster).resolve())
                if existing_transmissivity_raster is not None
                else None
            ),
            "rasterization": "pixel-center categorical rasterization; no focal smoothing",
            "crs": str(crs),
            "resolution_m": [abs(transform.a), abs(transform.e)],
        },
        "comparison": summary,
        "hydrogeology_comparison": hydrogeology_comparison,
        "artifacts": {
            "source_clip": str(clip_path.resolve()),
            "specific_yield_proxy": str(porosity_path.resolve()),
            "depth_to_bedrock": str(depth_path.resolve()),
            "alternate_shangguan_depth_to_bedrock": (
                str(shangguan_depth_path.resolve())
                if shangguan_depth_path is not None
                else None
            ),
            "pelletier_land_cover": (
                str(land_cover_path.resolve()) if land_cover_path is not None else None
            ),
            "hydraulic_conductivity": str(conductivity_path.resolve()),
            "transmissivity": str(transmissivity_path.resolve()),
            "class_areas": str(class_path.resolve()),
            "value_comparison": (
                str(comparison_path.resolve()) if comparison_path is not None else None
            ),
            "comparison_plot": (
                str(plot_path.resolve()) if plot_path is not None else None
            ),
            "hydrogeology_comparison_plot": (
                str(hydro_plot_path.resolve()) if hydro_plot_path.exists() else None
            ),
        },
    }
    metadata_path = output_dir / "glhymps2_provenance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "porosity": porosity_path,
        "depth_to_bedrock": depth_path,
        "alternate_shangguan_depth": shangguan_depth_path,
        "pelletier_land_cover": land_cover_path,
        "hydraulic_conductivity": conductivity_path,
        "transmissivity": transmissivity_path,
        "source_clip": clip_path,
        "comparison": comparison_path,
        "plot": plot_path,
        "hydrogeology_plot": hydro_plot_path if hydro_plot_path.exists() else None,
        "metadata": metadata_path,
    }
