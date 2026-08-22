from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from landlab import RasterModelGrid
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from tqdm import tqdm

from .recharge import (
    RECHARGE_RASTER_UNIT_TO_MM_DAY,
    load_recharge_raster_manifest,
)
from .stream_limiter import (
    RoutedStreamLimitedGroundwaterDupuitPercolator,
    route_streamflow_with_availability,
)


SECONDS_PER_DAY = 86400.0
REACH_OUTPUT_SCHEMA_VERSION = "2.0.0"
REACH_DEFINITION_VERSION = "d8-maximal-stream-chain-v1"
HYDROGRAPH_DISPLAY_FLOOR_M3D = 1.0


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
    additional_aquifer_depth: float = 0.0
    deep_aquifer_hydraulics: str = "preserve_transmissivity"
    well_additional_depth: float = 0.0
    pumping_source_mode: str = "well_cell"
    pumping_source_area_threshold: float = 500000.0
    specific_yield_floor: float = 0.0
    source_zone_storage_fraction: float = 0.5
    pumping_storage_fraction: float = 0.5
    strict_pumping_supply: bool = False
    stream_loss_mode: str = "routed_volume_limited"
    stream_limiter_tolerance_m3: float = 1.0e-6
    stream_limiter_max_iterations: int = 25


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
    outlet_node: int
    stream_indices: np.ndarray
    active_nodes: np.ndarray
    cell_area: float
    basin_area_m2: float
    stream_area_threshold: float = 0.0


@dataclass(frozen=True)
class DailyRechargeForcing:
    """Validated basin-mean or spatial daily recharge input."""

    frame: pd.DataFrame
    source_type: str
    source_path: Path

    @property
    def is_spatial(self) -> bool:
        return self.source_type == "raster_manifest"

    def value_for_record(self, record, state: GroundwaterState) -> float | np.ndarray:
        if not self.is_spatial:
            return float(record.Recharge)
        return load_recharge_raster_field(
            state,
            Path(record.raster_path),
            band=int(record.band),
            units=str(record.units),
        )


@dataclass(frozen=True)
class ReachNetwork:
    """Static, disjoint reach network used for incremental spatial accounting."""

    reach_at_stream_node: np.ndarray
    reach_at_core_node: np.ndarray
    stream_nodes_by_reach: tuple[np.ndarray, ...]
    downstream_reach: np.ndarray
    boundary_links: np.ndarray
    boundary_link_directions: np.ndarray
    boundary_link_reaches: np.ndarray
    incremental_area_m2: np.ndarray
    upstream_area_m2: np.ndarray
    reach_length_m: np.ndarray
    definition_sha256: str

    @property
    def number_of_reaches(self) -> int:
        return len(self.stream_nodes_by_reach)


def build_reach_network(state: GroundwaterState) -> ReachNetwork:
    """Build maximal D8 stream chains and their disjoint incremental catchments.

    A new reach begins at each headwater or confluence. Every core aquifer cell is
    assigned to the reach containing its first downstream fixed-head stream node.
    The assignment is topographic and is used only to locate modeled streamflow
    contributions; it is not an attribution of pumping sources to reaches.
    """
    grid = state.grid
    required_fields = {"flow__receiver_node", "drainage_area"}
    missing = required_fields.difference(grid.at_node)
    if missing:
        raise ValueError(
            "Reach accounting requires D8 routing fields: " + ", ".join(sorted(missing))
        )

    receiver = np.asarray(grid.at_node["flow__receiver_node"], dtype=int)
    stream_nodes = np.unique(np.asarray(state.stream_indices, dtype=int))
    if stream_nodes.size == 0:
        raise ValueError("Cannot build a reach network without stream nodes.")
    is_stream = np.zeros(grid.number_of_nodes, dtype=bool)
    is_stream[stream_nodes] = True

    donor_count = np.zeros(grid.number_of_nodes, dtype=int)
    downstream = receiver[stream_nodes]
    internal = is_stream[downstream] & (downstream != stream_nodes)
    np.add.at(donor_count, downstream[internal], 1)

    # Ordering by contributing area and node number makes reach IDs stable and
    # generally proceeds from headwaters toward the outlet.
    starts = stream_nodes[donor_count[stream_nodes] != 1]
    starts = np.asarray(
        sorted(
            starts.tolist(),
            key=lambda node: (float(grid.at_node["drainage_area"][node]), int(node)),
        ),
        dtype=int,
    )
    reach_at_stream = np.full(grid.number_of_nodes, -1, dtype=np.int32)
    chains: list[np.ndarray] = []
    for start in starts:
        chain: list[int] = []
        node = int(start)
        seen: set[int] = set()
        while True:
            if node in seen:
                raise ValueError(f"Cycle detected in D8 stream routing at node {node}.")
            if reach_at_stream[node] >= 0:
                raise ValueError(f"Stream node {node} was assigned to more than one reach.")
            seen.add(node)
            chain.append(node)
            next_node = int(receiver[node])
            if (
                next_node == node
                or not is_stream[next_node]
                or donor_count[next_node] != 1
            ):
                break
            node = next_node
        reach_index = len(chains)
        chain_array = np.asarray(chain, dtype=int)
        reach_at_stream[chain_array] = reach_index
        chains.append(chain_array)

    unassigned_streams = stream_nodes[reach_at_stream[stream_nodes] < 0]
    if unassigned_streams.size:
        preview = ", ".join(str(node) for node in unassigned_streams[:5])
        raise ValueError(
            f"Reach construction left {unassigned_streams.size} stream node(s) "
            f"unassigned, including {preview}."
        )
    n_reaches = len(chains)

    downstream_reach = np.full(n_reaches, -1, dtype=np.int32)
    for reach_index, chain in enumerate(chains):
        next_node = int(receiver[int(chain[-1])])
        if next_node != int(chain[-1]) and is_stream[next_node]:
            next_reach = int(reach_at_stream[next_node])
            if next_reach == reach_index:
                raise ValueError(f"Reach {reach_index + 1} routes to itself.")
            downstream_reach[reach_index] = next_reach

    # Assign each core node by following D8 routing to its first stream node.
    reach_at_core = np.full(grid.number_of_nodes, -1, dtype=np.int32)
    routed_reach = reach_at_stream.copy()
    for start in np.asarray(grid.core_nodes, dtype=int):
        node = int(start)
        path: list[int] = []
        seen: set[int] = set()
        while routed_reach[node] < 0:
            if node in seen:
                raise ValueError(f"Cycle detected while routing core node {start}.")
            seen.add(node)
            path.append(node)
            next_node = int(receiver[node])
            if next_node == node:
                raise ValueError(
                    f"Core node {start} does not drain to the extracted stream network."
                )
            node = next_node
        destination = int(routed_reach[node])
        reach_at_core[np.asarray(path, dtype=int)] = destination
        routed_reach[np.asarray(path, dtype=int)] = destination
    reach_at_core[grid.core_nodes] = routed_reach[grid.core_nodes]
    if np.any(reach_at_core[grid.core_nodes] < 0):
        raise ValueError("Every core aquifer cell must be assigned to exactly one reach.")

    incremental_area = np.bincount(
        reach_at_core[grid.core_nodes],
        weights=grid.cell_area_at_node[grid.core_nodes],
        minlength=n_reaches,
    ).astype(float)
    if not np.isclose(
        float(np.sum(incremental_area)),
        float(np.sum(grid.cell_area_at_node[grid.core_nodes])),
    ):
        raise ValueError("Incremental reach areas do not sum to the modeled aquifer area.")

    # Accumulate upstream areas on the reach graph. The recursion also verifies
    # that the graph is acyclic and drains to a single outlet reach.
    donors_by_reach: list[list[int]] = [[] for _ in range(n_reaches)]
    for reach_index, next_reach in enumerate(downstream_reach):
        if next_reach >= 0:
            donors_by_reach[int(next_reach)].append(reach_index)
    outlet_reaches = np.flatnonzero(downstream_reach < 0)
    if outlet_reaches.size != 1:
        raise ValueError(
            f"Reach graph must have one outlet; found {outlet_reaches.size}."
        )
    upstream_area = np.full(n_reaches, np.nan, dtype=float)
    visiting: set[int] = set()

    def area_above(reach_index: int) -> float:
        if np.isfinite(upstream_area[reach_index]):
            return float(upstream_area[reach_index])
        if reach_index in visiting:
            raise ValueError("Cycle detected in the reach graph.")
        visiting.add(reach_index)
        value = incremental_area[reach_index] + sum(
            area_above(donor) for donor in donors_by_reach[reach_index]
        )
        visiting.remove(reach_index)
        upstream_area[reach_index] = value
        return float(value)

    for reach_index in range(n_reaches):
        area_above(reach_index)

    reach_length = np.zeros(n_reaches, dtype=float)
    for reach_index, chain in enumerate(chains):
        geometry_nodes = chain
        next_reach = int(downstream_reach[reach_index])
        if next_reach >= 0:
            geometry_nodes = np.append(geometry_nodes, chains[next_reach][0])
        if geometry_nodes.size >= 2:
            dx = np.diff(grid.x_of_node[geometry_nodes])
            dy = np.diff(grid.y_of_node[geometry_nodes])
            reach_length[reach_index] = float(np.sum(np.hypot(dx, dy)))

    open_nodes = np.asarray(grid.open_boundary_nodes, dtype=int)
    links_at_open = grid.links_at_node[open_nodes]
    directions_at_open = grid.active_link_dirs_at_node[open_nodes]
    active_links = directions_at_open != 0
    boundary_links = np.asarray(links_at_open[active_links], dtype=int)
    boundary_directions = np.asarray(directions_at_open[active_links], dtype=float)
    boundary_nodes = np.broadcast_to(
        open_nodes[:, np.newaxis], links_at_open.shape
    )[active_links]
    boundary_reaches = reach_at_stream[boundary_nodes]
    if np.any(boundary_reaches < 0):
        raise ValueError("Every open groundwater boundary node must belong to a reach.")
    if np.unique(boundary_links).size != boundary_links.size:
        raise ValueError("An active groundwater boundary link was assigned more than once.")

    digest = hashlib.sha256()
    for values in (
        reach_at_stream[stream_nodes],
        reach_at_core[grid.core_nodes],
        downstream_reach,
        boundary_links,
        boundary_reaches,
    ):
        digest.update(np.asarray(values, dtype="<i8").tobytes())

    return ReachNetwork(
        reach_at_stream_node=reach_at_stream,
        reach_at_core_node=reach_at_core,
        stream_nodes_by_reach=tuple(chains),
        downstream_reach=downstream_reach,
        boundary_links=boundary_links,
        boundary_link_directions=boundary_directions,
        boundary_link_reaches=np.asarray(boundary_reaches, dtype=np.int32),
        incremental_area_m2=incremental_area,
        upstream_area_m2=upstream_area,
        reach_length_m=reach_length,
        definition_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True)
class PumpingForcing:
    """Spatial pumping fluxes keyed by calendar month or year-month."""

    mode: str
    flux_fields: dict[int | pd.Period, np.ndarray]
    zero_flux: np.ndarray
    coverage_start: pd.Period | None = None
    coverage_end: pd.Period | None = None
    well_nodes: np.ndarray | None = None
    source_mode: str = "well_cell"
    source_zones: dict[int, np.ndarray] | None = None
    zone_demands_m3d: dict[int | pd.Period, dict[int, float]] | None = None
    source_nodes: np.ndarray | None = None
    source_area_threshold_m2: float | None = None

    def _point_flux_for_date(
        self,
        date: str | pd.Timestamp,
    ) -> tuple[pd.Timestamp, int | pd.Period, np.ndarray]:
        date = pd.Timestamp(date)
        key: int | pd.Period = (
            date.month if self.mode == "climatology" else date.to_period("M")
        )
        try:
            point_flux = self.flux_fields[key]
        except KeyError as exc:
            if (
                self.mode == "timeseries"
                and self.coverage_start is not None
                and self.coverage_end is not None
                and (key < self.coverage_start or key > self.coverage_end)
            ):
                return date, key, self.zero_flux
            raise ValueError(
                f"No pumping data are available for {date:%Y-%m} in {self.mode!r} mode."
            ) from exc
        return date, key, point_flux

    def flux_for_date(
        self,
        date: str | pd.Timestamp,
        *,
        state: GroundwaterState | None = None,
        source_zone_storage_fraction: float = 0.5,
        strict_pumping_supply: bool = True,
    ) -> np.ndarray:
        date, key, point_flux = self._point_flux_for_date(date)
        if self.source_mode == "well_cell":
            return point_flux
        if self.source_mode != "topographic":
            raise ValueError(f"Unknown pumping source mode: {self.source_mode}")
        if state is None:
            raise ValueError("Topographic pumping allocation requires groundwater state.")
        if self.source_zones is None or self.zone_demands_m3d is None:
            raise ValueError("Topographic pumping forcing has no source-zone data.")
        try:
            return _allocate_topographic_pumping(
                state,
                self.source_zones,
                self.zone_demands_m3d.get(key, {}),
                source_zone_storage_fraction,
                strict_pumping_supply,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cannot allocate pumping for {date:%Y-%m-%d}: {exc}"
            ) from exc

    def scheduled_volume_for_date(
        self,
        date: str | pd.Timestamp,
        state: GroundwaterState,
    ) -> float:
        """Return the full reported demand before source-capacity clipping."""
        _, key, point_flux = self._point_flux_for_date(date)
        if self.source_mode == "topographic" and self.zone_demands_m3d is not None:
            return float(sum(self.zone_demands_m3d.get(key, {}).values()))
        return float(
            -np.sum(point_flux * state.grid.cell_area_at_node) * SECONDS_PER_DAY
        )


def _landlab_node_values(values) -> np.ndarray:
    """Convert north-up raster rows to Landlab's lower-left node ordering."""
    return np.flipud(np.asarray(values)).ravel()


def _load_and_align(path: Path, match_da, method=Resampling.average):
    da = rioxarray.open_rasterio(path, masked=True).squeeze()
    return da.rio.reproject_match(match_da, resampling=method)


def load_recharge_raster_field(
    state: GroundwaterState,
    path: Path,
    *,
    band: int = 1,
    units: str = "mm/day",
) -> np.ndarray:
    """Align one recharge raster band to the model and return node rates in mm/day."""
    if state.dem_coarse is None or not hasattr(state.dem_coarse, "rio"):
        raise ValueError("Spatial recharge requires a georeferenced model grid.")
    source = rioxarray.open_rasterio(path, masked=True)
    try:
        if source.rio.crs is None:
            raise ValueError(f"Recharge raster has no CRS: {path}")
        band_count = int(source.sizes.get("band", 1))
        if band < 1 or band > band_count:
            raise ValueError(
                f"Recharge raster band {band} is outside 1..{band_count}: {path}"
            )
        layer = source.isel(band=band - 1, drop=True)
        aligned = layer.rio.reproject_match(
            state.dem_coarse,
            resampling=Resampling.average,
        )
        raster_values = np.asarray(aligned.values, dtype=float)
    finally:
        source.close()

    if raster_values.shape != state.dem_coarse.shape:
        raise ValueError(
            f"Aligned recharge raster shape {raster_values.shape} does not match "
            f"the model grid {state.dem_coarse.shape}."
        )
    try:
        factor = RECHARGE_RASTER_UNIT_TO_MM_DAY[units]
    except KeyError as exc:
        raise ValueError(f"Unsupported normalized recharge raster unit: {units}") from exc
    node_values = _landlab_node_values(raster_values) * factor
    core_values = node_values[state.grid.core_nodes]
    if np.any(~np.isfinite(core_values)):
        raise ValueError(
            f"Recharge raster has missing values at modeled aquifer nodes: {path}, band {band}."
        )
    if np.any(core_values < 0.0):
        raise ValueError(
            f"Recharge raster has negative values at modeled aquifer nodes: {path}, band {band}."
        )
    return node_values


def _inside_boundary_nodes(grid: RasterModelGrid, inside: np.ndarray) -> np.ndarray:
    """Return inside nodes that touch the exterior of a rasterized basin."""
    neighbors = np.column_stack(
        (grid.adjacent_nodes_at_node, grid.diagonal_adjacent_nodes_at_node)
    )
    boundary_nodes = []
    for node in np.flatnonzero(inside):
        valid_neighbors = neighbors[node][neighbors[node] != grid.BAD_INDEX]
        if valid_neighbors.size == 0 or np.any(~inside[valid_neighbors]):
            boundary_nodes.append(node)
    return np.asarray(boundary_nodes, dtype=int)


def _validate_aquifer_field(
    values: np.ndarray,
    nodes: np.ndarray,
    name: str,
    *,
    lower: float,
    upper: float | None = None,
) -> None:
    selected = values[nodes]
    invalid = ~np.isfinite(selected) | (selected <= lower)
    if upper is not None:
        invalid |= selected > upper
    if np.any(invalid):
        bounds = f"> {lower}" if upper is None else f"> {lower} and <= {upper}"
        raise ValueError(
            f"{name} must be finite and {bounds} at every modeled node; "
            f"found {int(np.sum(invalid))} invalid values."
        )


def build_groundwater_state(
    inputs: GroundwaterInputs,
    config: GroundwaterConfig = GroundwaterConfig(),
) -> GroundwaterState:
    """Build the Landlab grid and static aquifer fields used by all simulations."""
    if config.target_resolution <= 0.0:
        raise ValueError("Target resolution must be positive.")
    if config.stream_area_threshold < 0.0:
        raise ValueError("Stream area threshold cannot be negative.")
    if config.courant_coefficient <= 0.0:
        raise ValueError("Courant coefficient must be positive.")
    if config.stream_drain_offset < 0.0:
        raise ValueError("Stream drain offset cannot be negative.")
    if config.additional_aquifer_depth < 0.0:
        raise ValueError("Additional aquifer depth cannot be negative.")
    if config.well_additional_depth < 0.0:
        raise ValueError("Additional well depth cannot be negative.")
    if config.pumping_source_mode not in {"well_cell", "topographic"}:
        raise ValueError("Pumping source mode must be 'well_cell' or 'topographic'.")
    if (
        config.pumping_source_mode == "topographic"
        and config.pumping_source_area_threshold < config.stream_area_threshold
    ):
        raise ValueError(
            "Pumping source area threshold must be at least the stream area threshold."
        )
    if not 0.0 <= config.specific_yield_floor <= 1.0:
        raise ValueError("Specific-yield floor must be between 0 and 1.")
    if not 0.0 < config.source_zone_storage_fraction <= 1.0:
        raise ValueError(
            "Source-zone storage fraction must be greater than 0 and at most 1."
        )
    if config.deep_aquifer_hydraulics not in {
        "preserve_transmissivity",
        "preserve_conductivity",
    }:
        raise ValueError(
            "Deep-aquifer hydraulics must preserve transmissivity or conductivity."
        )
    if not 0.0 < config.pumping_storage_fraction <= 1.0:
        raise ValueError("Pumping storage fraction must be greater than 0 and at most 1.")
    if config.stream_loss_mode not in {
        "unlimited_fixed_head",
        "routed_volume_limited",
    }:
        raise ValueError(
            "Stream loss mode must be 'unlimited_fixed_head' or "
            "'routed_volume_limited'."
        )
    if config.stream_limiter_tolerance_m3 < 0.0:
        raise ValueError("Stream-limiter tolerance cannot be negative.")
    if config.stream_limiter_max_iterations < 1:
        raise ValueError("Stream-limiter maximum iterations must be positive.")

    FlowAccumulator, _, SinkFillerBarnes = _landlab_components()

    dem_raw = rioxarray.open_rasterio(inputs.dem_path, masked=True).squeeze()
    dem_utm = dem_raw.rio.reproject(config.target_crs)
    bounds = dem_utm.rio.bounds()
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    new_width = int(width_m / config.target_resolution)
    new_height = int(height_m / config.target_resolution)
    if new_width < 3 or new_height < 3:
        raise ValueError("The target resolution produces a grid smaller than 3 by 3 nodes.")
    dem_coarse = dem_utm.rio.reproject(
        config.target_crs,
        shape=(new_height, new_width),
        resampling=Resampling.average,
    )

    dy = abs(dem_coarse.rio.resolution()[1])
    dx = abs(dem_coarse.rio.resolution()[0])
    origin_x = float(np.min(dem_coarse.x.values))
    origin_y = float(np.min(dem_coarse.y.values))
    grid = RasterModelGrid(
        dem_coarse.shape,
        xy_spacing=(dx, dy),
        xy_of_lower_left=(origin_x, origin_y),
        xy_axis_units="m",
    )

    z = grid.add_zeros("topographic__elevation", at="node")
    dem_vals = _landlab_node_values(dem_coarse.values)

    basin_gdf = gpd.read_file(inputs.basin_path)
    if basin_gdf.crs is None:
        raise ValueError(f"Basin boundary has no CRS: {inputs.basin_path}")
    if basin_gdf.crs != dem_coarse.rio.crs:
        basin_gdf = basin_gdf.to_crs(dem_coarse.rio.crs)
    is_inside_basin = geometry_mask(
        basin_gdf.geometry,
        out_shape=dem_coarse.shape,
        transform=dem_coarse.rio.transform(),
        invert=True,
    )
    basin_nodes = _landlab_node_values(is_inside_basin).astype(bool)
    if not np.any(basin_nodes):
        raise ValueError("The basin boundary does not overlap the groundwater grid.")
    if np.any(~np.isfinite(dem_vals[basin_nodes])):
        raise ValueError("The DEM contains missing elevations inside the basin boundary.")

    outside_fill = float(np.nanmax(dem_vals[basin_nodes]) + np.ptp(dem_vals[basin_nodes]) + 1.0)
    # Keep every closed, out-of-basin node above the modeled watershed. Some
    # source DEM pixels beyond an irregular polygon are valid but lower than
    # adjacent in-basin cells; leaving those elevations in place can make the
    # depression filler treat the closed boundary as a false drainage outlet.
    z[:] = np.where(basin_nodes, dem_vals, outside_fill)

    # Route the rasterized watershed to one topographic outlet before extracting
    # channels. This prevents NoData cells outside the polygon from attracting flow.
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
    grid.status_at_node[~basin_nodes] = grid.BC_NODE_IS_CLOSED
    boundary_nodes = _inside_boundary_nodes(grid, basin_nodes)
    if boundary_nodes.size == 0:
        raise ValueError("Could not identify a rasterized basin boundary.")
    outlet_node = int(boundary_nodes[np.argmin(z[boundary_nodes])])
    grid.status_at_node[outlet_node] = grid.BC_NODE_IS_FIXED_VALUE

    # Add a minimal gradient across filled flats so modeled cells have a
    # deterministic D8 path toward the selected outlet.
    SinkFillerBarnes(grid, method="D8", fill_flat=False).run_one_step()
    FlowAccumulator(grid, flow_director="FlowDirectorD8").run_one_step()
    is_stream = grid.at_node["drainage_area"] >= config.stream_area_threshold
    is_stream[outlet_node] = True
    stream_indices = np.where(is_stream & basin_nodes)[0]

    grid.status_at_node[stream_indices] = grid.BC_NODE_IS_FIXED_VALUE
    if grid.core_nodes.size == 0:
        raise ValueError(
            "The stream threshold leaves no active aquifer nodes in the basin."
        )

    trans_aligned = _load_and_align(inputs.transmissivity_path, dem_coarse)
    dtb_aligned = _load_and_align(inputs.depth_to_bedrock_path, dem_coarse)
    poro_aligned = _load_and_align(inputs.porosity_path, dem_coarse, method=Resampling.nearest)

    modeled_nodes = np.flatnonzero(grid.status_at_node != grid.BC_NODE_IS_CLOSED)
    depth_raw = _landlab_node_values(dtb_aligned.values).astype(float)
    trans_raw = _landlab_node_values(trans_aligned.values).astype(float)
    porosity = _landlab_node_values(poro_aligned.values).astype(float)
    _validate_aquifer_field(depth_raw, modeled_nodes, "Depth to bedrock", lower=0.0)
    _validate_aquifer_field(trans_raw, modeled_nodes, "Transmissivity", lower=0.0)
    _validate_aquifer_field(porosity, modeled_nodes, "Drainable porosity", lower=0.0, upper=1.0)

    depth_raw[~np.isfinite(depth_raw)] = 0.1
    database_depth = np.maximum(depth_raw, 0.1)
    modeled_depth = database_depth + config.additional_aquifer_depth
    trans_raw[~np.isfinite(trans_raw)] = 0.0
    transmissivity_sec = trans_raw / SECONDS_PER_DAY
    if config.deep_aquifer_hydraulics == "preserve_transmissivity":
        k_nodes = transmissivity_sec / modeled_depth
    else:
        k_nodes = transmissivity_sec / database_depth
    grid.add_field(
        "aquifer__hydraulic_conductivity",
        k_nodes,
        at="node",
        clobber=True,
    )
    grid.add_field(
        "aquifer__transmissivity",
        transmissivity_sec,
        at="node",
        clobber=True,
    )
    grid.add_field(
        "hydraulic_conductivity",
        grid.map_mean_of_link_nodes_to_link(k_nodes),
        at="link",
        clobber=True,
    )

    grid.add_field(
        "database_aquifer__thickness",
        database_depth,
        at="node",
        clobber=True,
    )
    grid.add_field(
        "modeled_aquifer__thickness",
        modeled_depth,
        at="node",
        clobber=True,
    )

    base_elev = z - modeled_depth
    drain_elev = z[stream_indices] - config.stream_drain_offset
    base_elev[stream_indices] = np.minimum(base_elev[stream_indices], drain_elev)
    grid.add_field("aquifer_base__elevation", base_elev, at="node", clobber=True)

    porosity[~np.isfinite(porosity)] = 0.2
    grid.add_field(
        "database_aquifer__porosity",
        porosity.copy(),
        at="node",
        clobber=True,
    )
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
        outlet_node=outlet_node,
        stream_indices=stream_indices,
        active_nodes=active_nodes,
        cell_area=cell_area,
        basin_area_m2=basin_area_m2,
        stream_area_threshold=config.stream_area_threshold,
    )


def apply_well_aquifer_depth(
    state: GroundwaterState,
    well_nodes: np.ndarray,
    additional_depth: float,
) -> np.ndarray:
    """Extend only mapped pumping cells while preserving link conductivity."""
    if additional_depth < 0.0:
        raise ValueError("Additional well depth cannot be negative.")

    grid = state.grid
    nodes = np.unique(np.asarray(well_nodes, dtype=int))
    if nodes.size == 0:
        raise ValueError("Cannot add well depth because no pumping nodes were mapped.")
    if np.any((nodes < 0) | (nodes >= grid.number_of_nodes)):
        raise ValueError("Pumping nodes contain indices outside the groundwater grid.")
    if np.any(grid.status_at_node[nodes] != grid.BC_NODE_IS_CORE):
        raise ValueError("Additional well depth can only be applied at core aquifer nodes.")

    if "well_aquifer__additional_depth" in grid.at_node:
        existing = grid.at_node["well_aquifer__additional_depth"]
        if np.any(existing != 0.0):
            raise RuntimeError("Additional well depth has already been applied to this state.")
    else:
        existing = grid.add_zeros("well_aquifer__additional_depth", at="node")

    if additional_depth == 0.0:
        return nodes

    conductivity_before = grid.at_link["hydraulic_conductivity"].copy()
    existing[nodes] = additional_depth
    grid.at_node["modeled_aquifer__thickness"][nodes] += additional_depth
    grid.at_node["aquifer_base__elevation"][nodes] -= additional_depth

    if not np.array_equal(
        grid.at_link["hydraulic_conductivity"], conductivity_before
    ):
        raise RuntimeError("Applying well depth unexpectedly changed conductivity.")
    return nodes


def _topographic_source_zones(
    state: GroundwaterState,
    well_nodes: np.ndarray,
    source_area_threshold: float,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Group core nodes by disjoint reaches on a coarser source network."""
    grid = state.grid
    receiver = np.asarray(grid.at_node["flow__receiver_node"], dtype=int)
    if source_area_threshold < state.stream_area_threshold:
        raise ValueError(
            "Pumping source area threshold must be at least the stream area threshold."
        )
    modeled = grid.status_at_node != grid.BC_NODE_IS_CLOSED
    stream_nodes = set(
        np.flatnonzero(
            (grid.at_node["drainage_area"] >= source_area_threshold) & modeled
        ).tolist()
    )
    stream_nodes.add(int(state.outlet_node))

    # A reach extends from one source-network junction to the next downstream
    # junction. The coarser network groups neighboring hillslopes while retaining
    # disjoint, drainage-bounded source zones.
    stream_donor_count: dict[int, int] = {}
    for node in stream_nodes:
        downstream = int(receiver[node])
        if downstream in stream_nodes and downstream != node:
            stream_donor_count[downstream] = (
                stream_donor_count.get(downstream, 0) + 1
            )
    junctions = {
        node for node, count in stream_donor_count.items() if count >= 2
    }
    junctions.add(int(state.outlet_node))

    reach_at_stream: dict[int, int] = {}

    def downstream_reach(start: int) -> int:
        if start in reach_at_stream:
            return reach_at_stream[start]
        path: list[int] = []
        seen: set[int] = set()
        node = start
        first_node = True
        while True:
            if not first_node and node in junctions:
                endpoint = node
                break
            downstream = int(receiver[node])
            if node in seen or downstream == node or downstream not in stream_nodes:
                endpoint = node
                break
            if node in reach_at_stream:
                endpoint = reach_at_stream[node]
                break
            seen.add(node)
            path.append(node)
            node = downstream
            first_node = False
        for path_node in path:
            reach_at_stream[path_node] = endpoint
        reach_at_stream.setdefault(start, endpoint)
        return endpoint

    for node in stream_nodes:
        downstream_reach(node)

    first_stream_at_node: dict[int, int | None] = {
        node: node for node in stream_nodes
    }

    def first_stream(start: int) -> int | None:
        if start in first_stream_at_node:
            return first_stream_at_node[start]
        path: list[int] = []
        seen: set[int] = set()
        node = start
        while node not in first_stream_at_node:
            if node in seen or receiver[node] == node:
                first_stream_at_node[node] = None
                break
            seen.add(node)
            path.append(node)
            node = int(receiver[node])
        destination = first_stream_at_node.get(node)
        for path_node in path:
            first_stream_at_node[path_node] = destination
        return destination

    core_by_reach: dict[int, list[int]] = {}
    for node in grid.core_nodes:
        stream = first_stream(int(node))
        if stream is not None:
            reach = downstream_reach(stream)
            core_by_reach.setdefault(reach, []).append(int(node))

    node_to_zone: dict[int, int] = {}
    source_zones: dict[int, np.ndarray] = {}
    for node in np.unique(np.asarray(well_nodes, dtype=int)):
        stream = first_stream(int(node))
        if stream is None:
            raise ValueError(
                f"Pumping node {node} does not drain to an extracted stream source zone."
            )
        reach = downstream_reach(stream)
        if reach not in core_by_reach:
            raise ValueError(
                f"Pumping node {node} has no core cells in its channel-reach source zone."
            )
        node_to_zone[int(node)] = reach
        source_zones[reach] = np.asarray(core_by_reach[reach], dtype=int)
    return source_zones, node_to_zone


def apply_specific_yield_floor(
    state: GroundwaterState,
    specific_yield_floor: float,
) -> np.ndarray:
    """Apply a minimum effective specific yield across the modeled aquifer."""
    if not 0.0 <= specific_yield_floor <= 1.0:
        raise ValueError("Specific-yield floor must be between 0 and 1.")
    if specific_yield_floor == 0.0:
        return np.array([], dtype=int)
    grid = state.grid
    nodes = grid.core_nodes
    porosity = grid.at_node["aquifer__porosity"]
    source_values = porosity[nodes]
    changed = nodes[
        (source_values < specific_yield_floor)
        & ~np.isclose(
            source_values,
            specific_yield_floor,
            rtol=1.0e-6,
            atol=1.0e-12,
        )
    ]
    porosity[changed] = specific_yield_floor
    return changed


def _allocate_capped_weighted_volume(
    demand_m3d: float,
    capacities_m3d: np.ndarray,
    weights: np.ndarray,
    *,
    strict: bool = True,
) -> np.ndarray:
    """Allocate a daily volume by weight without exceeding node capacities."""
    allocation = np.zeros_like(capacities_m3d, dtype=float)
    remaining = float(demand_m3d)
    available = capacities_m3d > 0.0
    tolerance = max(1.0e-9, abs(demand_m3d) * 1.0e-10)

    while remaining > tolerance and np.any(available):
        active = np.flatnonzero(available)
        active_weights = np.maximum(weights[active], 0.0)
        if np.sum(active_weights) <= 0.0:
            active_weights = capacities_m3d[active] - allocation[active]
        shares = remaining * active_weights / np.sum(active_weights)
        room = capacities_m3d[active] - allocation[active]
        capped = shares >= room - tolerance
        if not np.any(capped):
            allocation[active] += shares
            remaining = 0.0
            break
        capped_nodes = active[capped]
        added = capacities_m3d[capped_nodes] - allocation[capped_nodes]
        allocation[capped_nodes] = capacities_m3d[capped_nodes]
        remaining -= float(np.sum(added))
        available[capped_nodes] = False

    if remaining > tolerance and strict:
        raise RuntimeError(
            f"Topographic source zone has {remaining:.3f} m3/day of demand beyond "
            "its available drainable-storage capacity."
        )
    return allocation


def _allocate_topographic_pumping(
    state: GroundwaterState,
    source_zones: dict[int, np.ndarray],
    zone_demands_m3d: dict[int, float],
    source_zone_storage_fraction: float,
    strict_pumping_supply: bool,
) -> np.ndarray:
    """Allocate demand within D8-bounded hillslopes using current hydraulic capacity."""
    grid = state.grid
    flux = np.zeros(grid.number_of_nodes)
    thickness = np.maximum(
        grid.at_node["water_table__elevation"]
        - grid.at_node["aquifer_base__elevation"],
        0.0,
    )
    storage = (
        thickness
        * grid.at_node["aquifer__porosity"]
        * grid.cell_area_at_node
    )
    transmissive_weight = (
        grid.at_node["aquifer__hydraulic_conductivity"]
        * thickness
        * grid.cell_area_at_node
    )

    for zone_id, demand_m3d in zone_demands_m3d.items():
        if demand_m3d <= 0.0:
            continue
        nodes = source_zones[zone_id]
        capacities = source_zone_storage_fraction * storage[nodes]
        capacities[thickness[nodes] <= 1.0e-8] = 0.0
        try:
            allocation = _allocate_capped_weighted_volume(
                demand_m3d,
                capacities,
                transmissive_weight[nodes],
                strict=strict_pumping_supply,
            )
        except RuntimeError as exc:
            available = float(np.sum(capacities))
            raise RuntimeError(
                f"Topographic source zone {zone_id} cannot supply {demand_m3d:.3f} "
                f"m3/day from {nodes.size} cell(s); daily capacity is "
                f"{available:.3f} m3/day."
            ) from exc
        flux[nodes] -= allocation / (
            grid.cell_area_at_node[nodes] * SECONDS_PER_DAY
        )
    return flux


def initialize_water_table(
    state: GroundwaterState,
    mode: str,
    config: GroundwaterConfig,
    heads_path: Path | None = None,
) -> None:
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
        heads = np.load(heads_path)
        if heads.shape != (grid.number_of_nodes,):
            raise ValueError(
                f"Head file has shape {heads.shape}; expected {(grid.number_of_nodes,)}. "
                "Regenerate spin-up heads with the current grid configuration."
            )
        if np.any(~np.isfinite(heads[grid.core_nodes])):
            raise ValueError("Head file contains non-finite values at active aquifer nodes.")
        base = grid.at_node["aquifer_base__elevation"]
        surface = grid.at_node["topographic__elevation"]
        if np.any(heads[grid.core_nodes] < base[grid.core_nodes] - 1e-8):
            raise ValueError("Head file contains water-table elevations below the aquifer base.")
        if np.any(heads[grid.core_nodes] > surface[grid.core_nodes] + 1e-8):
            raise ValueError("Head file contains water-table elevations above land surface.")
        grid.at_node["water_table__elevation"][:] = heads
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


def _groundwater_flux_out_m3s(grid: RasterModelGrid) -> float:
    """Return signed groundwater flux from core nodes into open stream nodes."""
    open_nodes = grid.open_boundary_nodes
    links_at_open = grid.links_at_node[open_nodes]
    directions = grid.active_link_dirs_at_node[open_nodes]
    active = directions != 0
    links = links_at_open[active]
    link_directions = directions[active]
    faces = grid.face_at_link[links]
    specific_discharge = grid.at_link["groundwater__specific_discharge"][links]
    return float(
        np.sum(specific_discharge * link_directions * grid.length_of_face[faces])
    )


@dataclass
class _GroundwaterFluxAccumulator:
    """Integrate groundwater boundary exchange over adaptive substeps."""

    reach_network: ReachNetwork | None = None
    volume_m3: float = 0.0
    reach_volume_m3: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_reaches = (
            self.reach_network.number_of_reaches
            if self.reach_network is not None
            else 0
        )
        self.reach_volume_m3 = np.zeros(n_reaches, dtype=float)

    def reset(self) -> None:
        self.volume_m3 = 0.0
        self.reach_volume_m3.fill(0.0)

    def __call__(self, grid, recharge_rate, substep_dt, **kwargs) -> None:
        del recharge_rate, kwargs
        if self.reach_network is None:
            self.volume_m3 += _groundwater_flux_out_m3s(grid) * substep_dt
            return

        network = self.reach_network
        links = network.boundary_links
        faces = grid.face_at_link[links]
        link_volumes = (
            grid.at_link["groundwater__specific_discharge"][links]
            * network.boundary_link_directions
            * grid.length_of_face[faces]
            * substep_dt
        )
        self.volume_m3 += float(np.sum(link_volumes))
        self.reach_volume_m3 += np.bincount(
            network.boundary_link_reaches,
            weights=link_volumes,
            minlength=network.number_of_reaches,
        )


def _availability_reconciled_reach_flow(
    state: GroundwaterState,
    network: ReachNetwork,
    groundwater_by_reach_m3: np.ndarray,
    surface_water_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical local/routed flow and the residual loss correction.

    The in-solver limiter enforces availability at every adaptive substep. This
    second routing pass removes only accumulated floating-point/convergence
    residue so saved daily flows cannot be slightly negative.
    """
    grid = state.grid
    core = grid.core_nodes
    raw_surface = np.bincount(
        network.reach_at_core_node[core],
        weights=surface_water_depth[core] * grid.cell_area_at_node[core],
        minlength=network.number_of_reaches,
    ).astype(float)
    raw_local = np.asarray(groundwater_by_reach_m3, dtype=float) + raw_surface
    routed, unavailable = route_streamflow_with_availability(
        raw_local, network.downstream_reach
    )
    corrected_local = raw_local + unavailable
    return corrected_local, routed, unavailable


def make_dupuit_component(
    state: GroundwaterState,
    config: GroundwaterConfig,
    *,
    callback_fun=None,
    reach_network: ReachNetwork | None = None,
):
    _, GroundwaterDupuitPercolator, _ = _landlab_components()
    kwargs = {}
    if callback_fun is not None:
        kwargs["callback_fun"] = callback_fun
    common = {
        "hydraulic_conductivity": "hydraulic_conductivity",
        "recharge_rate": "recharge_rate",
        "porosity": "aquifer__porosity",
        "courant_coefficient": config.courant_coefficient,
        **kwargs,
    }
    if config.stream_loss_mode == "unlimited_fixed_head":
        return GroundwaterDupuitPercolator(state.grid, **common)
    if reach_network is None:
        raise ValueError("Routed stream-loss limiting requires a reach network.")
    return RoutedStreamLimitedGroundwaterDupuitPercolator(
        state.grid,
        reach_at_core_node=reach_network.reach_at_core_node,
        boundary_links=reach_network.boundary_links,
        boundary_link_directions=reach_network.boundary_link_directions,
        boundary_link_reaches=reach_network.boundary_link_reaches,
        downstream_reach=reach_network.downstream_reach,
        limiter_tolerance_m3=config.stream_limiter_tolerance_m3,
        limiter_max_iterations=config.stream_limiter_max_iterations,
        **common,
    )


def _run_daily_step(
    state: GroundwaterState,
    gdp,
    groundwater_flux: _GroundwaterFluxAccumulator,
    *,
    recharge_mm_day: float | np.ndarray,
    previous_storage_m3: float,
    pumping_flux: np.ndarray | None = None,
    pumping_storage_fraction: float = 0.5,
    strict_pumping_supply: bool = False,
    return_surface_water_depth: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Advance one day and return explicit flux and mass-balance diagnostics."""
    grid = state.grid
    recharge_values = np.asarray(recharge_mm_day, dtype=float)
    if recharge_values.ndim == 0:
        core_recharge_mm_day = np.full(
            grid.core_nodes.size,
            float(recharge_values),
            dtype=float,
        )
    elif recharge_values.shape == (grid.number_of_nodes,):
        core_recharge_mm_day = recharge_values[grid.core_nodes]
    else:
        raise ValueError(
            "Recharge must be a basin-mean scalar or one value per groundwater-grid node."
        )
    if np.any(~np.isfinite(core_recharge_mm_day)) or np.any(core_recharge_mm_day < 0.0):
        raise ValueError("Recharge values at modeled aquifer nodes must be finite and nonnegative.")
    core_recharge_ms = (core_recharge_mm_day / 1000.0) / SECONDS_PER_DAY
    grid.at_node["recharge_rate"][:] = 0.0
    grid.at_node["recharge_rate"][grid.core_nodes] = core_recharge_ms

    pumping_m3d = 0.0
    if pumping_flux is not None:
        if pumping_flux.shape != (grid.number_of_nodes,):
            raise ValueError("Pumping flux field does not match the groundwater grid.")
        grid.at_node["recharge_rate"][grid.core_nodes] += pumping_flux[grid.core_nodes]
        pumping_m3d = float(
            -np.sum(
                pumping_flux[grid.core_nodes]
                * grid.cell_area_at_node[grid.core_nodes]
            )
            * SECONDS_PER_DAY
        )

    recharge_m3d = float(
        np.sum(
            core_recharge_ms
            * grid.cell_area_at_node[grid.core_nodes]
        )
        * SECONDS_PER_DAY
    )
    recharge_area_m2 = float(np.sum(grid.cell_area_at_node[grid.core_nodes]))
    basin_mean_recharge_mm_day = (
        recharge_m3d / recharge_area_m2 * 1000.0
        if recharge_area_m2 > 0.0
        else 0.0
    )

    groundwater_flux.reset()
    if hasattr(gdp, "reset_stream_limiter_diagnostics"):
        gdp.reset_stream_limiter_diagnostics()
    surface_water_depth = np.zeros(grid.number_of_nodes)
    total_substeps = 0
    pumping_chunks = 0

    if pumping_flux is None or not strict_pumping_supply:
        gdp.run_with_adaptive_time_step_solver(SECONDS_PER_DAY)
        surface_water_depth[:] = (
            grid.at_node["average_surface_water__specific_discharge"]
            * SECONDS_PER_DAY
        )
        total_substeps = int(gdp.number_of_substeps)
        pumping_chunks = 1
    else:
        remaining_time = SECONDS_PER_DAY
        while remaining_time > 1.0e-9:
            chunk_dt = remaining_time
            core = grid.core_nodes
            net_sink = np.maximum(-grid.at_node["recharge_rate"][core], 0.0)
            pumping_nodes = net_sink > 0.0
            if np.any(pumping_nodes):
                thickness = (
                    grid.at_node["water_table__elevation"][core]
                    - grid.at_node["aquifer_base__elevation"][core]
                )
                dry = pumping_nodes & (thickness <= 1.0e-10)
                if np.any(dry):
                    dry_nodes = core[dry]
                    unresolved_m3d = float(
                        np.sum(
                            net_sink[dry]
                            * grid.cell_area_at_node[dry_nodes]
                            * SECONDS_PER_DAY
                        )
                    )
                    preview = ", ".join(str(node) for node in dry_nodes[:5])
                    raise RuntimeError(
                        f"Pumping demand reaches {dry_nodes.size} dry aquifer node(s) "
                        f"({preview}) with {unresolved_m3d:.3f} m3/day unresolved. Increase the "
                        "modeled aquifer depth or revise its hydraulic properties."
                    )
                time_to_empty = (
                    grid.at_node["aquifer__porosity"][core][pumping_nodes]
                    * thickness[pumping_nodes]
                    / net_sink[pumping_nodes]
                )
                chunk_dt = min(
                    chunk_dt,
                    pumping_storage_fraction * float(np.min(time_to_empty)),
                )
                if chunk_dt <= 1.0e-6:
                    pumping_core = core[pumping_nodes]
                    limiting_index = int(np.argmin(time_to_empty))
                    limiting_node = int(pumping_core[limiting_index])
                    limiting_thickness = float(thickness[pumping_nodes][limiting_index])
                    limiting_porosity = float(
                        grid.at_node["aquifer__porosity"][limiting_node]
                    )
                    limiting_sink_m3d = float(
                        net_sink[pumping_nodes][limiting_index]
                        * grid.cell_area_at_node[limiting_node]
                        * SECONDS_PER_DAY
                    )
                    raise RuntimeError(
                        "Pumping-aware timestep collapsed below one microsecond at node "
                        f"{limiting_node} (saturated thickness={limiting_thickness:.6g} m, "
                        f"specific yield={limiting_porosity:.6g}, scheduled sink="
                        f"{limiting_sink_m3d:.3f} m3/day); the scheduled sink cannot be "
                        "supplied by the modeled aquifer."
                    )

            gdp.run_with_adaptive_time_step_solver(chunk_dt)
            surface_water_depth += (
                grid.at_node["average_surface_water__specific_discharge"] * chunk_dt
            )
            total_substeps += int(gdp.number_of_substeps)
            pumping_chunks += 1
            remaining_time = max(0.0, remaining_time - chunk_dt)

    core = grid.core_nodes
    groundwater_to_stream_m3d = groundwater_flux.volume_m3
    landlab_saturation_excess_m3d = float(
        np.sum(surface_water_depth[core] * grid.cell_area_at_node[core])
    )
    current_storage_m3 = _storage_volume(state)
    storage_change_m3 = current_storage_m3 - previous_storage_m3
    mass_balance_streamflow_m3d = recharge_m3d - pumping_m3d - storage_change_m3
    # Landlab reports surface discharge at the end of each adaptive substep. Its
    # time integral can be biased when low conductivity permits one long substep.
    # With no reinfiltration or other boundary loss, the surface-water remainder
    # closes exactly from the applied sources, storage update, and integrated
    # groundwater exchange. Preserve Landlab's estimate as a numerical diagnostic.
    availability_roundoff_m3d = 0.0
    if isinstance(gdp, RoutedStreamLimitedGroundwaterDupuitPercolator):
        if groundwater_flux.reach_network is None:
            raise RuntimeError(
                "Routed stream-loss limiting requires reach-aware daily accounting."
            )
        _, routed_flow, unavailable = _availability_reconciled_reach_flow(
            state,
            groundwater_flux.reach_network,
            groundwater_flux.reach_volume_m3,
            surface_water_depth,
        )
        availability_roundoff_m3d = float(np.sum(unavailable))
        roundoff_limit_m3 = max(
            1.0e-3, 100.0 * float(gdp._limiter_tolerance_m3)
        )
        if availability_roundoff_m3d > roundoff_limit_m3:
            raise RuntimeError(
                "Daily reach routing required an availability correction of "
                f"{availability_roundoff_m3d:.6g} m3/day, above the numerical "
                f"limit of {roundoff_limit_m3:.6g} m3/day."
            )
        outlet = np.flatnonzero(
            groundwater_flux.reach_network.downstream_reach < 0
        )
        if outlet.size != 1:
            raise RuntimeError("Reach network must contain exactly one outlet.")
        total_streamflow_m3d = float(routed_flow[outlet[0]])
        saturation_excess_m3d = total_streamflow_m3d - groundwater_to_stream_m3d
    else:
        saturation_excess_m3d = (
            mass_balance_streamflow_m3d - groundwater_to_stream_m3d
        )
        tolerance_m3 = 1.0e-8 * max(
            abs(recharge_m3d), abs(pumping_m3d), abs(storage_change_m3), 1.0
        )
        if saturation_excess_m3d < -tolerance_m3:
            raise RuntimeError(
                "Closed surface-water remainder is negative by "
                f"{-saturation_excess_m3d:.6g} m3/day; inspect groundwater flux and storage."
            )
        saturation_excess_m3d = max(saturation_excess_m3d, 0.0)
        total_streamflow_m3d = groundwater_to_stream_m3d + saturation_excess_m3d
    mass_balance_error_m3d = mass_balance_streamflow_m3d - total_streamflow_m3d
    balance_scale = max(
        abs(recharge_m3d),
        abs(pumping_m3d),
        abs(storage_change_m3),
        abs(total_streamflow_m3d),
        1.0,
    )

    record = {
        "recharge_mm_day": basin_mean_recharge_mm_day,
        "recharge_m3d": recharge_m3d,
        "pumping_m3d": pumping_m3d,
        "groundwater_to_stream_m3d": groundwater_to_stream_m3d,
        "groundwater_discharge_m3d": max(groundwater_to_stream_m3d, 0.0),
        "stream_loss_to_groundwater_m3d": max(-groundwater_to_stream_m3d, 0.0),
        "saturation_excess_m3d": saturation_excess_m3d,
        "landlab_saturation_excess_m3d": landlab_saturation_excess_m3d,
        "landlab_surface_flux_integration_error_m3d": (
            landlab_saturation_excess_m3d - saturation_excess_m3d
        ),
        "total_streamflow_m3d": total_streamflow_m3d,
        "Q_m3d": total_streamflow_m3d,
        "storage_m3": current_storage_m3,
        "storage_change_m3": storage_change_m3,
        "mass_balance_streamflow_m3d": mass_balance_streamflow_m3d,
        "mass_balance_error_m3d": mass_balance_error_m3d,
        "mass_balance_error_pct": 100.0 * mass_balance_error_m3d / balance_scale,
        "streamflow_availability_roundoff_m3d": availability_roundoff_m3d,
        "substeps": total_substeps,
        "pumping_chunks": pumping_chunks,
        "unavailable_stream_loss_m3d": float(
            getattr(gdp, "unavailable_stream_loss_m3", 0.0)
        ),
        "stream_limiter_substeps": int(
            getattr(gdp, "stream_limiter_substeps", 0)
        ),
        "stream_limiter_max_iterations_used": int(
            getattr(gdp, "stream_limiter_max_iterations_used", 0)
        ),
        "stream_limiter_max_dry_reaches": int(
            getattr(gdp, "stream_limiter_max_dry_reaches", 0)
        ),
        "stream_limiter_numerical_clip_m3": float(
            getattr(gdp, "stream_limiter_numerical_clip_m3", 0.0)
        ),
    }
    if return_surface_water_depth:
        return record, surface_water_depth
    return record


def spin_up_transient(
    state: GroundwaterState,
    output_heads: Path,
    recharge_csv: Path | None,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    config: GroundwaterConfig = GroundwaterConfig(),
    initial_heads_path: Path | None = None,
    pumping_forcing: PumpingForcing | None = None,
    reach_network: ReachNetwork | None = None,
    recharge_raster_manifest: Path | None = None,
) -> pd.DataFrame:
    """Spin up the aquifer with transient recharge and optional pumping."""
    initialize_water_table(
        state,
        "heads" if initial_heads_path is not None else "empty",
        config,
        initial_heads_path,
    )
    if config.stream_loss_mode == "routed_volume_limited" and reach_network is None:
        reach_network = build_reach_network(state)
    groundwater_flux = _GroundwaterFluxAccumulator(reach_network)
    gdp = make_dupuit_component(
        state,
        config,
        callback_fun=groundwater_flux,
        reach_network=reach_network,
    )
    recharge = load_daily_recharge_forcing(
        recharge_csv=recharge_csv,
        recharge_raster_manifest=recharge_raster_manifest,
        start_date=start_date,
        end_date=end_date,
    )
    records = []

    previous_vol = _storage_volume(state)
    initial_storage_m3 = previous_vol

    for row in tqdm(
        recharge.frame.itertuples(index=False),
        total=len(recharge.frame),
        desc="Spin-up (Transient)",
    ):
        pump_field = (
            pumping_forcing.flux_for_date(
                row.date,
                state=state,
                source_zone_storage_fraction=config.source_zone_storage_fraction,
                strict_pumping_supply=config.strict_pumping_supply,
            )
            if pumping_forcing is not None
            else None
        )
        record = _run_daily_step(
            state,
            gdp,
            groundwater_flux,
            recharge_mm_day=recharge.value_for_record(row, state),
            previous_storage_m3=previous_vol,
            pumping_flux=pump_field,
            pumping_storage_fraction=config.pumping_storage_fraction,
            strict_pumping_supply=(
                config.strict_pumping_supply
                or (
                    pumping_forcing is not None
                    and pumping_forcing.source_mode == "topographic"
                )
            ),
        )
        record["date"] = row.date
        record["initial_storage_m3"] = initial_storage_m3
        record["scheduled_pumping_m3d"] = (
            pumping_forcing.scheduled_volume_for_date(row.date, state)
            if pumping_forcing is not None
            else 0.0
        )
        records.append(record)
        previous_vol = record["storage_m3"]

    output_heads = Path(output_heads)
    output_heads.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_heads, state.grid.at_node["water_table__elevation"])
    return pd.DataFrame(records)


def _nearest_core_nodes(grid: RasterModelGrid, node_ids: np.ndarray) -> np.ndarray:
    """Move non-core well locations to the closest modeled aquifer node."""
    node_ids = np.asarray(node_ids, dtype=int).copy()
    non_core = grid.status_at_node[node_ids] != grid.BC_NODE_IS_CORE
    if not np.any(non_core):
        return node_ids
    core_xy = np.column_stack(
        (grid.x_of_node[grid.core_nodes], grid.y_of_node[grid.core_nodes])
    )
    query_xy = np.column_stack(
        (grid.x_of_node[node_ids[non_core]], grid.y_of_node[node_ids[non_core]])
    )
    _, nearest = cKDTree(core_xy).query(query_xy)
    node_ids[non_core] = grid.core_nodes[nearest]
    return node_ids


def build_monthly_pumping_maps(
    state: GroundwaterState,
    wells_path: Path,
    pumping_path: Path,
    basin_path: Path,
    *,
    target_crs: str = "EPSG:26910",
    apn_col: str = "APN",
    mode: str = "timeseries",
    source_mode: str = "well_cell",
    source_area_threshold: float | None = None,
) -> PumpingForcing:
    """Map APN pumping to nodes as dated or climatological negative fluxes."""
    if mode not in {"climatology", "timeseries"}:
        raise ValueError("Pumping mode must be 'climatology' or 'timeseries'.")
    if source_mode not in {"well_cell", "topographic"}:
        raise ValueError("Pumping source mode must be 'well_cell' or 'topographic'.")

    basin_gdf = gpd.read_file(basin_path).to_crs(target_crs)
    wells_gdf = gpd.read_file(wells_path).to_crs(target_crs)
    if apn_col not in wells_gdf.columns:
        raise ValueError(f"Well file must contain an {apn_col!r} column.")
    basin_geometry = basin_gdf.geometry.union_all()
    wells_in_catchment = wells_gdf[wells_gdf.geometry.intersects(basin_geometry)].copy()
    if wells_in_catchment.empty:
        raise ValueError("No pumping wells intersect the basin boundary.")

    df_pump = pd.read_csv(pumping_path)
    if apn_col not in df_pump.columns:
        raise ValueError(f"Pumping CSV must contain an {apn_col!r} column.")

    well_x = wells_in_catchment.geometry.x.values
    well_y = wells_in_catchment.geometry.y.values
    node_ids = state.grid.find_nearest_node((well_x, well_y), mode="clip")
    node_ids = _nearest_core_nodes(state.grid, node_ids)
    spatial_map = pd.DataFrame(
        {apn_col: wells_in_catchment[apn_col].values, "NodeID": node_ids}
    )
    spatial_map["well_fraction"] = 1.0 / spatial_map.groupby(apn_col)[apn_col].transform("size")

    if "waterUse_m3Day" in df_pump.columns:
        df_pump["rate_m3d"] = pd.to_numeric(df_pump["waterUse_m3Day"], errors="coerce")
    elif "waterUse_m3Month" in df_pump.columns:
        monthly_volume = pd.to_numeric(df_pump["waterUse_m3Month"], errors="coerce")
        if "MonthLengthDays" in df_pump.columns:
            month_length = pd.to_numeric(df_pump["MonthLengthDays"], errors="coerce")
        elif "Date" in df_pump.columns:
            month_length = pd.to_datetime(df_pump["Date"], errors="coerce").dt.days_in_month
        elif "Year" in df_pump.columns:
            month_length = pd.to_datetime(
                {
                    "year": pd.to_numeric(df_pump["Year"], errors="coerce"),
                    "month": pd.to_numeric(df_pump["Month"], errors="coerce"),
                    "day": 1,
                },
                errors="coerce",
            ).dt.days_in_month
        else:
            raise ValueError(
                "Monthly pumping volumes require MonthLengthDays, Date, or Year."
            )
        df_pump["rate_m3d"] = monthly_volume / month_length
    else:
        raise ValueError("Pumping CSV must contain `waterUse_m3Day` or `waterUse_m3Month`.")

    if np.any(~np.isfinite(df_pump["rate_m3d"])) or np.any(df_pump["rate_m3d"] < 0.0):
        raise ValueError("Pumping rates must be finite and nonnegative.")

    coverage_start = None
    coverage_end = None
    schedule_keys: set[int | pd.Period]
    if mode == "timeseries":
        if "Date" in df_pump.columns:
            dates = pd.to_datetime(df_pump["Date"], errors="coerce")
        elif "Year" in df_pump.columns:
            dates = pd.to_datetime(
                {
                    "year": pd.to_numeric(df_pump["Year"], errors="coerce"),
                    "month": pd.to_numeric(df_pump["Month"], errors="coerce"),
                    "day": 1,
                },
                errors="coerce",
            )
        else:
            raise ValueError("Timeseries pumping mode requires a Date or Year column.")
        if dates.isna().any():
            raise ValueError("Pumping CSV contains invalid dates.")
        df_pump["forcing_key"] = dates.dt.to_period("M")
        coverage_start = df_pump["forcing_key"].min()
        coverage_end = df_pump["forcing_key"].max()
        schedule_keys = set(pd.period_range(coverage_start, coverage_end, freq="M"))
        missing_schedule_months = schedule_keys.difference(df_pump["forcing_key"].unique())
        if missing_schedule_months:
            missing = ", ".join(str(value) for value in sorted(missing_schedule_months))
            raise ValueError(f"Pumping schedule has missing months within its coverage: {missing}")
        grouping = ["forcing_key", apn_col]
    else:
        if "Month" in df_pump.columns:
            month_values = df_pump["Month"]
        elif "Date" in df_pump.columns:
            month_values = pd.to_datetime(
                df_pump["Date"], errors="coerce"
            ).dt.month
        else:
            raise ValueError(
                "Climatology pumping mode requires a Month or Date column."
            )
        df_pump["forcing_key"] = pd.to_numeric(month_values, errors="coerce")
        if df_pump["forcing_key"].isna().any() or not df_pump["forcing_key"].between(1, 12).all():
            raise ValueError("Pumping months must be integers from 1 through 12.")
        df_pump["forcing_key"] = df_pump["forcing_key"].astype(int)
        schedule_keys = set(range(1, 13))
        grouping = ["forcing_key", apn_col]

    valid_apns = set(spatial_map[apn_col])
    schedule_for_wells = df_pump[df_pump[apn_col].isin(valid_apns)]
    if schedule_for_wells.empty:
        raise ValueError("No pumping records match wells inside the basin.")
    if mode == "timeseries":
        duplicate_rates = schedule_for_wells.duplicated(["forcing_key", apn_col])
        if duplicate_rates.any():
            raise ValueError("Pumping schedule has duplicate APN records within a month.")
        expected_pairs = pd.MultiIndex.from_product(
            [sorted(schedule_keys), sorted(valid_apns)],
            names=["forcing_key", apn_col],
        )
        actual_pairs = pd.MultiIndex.from_frame(
            schedule_for_wells[["forcing_key", apn_col]]
        )
        missing_pairs = expected_pairs.difference(actual_pairs)
        if len(missing_pairs):
            preview = ", ".join(
                f"{period}/{apn}" for period, apn in missing_pairs[:5]
            )
            raise ValueError(
                f"Pumping schedule is missing {len(missing_pairs)} well-month records, "
                f"including {preview}."
            )
    rates = (
        schedule_for_wells
        .groupby(grouping, as_index=False)["rate_m3d"]
        .mean()
    )
    pumping_apns = set(rates.loc[rates["rate_m3d"] > 0.0, apn_col])
    well_nodes = np.unique(
        spatial_map.loc[spatial_map[apn_col].isin(pumping_apns), "NodeID"].to_numpy(
            dtype=int
        )
    )
    if well_nodes.size == 0:
        raise ValueError("Pumping records for wells inside the basin are all zero.")
    source_zones = None
    node_to_zone = None
    if source_mode == "topographic":
        if source_area_threshold is None:
            source_area_threshold = state.stream_area_threshold
        source_zones, node_to_zone = _topographic_source_zones(
            state,
            well_nodes,
            source_area_threshold,
        )
        spatial_map["SourceZoneID"] = spatial_map["NodeID"].map(node_to_zone)
        if spatial_map.loc[
            spatial_map[apn_col].isin(pumping_apns), "SourceZoneID"
        ].isna().any():
            raise ValueError("At least one pumping well has no topographic source zone.")
    maps: dict[int | pd.Period, np.ndarray] = {}
    zone_demands: dict[int | pd.Period, dict[int, float]] = {}
    for key, key_rates in rates.groupby("forcing_key"):
        merged = spatial_map.merge(key_rates[[apn_col, "rate_m3d"]], on=apn_col, how="inner")
        vol_per_node_m3d = np.bincount(
            merged["NodeID"],
            weights=merged["rate_m3d"] * merged["well_fraction"],
            minlength=state.grid.number_of_nodes,
        )
        maps[key] = -1.0 * (vol_per_node_m3d / SECONDS_PER_DAY) / state.cell_area
        if source_mode == "topographic":
            merged["weighted_rate_m3d"] = (
                merged["rate_m3d"] * merged["well_fraction"]
            )
            by_zone = merged.groupby("SourceZoneID")["weighted_rate_m3d"].sum()
            zone_demands[key] = {
                int(zone_id): float(value)
                for zone_id, value in by_zone.items()
                if value > 0.0
            }

    zero_flux = np.zeros(state.grid.number_of_nodes)
    for key in schedule_keys:
        maps.setdefault(key, zero_flux.copy())
        zone_demands.setdefault(key, {})
    source_nodes = None
    if source_zones:
        demanded_zone_ids = {
            zone_id
            for demands in zone_demands.values()
            for zone_id in demands
        }
        source_zones = {
            zone_id: nodes
            for zone_id, nodes in source_zones.items()
            if zone_id in demanded_zone_ids
        }
        source_nodes = np.unique(np.concatenate(list(source_zones.values())))
    return PumpingForcing(
        mode=mode,
        flux_fields=maps,
        zero_flux=zero_flux,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        well_nodes=well_nodes,
        source_mode=source_mode,
        source_zones=source_zones,
        zone_demands_m3d=zone_demands if source_mode == "topographic" else None,
        source_nodes=source_nodes,
        source_area_threshold_m2=(
            source_area_threshold if source_mode == "topographic" else None
        ),
    )


def load_recharge_data(
    recharge_csv: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load a complete, unique, finite daily recharge series without silent filling."""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    if end_dt < start_dt:
        raise ValueError("Recharge end date must be on or after the start date.")
    df = pd.read_csv(recharge_csv)
    if "date" not in df.columns or "Recharge" not in df.columns:
        raise ValueError("Recharge CSV must contain `date` and `Recharge` columns.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Recharge"] = pd.to_numeric(df["Recharge"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Recharge CSV contains invalid dates.")
    if df["date"].duplicated().any():
        duplicates = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(f"Recharge CSV contains duplicate dates: {', '.join(duplicates[:5])}")
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].sort_values("date")
    if df.empty:
        raise ValueError(
            f"No recharge data found for period {start_dt.date()} to {end_dt.date()}."
        )
    expected = pd.date_range(start_dt, end_dt, freq="D")
    missing = expected.difference(df["date"])
    if len(missing):
        preview = ", ".join(value.strftime("%Y-%m-%d") for value in missing[:5])
        raise ValueError(
            f"Recharge forcing is missing {len(missing)} requested day(s), including {preview}. "
            "Regenerate the forcing rather than silently carrying values forward."
        )
    if np.any(~np.isfinite(df["Recharge"])) or np.any(df["Recharge"] < 0.0):
        raise ValueError("Recharge values must be finite and nonnegative.")
    return df.reset_index(drop=True)


def load_daily_recharge_forcing(
    *,
    recharge_csv: Path | None,
    recharge_raster_manifest: Path | None,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> DailyRechargeForcing:
    """Load exactly one complete scalar or spatial recharge source."""
    if (recharge_csv is None) == (recharge_raster_manifest is None):
        raise ValueError(
            "Provide exactly one of a recharge CSV or recharge raster manifest."
        )
    if recharge_raster_manifest is not None:
        path = Path(recharge_raster_manifest)
        frame = load_recharge_raster_manifest(path, start_date, end_date)
        return DailyRechargeForcing(
            frame=frame,
            source_type="raster_manifest",
            source_path=path.resolve(),
        )
    path = Path(recharge_csv)
    return DailyRechargeForcing(
        frame=load_recharge_data(path, start_date, end_date),
        source_type="basin_mean_csv",
        source_path=path.resolve(),
    )


def _reach_daily_values(
    state: GroundwaterState,
    network: ReachNetwork,
    groundwater_flux: _GroundwaterFluxAccumulator,
    surface_water_depth: np.ndarray,
    basin_record: dict[str, float],
    *,
    enforce_stream_availability: bool = False,
) -> dict[str, np.ndarray]:
    """Aggregate one modeled day into non-overlapping receiving-reach values."""
    grid = state.grid
    core = grid.core_nodes
    groundwater = groundwater_flux.reach_volume_m3.copy()
    if enforce_stream_availability:
        local_total, _, unavailable = _availability_reconciled_reach_flow(
            state,
            network,
            groundwater,
            surface_water_depth,
        )
        expected_roundoff = float(
            basin_record["streamflow_availability_roundoff_m3d"]
        )
        if not np.isclose(
            float(np.sum(unavailable)), expected_roundoff, rtol=1.0e-10, atol=1.0e-12
        ):
            raise RuntimeError("Reach availability correction is not reproducible.")
    else:
        reach_index = network.reach_at_core_node[core]
        raw_surface = np.bincount(
            reach_index,
            weights=surface_water_depth[core] * grid.cell_area_at_node[core],
            minlength=network.number_of_reaches,
        ).astype(float)
        raw_total = float(np.sum(raw_surface))
        canonical_total = float(basin_record["saturation_excess_m3d"])
        if canonical_total == 0.0:
            canonical_surface = np.zeros(network.number_of_reaches, dtype=float)
        elif raw_total > 0.0:
            canonical_surface = raw_surface * (canonical_total / raw_total)
        else:
            # This fallback is only relevant if Landlab reports no spatial surface
            # flux despite a positive closed basin remainder. It preserves exact
            # closure using the least-assumptive available spatial weight.
            canonical_surface = (
                canonical_total
                * network.incremental_area_m2
                / np.sum(network.incremental_area_m2)
            )
        local_total = groundwater + canonical_surface

    checks = {
        "groundwater": (
            float(np.sum(groundwater)),
            float(basin_record["groundwater_to_stream_m3d"]),
        ),
        "total streamflow": (
            float(np.sum(local_total)),
            float(basin_record["total_streamflow_m3d"]),
        ),
    }
    for label, (reach_total, basin_total) in checks.items():
        tolerance = 1.0e-8 * max(abs(basin_total), 1.0)
        if not np.isclose(reach_total, basin_total, rtol=1.0e-10, atol=tolerance):
            raise RuntimeError(
                f"Reach-local {label} ({reach_total:.12g} m3/day) does not match "
                f"the basin value ({basin_total:.12g} m3/day)."
            )

    return {"local_total_streamflow_m3d": local_total}


def _stack_reach_daily_records(
    records: list[tuple[pd.Timestamp, dict[str, np.ndarray]]],
    n_reaches: int,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    columns = records[0][1].keys()
    output: dict[str, np.ndarray] = {
        "date": np.concatenate(
            [np.repeat(np.datetime64(date, "ns"), n_reaches) for date, _ in records]
        ),
        "reach_id": np.tile(np.arange(1, n_reaches + 1, dtype=np.int32), len(records)),
    }
    for column in columns:
        output[column] = np.concatenate([values[column] for _, values in records])
    return pd.DataFrame(output)


def build_reach_daily_table(
    reach_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine scenario reach contributions into one stable paired-output schema."""
    natural = reach_results["Unimpaired (Natural)"].sort_values(
        ["date", "reach_id"]
    ).reset_index(drop=True)
    keys = natural[["date", "reach_id"]]
    output = keys.copy()
    output["unimpaired_local_total_streamflow_m3d"] = natural[
        "local_total_streamflow_m3d"
    ].to_numpy()

    pumped = reach_results.get("With Pumping")
    if pumped is None:
        output["pumped_local_total_streamflow_m3d"] = np.nan
        output["local_total_streamflow_depletion_m3d"] = np.nan
        output["local_streamflow_depletion_fraction_pct"] = np.nan
        return output

    pumped = pumped.sort_values(["date", "reach_id"]).reset_index(drop=True)
    if not keys.equals(pumped[["date", "reach_id"]]):
        raise ValueError("Natural and pumped reach results do not have identical keys.")
    output["pumped_local_total_streamflow_m3d"] = pumped[
        "local_total_streamflow_m3d"
    ].to_numpy()
    total_depletion = (
        output["unimpaired_local_total_streamflow_m3d"].to_numpy()
        - output["pumped_local_total_streamflow_m3d"].to_numpy()
    )
    output["local_total_streamflow_depletion_m3d"] = total_depletion
    depletion_fraction = np.full(len(output), np.nan, dtype=float)
    natural_total = output["unimpaired_local_total_streamflow_m3d"].to_numpy()
    np.divide(
        100.0 * total_depletion,
        natural_total,
        out=depletion_fraction,
        where=natural_total > 0.0,
    )
    output["local_streamflow_depletion_fraction_pct"] = depletion_fraction
    return output


def route_reach_daily_table(
    daily: pd.DataFrame,
    downstream_reach: np.ndarray,
    *,
    enforce_nonnegative: bool = False,
) -> pd.DataFrame:
    """Add upstream-integrated total flows and depletion to each stream reach."""
    output = daily.sort_values(["date", "reach_id"]).reset_index(drop=True).copy()
    downstream = np.asarray(downstream_reach, dtype=int)
    n_reaches = downstream.size
    reach_ids = np.arange(1, n_reaches + 1, dtype=int)
    if not np.array_equal(np.sort(output["reach_id"].unique()), reach_ids):
        raise ValueError("Reach daily IDs do not match the reach network.")
    counts = output.groupby("date", sort=False)["reach_id"].size().to_numpy()
    if np.any(counts != n_reaches):
        raise ValueError("Every reach must have exactly one row on every modeled day.")

    donor_count = np.bincount(
        downstream[downstream >= 0], minlength=n_reaches
    ).astype(int)
    queue = [int(value) for value in np.flatnonzero(donor_count == 0)]
    topological_order: list[int] = []
    while queue:
        reach_index = queue.pop(0)
        topological_order.append(reach_index)
        next_reach = int(downstream[reach_index])
        if next_reach >= 0:
            donor_count[next_reach] -= 1
            if donor_count[next_reach] == 0:
                queue.append(next_reach)
    if len(topological_order) != n_reaches:
        raise ValueError("Reach routing graph is cyclic or incomplete.")

    n_dates = len(output) // n_reaches

    def routed_values(column: str) -> np.ndarray:
        local = output[column].to_numpy(dtype=float).reshape(n_dates, n_reaches)
        routed = local.copy()
        for reach_index in topological_order:
            next_reach = int(downstream[reach_index])
            if next_reach >= 0:
                routed[:, next_reach] += routed[:, reach_index]
        if enforce_nonnegative:
            tolerance = 1.0e-10
            minimum = float(np.min(routed))
            if minimum < -tolerance:
                raise RuntimeError(
                    f"Availability-limited routed flow is negative ({minimum:.6g} "
                    "m3/day)."
                )
            routed[routed < 0.0] = 0.0
        return routed.ravel()

    output["routed_unimpaired_total_streamflow_m3d"] = routed_values(
        "unimpaired_local_total_streamflow_m3d"
    )
    if output["pumped_local_total_streamflow_m3d"].notna().any():
        output["routed_pumped_total_streamflow_m3d"] = routed_values(
            "pumped_local_total_streamflow_m3d"
        )
        routed_depletion = (
            output["routed_unimpaired_total_streamflow_m3d"].to_numpy()
            - output["routed_pumped_total_streamflow_m3d"].to_numpy()
        )
        output["routed_total_streamflow_depletion_m3d"] = routed_depletion
        denominator = output["routed_unimpaired_total_streamflow_m3d"].to_numpy()
        fraction = np.full(len(output), np.nan, dtype=float)
        np.divide(
            100.0 * routed_depletion,
            denominator,
            out=fraction,
            where=denominator > 0.0,
        )
        if enforce_nonnegative:
            fraction = np.minimum(fraction, 100.0)
        output["routed_streamflow_depletion_fraction_pct"] = fraction
    else:
        output["routed_pumped_total_streamflow_m3d"] = np.nan
        output["routed_total_streamflow_depletion_m3d"] = np.nan
        output["routed_streamflow_depletion_fraction_pct"] = np.nan
    return output


def _reach_geodataframe(
    state: GroundwaterState,
    network: ReachNetwork,
) -> gpd.GeoDataFrame:
    from shapely.geometry import LineString

    grid = state.grid
    geometries = []
    for reach_index, chain in enumerate(network.stream_nodes_by_reach):
        geometry_nodes = chain
        next_reach = int(network.downstream_reach[reach_index])
        if next_reach >= 0:
            geometry_nodes = np.append(
                geometry_nodes, network.stream_nodes_by_reach[next_reach][0]
            )
        coordinates = [
            (float(grid.x_of_node[node]), float(grid.y_of_node[node]))
            for node in geometry_nodes
        ]
        if len(coordinates) == 1:
            x, y = coordinates[0]
            coordinates = [(x - state.dx / 2.0, y), (x + state.dx / 2.0, y)]
        geometries.append(LineString(coordinates))

    downstream_ids = pd.array(
        [
            int(value) + 1 if value >= 0 else pd.NA
            for value in network.downstream_reach
        ],
        dtype="Int32",
    )
    crs = None
    if state.dem_coarse is not None and hasattr(state.dem_coarse, "rio"):
        crs = state.dem_coarse.rio.crs
    return gpd.GeoDataFrame(
        {
            "reach_id": np.arange(1, network.number_of_reaches + 1, dtype=np.int32),
            "downstream_reach_id": downstream_ids,
            "is_outlet_reach": network.downstream_reach < 0,
            "stream_node_count": np.asarray(
                [len(nodes) for nodes in network.stream_nodes_by_reach], dtype=np.int32
            ),
            "reach_length_m": network.reach_length_m,
            "incremental_area_m2": network.incremental_area_m2,
            "upstream_area_m2": network.upstream_area_m2,
            "definition_version": REACH_DEFINITION_VERSION,
            "definition_sha256": network.definition_sha256,
        },
        geometry=geometries,
        crs=crs,
    )


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_reach_outputs(
    state: GroundwaterState,
    network: ReachNetwork,
    reach_results: dict[str, pd.DataFrame],
    basin_results: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    enforce_stream_availability: bool = False,
) -> dict[str, object]:
    """Write compact standard reach products and return provenance metadata."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Standard reach outputs require pyarrow for compressed Parquet files."
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily = route_reach_daily_table(
        build_reach_daily_table(reach_results),
        network.downstream_reach,
        enforce_nonnegative=enforce_stream_availability,
    )

    # Verify the defining distributed-output invariant independently on the final
    # table: daily local values sum to the already-audited basin series.
    validation: dict[str, float] = {}
    for scenario, prefix in (
        ("Unimpaired (Natural)", "unimpaired"),
        ("With Pumping", "pumped"),
    ):
        if scenario not in basin_results:
            continue
        reach_daily = daily.groupby("date", sort=True)[
            f"{prefix}_local_total_streamflow_m3d"
        ].sum()
        basin = basin_results[scenario].sort_values("date").set_index("date")[
            "total_streamflow_m3d"
        ]
        difference = reach_daily.to_numpy() - basin.to_numpy()
        max_error = float(np.max(np.abs(difference)))
        validation[f"maximum_{prefix}_daily_aggregation_error_m3d"] = max_error
        tolerance = 1.0e-8 * max(float(np.max(np.abs(basin.to_numpy()))), 1.0)
        if max_error > tolerance:
            raise RuntimeError(
                f"Final {scenario} reach table does not reproduce basin streamflow; "
                f"maximum daily error is {max_error:.6g} m3/day."
            )
        outlet_reach_id = int(np.flatnonzero(network.downstream_reach < 0)[0]) + 1
        routed_outlet = daily.loc[
            daily["reach_id"] == outlet_reach_id,
            f"routed_{prefix}_total_streamflow_m3d",
        ].to_numpy()
        routed_error = float(np.max(np.abs(routed_outlet - basin.to_numpy())))
        validation[f"maximum_{prefix}_routed_outlet_error_m3d"] = routed_error
        if routed_error > tolerance:
            raise RuntimeError(
                f"Routed {scenario} outlet flow does not reproduce basin streamflow; "
                f"maximum daily error is {routed_error:.6g} m3/day."
            )

    table = pa.Table.from_pandas(daily, preserve_index=False)
    parquet_metadata = dict(table.schema.metadata or {})
    parquet_metadata.update(
        {
            b"reach_output_schema_version": REACH_OUTPUT_SCHEMA_VERSION.encode(),
            b"reach_definition_version": REACH_DEFINITION_VERSION.encode(),
            b"reach_definition_sha256": network.definition_sha256.encode(),
            b"spatial_semantics": (
                b"Total streamflow only. Local columns exclude upstream inflow. Routed "
                b"columns integrate the local total-flow contribution from the reach and "
                b"all upstream reaches; no channel lag or loss is applied. Values do not "
                b"attribute effects to pumping-source zones."
            ),
        }
    )
    table = table.replace_schema_metadata(parquet_metadata)
    parquet_path = output_dir / "reach_daily.parquet"
    pq.write_table(table, parquet_path, compression="zstd")

    reaches = _reach_geodataframe(state, network)
    natural_summary = daily.groupby("reach_id", sort=True).agg(
        cumulative_unimpaired_local_streamflow_m3=(
            "unimpaired_local_total_streamflow_m3d",
            "sum",
        ),
        mean_unimpaired_local_streamflow_m3d=(
            "unimpaired_local_total_streamflow_m3d",
            "mean",
        ),
    )
    reaches = reaches.merge(natural_summary, on="reach_id", how="left")
    if "With Pumping" in basin_results:
        summary = daily.groupby("reach_id", sort=True).agg(
            cumulative_local_depletion_m3=(
                "local_total_streamflow_depletion_m3d",
                "sum",
            ),
            cumulative_routed_depletion_m3=(
                "routed_total_streamflow_depletion_m3d",
                "sum",
            ),
            mean_daily_local_depletion_m3d=(
                "local_total_streamflow_depletion_m3d",
                "mean",
            ),
            minimum_daily_local_depletion_m3d=(
                "local_total_streamflow_depletion_m3d",
                "min",
            ),
            maximum_daily_local_depletion_m3d=(
                "local_total_streamflow_depletion_m3d",
                "max",
            ),
            mean_daily_routed_depletion_m3d=(
                "routed_total_streamflow_depletion_m3d",
                "mean",
            ),
            maximum_daily_routed_depletion_m3d=(
                "routed_total_streamflow_depletion_m3d",
                "max",
            ),
        )
        reaches = reaches.merge(summary, on="reach_id", how="left")

        reach_depletion = daily.groupby("date", sort=True)[
            "local_total_streamflow_depletion_m3d"
        ].sum()
        basin_depletion = (
            basin_results["Unimpaired (Natural)"].sort_values("date")[
                "total_streamflow_m3d"
            ].to_numpy()
            - basin_results["With Pumping"].sort_values("date")[
                "total_streamflow_m3d"
            ].to_numpy()
        )
        max_depletion_error = float(
            np.max(np.abs(reach_depletion.to_numpy() - basin_depletion))
        )
        validation[
            "maximum_daily_depletion_aggregation_error_m3d"
        ] = max_depletion_error
        outlet_reach_id = int(np.flatnonzero(network.downstream_reach < 0)[0]) + 1
        routed_outlet_depletion = daily.loc[
            daily["reach_id"] == outlet_reach_id,
            "routed_total_streamflow_depletion_m3d",
        ].to_numpy()
        max_routed_depletion_error = float(
            np.max(np.abs(routed_outlet_depletion - basin_depletion))
        )
        validation[
            "maximum_routed_outlet_depletion_error_m3d"
        ] = max_routed_depletion_error
        tolerance = 1.0e-8 * max(float(np.max(np.abs(basin_depletion))), 1.0)
        if max_depletion_error > tolerance:
            raise RuntimeError(
                "Reach-local depletion does not reproduce basin depletion; maximum "
                f"daily error is {max_depletion_error:.6g} m3/day."
            )
        if max_routed_depletion_error > tolerance:
            raise RuntimeError(
                "Routed outlet depletion does not reproduce basin depletion; maximum "
                f"daily error is {max_routed_depletion_error:.6g} m3/day."
            )

    gpkg_path = output_dir / "reaches.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    reaches.to_file(gpkg_path, layer="reaches", driver="GPKG", index=False)

    files = {}
    for path in (parquet_path, gpkg_path):
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return {
        "schema_version": REACH_OUTPUT_SCHEMA_VERSION,
        "definition_version": REACH_DEFINITION_VERSION,
        "definition_sha256": network.definition_sha256,
        "reach_count": network.number_of_reaches,
        "daily_row_count": len(daily),
        "date_count": int(daily["date"].nunique()),
        "files": files,
        "validation": validation,
        "spatial_semantics": (
            "Total streamflow only. Local columns exclude upstream inflow. Routed "
            "columns integrate each reach's local total-flow contribution with all "
            "upstream reaches, without channel lag or loss. Neither is well or "
            "pumping-zone attribution."
        ),
    }


def run_scenarios(
    state: GroundwaterState,
    recharge_csv: Path | None,
    pumping_forcing: PumpingForcing | None = None,
    *,
    start_date: str,
    end_date: str,
    config: GroundwaterConfig = GroundwaterConfig(),
    initial_condition: str = "base",
    heads_path: Path | None = None,
    pumped_heads_path: Path | None = None,
    snapshot_dates: Collection[str | pd.Timestamp] | None = None,
    reach_network: ReachNetwork | None = None,
    recharge_raster_manifest: Path | None = None,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, np.ndarray]],
    ReachNetwork,
    dict[str, pd.DataFrame],
]:
    """Run daily scenarios with standard basin and receiving-reach accounting."""
    recharge = load_daily_recharge_forcing(
        recharge_csv=recharge_csv,
        recharge_raster_manifest=recharge_raster_manifest,
        start_date=start_date,
        end_date=end_date,
    )
    requested_snapshots = (
        None
        if snapshot_dates is None
        else {pd.Timestamp(value).normalize() for value in snapshot_dates}
    )
    if requested_snapshots is not None:
        simulation_dates = set(recharge.frame["date"].dt.normalize())
        unavailable = requested_snapshots.difference(simulation_dates)
        if unavailable:
            preview = ", ".join(value.strftime("%Y-%m-%d") for value in sorted(unavailable))
            raise ValueError(f"Snapshot date(s) fall outside the simulation: {preview}")
    if reach_network is None:
        reach_network = build_reach_network(state)
    results: dict[str, pd.DataFrame] = {}
    reach_results: dict[str, pd.DataFrame] = {}
    snapshots: dict[str, dict[str, np.ndarray]] = {"Unimpaired (Natural)": {}}
    scenarios_to_run = ["Unimpaired (Natural)"]
    if pumping_forcing is not None:
        snapshots["With Pumping"] = {}
        if config.strict_pumping_supply:
            scenarios_to_run.insert(0, "With Pumping")
        else:
            scenarios_to_run.append("With Pumping")

    for scenario in scenarios_to_run:
        print(f"Starting scenario: {scenario}")
        scenario_heads_path = (
            pumped_heads_path
            if scenario == "With Pumping" and pumped_heads_path is not None
            else heads_path
        )
        initialize_water_table(
            state,
            "heads" if initial_condition == "heads" else "base",
            config,
            scenario_heads_path,
        )
        groundwater_flux = _GroundwaterFluxAccumulator(reach_network)
        gdp = make_dupuit_component(
            state,
            config,
            callback_fun=groundwater_flux,
            reach_network=reach_network,
        )
        previous_vol = _storage_volume(state)
        initial_storage_m3 = previous_vol
        records = []
        reach_records: list[tuple[pd.Timestamp, dict[str, np.ndarray]]] = []

        for row in tqdm(
            recharge.frame.itertuples(index=False),
            total=len(recharge.frame),
            desc=scenario,
        ):
            if scenario == "With Pumping" and pumping_forcing is not None:
                pump_field = pumping_forcing.flux_for_date(
                    row.date,
                    state=state,
                    source_zone_storage_fraction=config.source_zone_storage_fraction,
                    strict_pumping_supply=config.strict_pumping_supply,
                )
            else:
                pump_field = None
            record, surface_water_depth = _run_daily_step(
                state,
                gdp,
                groundwater_flux,
                recharge_mm_day=recharge.value_for_record(row, state),
                previous_storage_m3=previous_vol,
                pumping_flux=pump_field,
                pumping_storage_fraction=config.pumping_storage_fraction,
                strict_pumping_supply=(
                    config.strict_pumping_supply
                    or (
                        scenario == "With Pumping"
                        and pumping_forcing is not None
                        and pumping_forcing.source_mode == "topographic"
                    )
                ),
                return_surface_water_depth=True,
            )
            record["date"] = row.date
            record["initial_storage_m3"] = initial_storage_m3
            record["scheduled_pumping_m3d"] = (
                pumping_forcing.scheduled_volume_for_date(row.date, state)
                if scenario == "With Pumping" and pumping_forcing is not None
                else 0.0
            )
            records.append(record)
            reach_records.append(
                (
                    row.date,
                    _reach_daily_values(
                        state,
                        reach_network,
                        groundwater_flux,
                        surface_water_depth,
                        record,
                        enforce_stream_availability=(
                            config.stream_loss_mode == "routed_volume_limited"
                        ),
                    ),
                )
            )
            save_snapshot = (
                row.date.normalize() in requested_snapshots
                if requested_snapshots is not None
                else row.date.month in (4, 9) and row.date.is_month_end
            )
            if save_snapshot:
                snapshots[scenario][row.date.strftime("%Y-%m-%d")] = (
                    state.grid.at_node["water_table__elevation"].copy()
                )
            previous_vol = record["storage_m3"]

        results[scenario] = pd.DataFrame(records)
        reach_results[scenario] = _stack_reach_daily_records(
            reach_records, reach_network.number_of_reaches
        )

    return results, snapshots, reach_network, reach_results


def build_depletion_table(natural: pd.DataFrame, pumped: pd.DataFrame) -> pd.DataFrame:
    """Build total-flow depletion and pumping water-balance diagnostics."""
    natural = natural.sort_values("date").reset_index(drop=True)
    pumped = pumped.sort_values("date").reset_index(drop=True)
    if not natural["date"].equals(pumped["date"]):
        raise ValueError("Natural and pumped simulations do not have identical dates.")

    total_depletion = (
        natural["total_streamflow_m3d"].to_numpy()
        - pumped["total_streamflow_m3d"].to_numpy()
    )
    pumping = pumped["pumping_m3d"].to_numpy()
    scheduled_pumping = (
        pumped["scheduled_pumping_m3d"].to_numpy()
        if "scheduled_pumping_m3d" in pumped
        else pumping.copy()
    )
    cumulative_depletion = np.cumsum(total_depletion)
    cumulative_pumping = np.cumsum(pumping)
    cumulative_scheduled_pumping = np.cumsum(scheduled_pumping)
    storage_depletion = natural["storage_m3"].to_numpy() - pumped["storage_m3"].to_numpy()
    if "initial_storage_m3" in natural and "initial_storage_m3" in pumped:
        initial_storage_depletion = float(
            natural["initial_storage_m3"].iloc[0]
            - pumped["initial_storage_m3"].iloc[0]
        )
    else:
        initial_storage_depletion = 0.0
    storage_depletion_change = storage_depletion - initial_storage_depletion
    daily_storage_depletion = np.diff(storage_depletion_change, prepend=0.0)
    modeled_extraction = total_depletion + daily_storage_depletion
    cumulative_modeled_extraction = cumulative_depletion + storage_depletion_change
    pumping_balance_gap = pumping - modeled_extraction
    cumulative_pumping_balance_gap = cumulative_pumping - cumulative_modeled_extraction
    allocation_difference = scheduled_pumping - pumping
    allocation_tolerance = np.maximum(1.0e-8, np.abs(scheduled_pumping) * 1.0e-9)
    allocation_difference[np.abs(allocation_difference) <= allocation_tolerance] = 0.0
    source_capacity_shortfall = np.maximum(allocation_difference, 0.0)
    allocated_pumping = scheduled_pumping - source_capacity_shortfall
    cumulative_source_capacity_shortfall = np.cumsum(source_capacity_shortfall)
    cumulative_allocated_pumping = np.cumsum(allocated_pumping)
    full_schedule_gap = scheduled_pumping - modeled_extraction
    cumulative_full_schedule_gap = (
        cumulative_scheduled_pumping - cumulative_modeled_extraction
    )
    unmodeled_supply_requirement = np.maximum(cumulative_full_schedule_gap, 0.0)

    natural_streamflow = natural["total_streamflow_m3d"].to_numpy()
    pumped_streamflow = pumped["total_streamflow_m3d"].to_numpy()
    impaired_streamflow_fraction = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * pumped_streamflow,
        natural_streamflow,
        out=impaired_streamflow_fraction,
        where=natural_streamflow != 0.0,
    )
    depletion_fraction = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * total_depletion,
        natural_streamflow,
        out=depletion_fraction,
        where=natural_streamflow > 0.0,
    )
    instantaneous_streamflow_response = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * total_depletion,
        pumping,
        out=instantaneous_streamflow_response,
        where=pumping > 0.0,
    )
    instantaneous_storage_response = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * daily_storage_depletion,
        pumping,
        out=instantaneous_storage_response,
        where=pumping > 0.0,
    )
    daily_scheduled_capture = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * total_depletion,
        scheduled_pumping,
        out=daily_scheduled_capture,
        where=scheduled_pumping > 0.0,
    )
    cumulative_scheduled_capture = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * cumulative_depletion,
        cumulative_scheduled_pumping,
        out=cumulative_scheduled_capture,
        where=cumulative_scheduled_pumping > 0.0,
    )
    cumulative_modeled_capture = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * cumulative_depletion,
        cumulative_modeled_extraction,
        out=cumulative_modeled_capture,
        where=cumulative_modeled_extraction > 0.0,
    )
    cumulative_schedule_fulfillment = np.full_like(total_depletion, np.nan, dtype=float)
    np.divide(
        100.0 * cumulative_modeled_extraction,
        cumulative_scheduled_pumping,
        out=cumulative_schedule_fulfillment,
        where=cumulative_scheduled_pumping > 0.0,
    )
    cumulative_source_allocation_fulfillment = np.full_like(
        total_depletion,
        np.nan,
        dtype=float,
    )
    np.divide(
        100.0 * cumulative_allocated_pumping,
        cumulative_scheduled_pumping,
        out=cumulative_source_allocation_fulfillment,
        where=cumulative_scheduled_pumping > 0.0,
    )

    return pd.DataFrame(
        {
            "date": natural["date"],
            "unimpaired_total_streamflow_m3d": natural["total_streamflow_m3d"],
            "pumped_total_streamflow_m3d": pumped["total_streamflow_m3d"],
            "total_streamflow_depletion_m3d": total_depletion,
            "impaired_streamflow_fraction_pct": impaired_streamflow_fraction,
            "streamflow_depletion_fraction_pct": depletion_fraction,
            "instantaneous_streamflow_response_fraction_pct": (
                instantaneous_streamflow_response
            ),
            "instantaneous_storage_response_fraction_pct": (
                instantaneous_storage_response
            ),
            "scheduled_pumping_m3d": scheduled_pumping,
            "allocated_pumping_m3d": allocated_pumping,
            "pumping_m3d": pumping,
            "modeled_extraction_m3d": modeled_extraction,
            "pumping_balance_gap_m3d": pumping_balance_gap,
            "source_capacity_shortfall_m3d": source_capacity_shortfall,
            "full_schedule_gap_m3d": full_schedule_gap,
            "daily_scheduled_capture_fraction_pct": daily_scheduled_capture,
            "daily_capture_fraction_pct": daily_scheduled_capture,
            "cumulative_scheduled_pumping_m3": cumulative_scheduled_pumping,
            "cumulative_allocated_pumping_m3": cumulative_allocated_pumping,
            "cumulative_pumping_m3": cumulative_pumping,
            "cumulative_streamflow_depletion_m3": cumulative_depletion,
            "cumulative_scheduled_capture_fraction_pct": cumulative_scheduled_capture,
            "cumulative_capture_fraction_pct": cumulative_scheduled_capture,
            "cumulative_modeled_extraction_m3": cumulative_modeled_extraction,
            "cumulative_modeled_capture_fraction_pct": cumulative_modeled_capture,
            "cumulative_schedule_fulfillment_pct": cumulative_schedule_fulfillment,
            "cumulative_source_allocation_fulfillment_pct": (
                cumulative_source_allocation_fulfillment
            ),
            "initial_aquifer_storage_depletion_m3": initial_storage_depletion,
            "aquifer_storage_depletion_m3": storage_depletion,
            "aquifer_storage_depletion_change_m3": storage_depletion_change,
            "daily_aquifer_storage_depletion_m3": daily_storage_depletion,
            "cumulative_pumping_balance_gap_m3": cumulative_pumping_balance_gap,
            "cumulative_source_capacity_shortfall_m3": (
                cumulative_source_capacity_shortfall
            ),
            "cumulative_full_schedule_gap_m3": cumulative_full_schedule_gap,
            "cumulative_unmodeled_supply_requirement_m3": unmodeled_supply_requirement,
            "source_balance_error_m3": cumulative_full_schedule_gap,
        }
    )


def _plot_hydrograph_series(axis, dates, streamflow_m3d, **plot_kwargs):
    """Plot streamflow on a log axis while retaining effectively dry days."""
    streamflow = np.asarray(streamflow_m3d, dtype=float)
    if not np.all(np.isfinite(streamflow)):
        raise ValueError("Hydrograph streamflow contains non-finite values.")
    if np.any(streamflow < 0.0):
        raise ValueError("Hydrograph streamflow contains negative values.")
    display_streamflow = np.maximum(streamflow, HYDROGRAPH_DISPLAY_FLOOR_M3D)
    return axis.plot(dates, display_streamflow, **plot_kwargs)


def _format_hydrograph_axis(axis) -> None:
    axis.set_yscale("log")
    axis.set_ylim(bottom=HYDROGRAPH_DISPLAY_FLOOR_M3D)
    axis.set_ylabel("Total streamflow ($m^3$/day; log scale)")
    axis.text(
        0.01,
        0.04,
        "Flows $\\leq$ 1 $m^3$/day shown at the 1 $m^3$/day display floor",
        transform=axis.transAxes,
        fontsize=8,
        color="0.35",
    )


def save_hydrograph_figure(
    natural: pd.DataFrame,
    recharge: pd.DataFrame,
    output_path: Path,
    *,
    start_date: str,
    end_date: str,
    pumped: pd.DataFrame | None = None,
) -> None:
    """Save the basin hydrograph with dry-season flow resolved logarithmically."""
    if pumped is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        _plot_hydrograph_series(
            axes[0],
            natural["date"],
            natural["total_streamflow_m3d"],
            color="green",
            label="Unimpaired",
        )
        axes[0].set_title(f"Hydrograph ({start_date} to {end_date})")
    else:
        depletion = build_depletion_table(natural, pumped)
        monthly_depletion = depletion.groupby(depletion["date"].dt.to_period("M"))[
            "total_streamflow_depletion_m3d"
        ].mean()
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        _plot_hydrograph_series(
            axes[0],
            natural["date"],
            natural["total_streamflow_m3d"],
            color="green",
            label="Unimpaired",
        )
        _plot_hydrograph_series(
            axes[0],
            pumped["date"],
            pumped["total_streamflow_m3d"],
            color="red",
            ls="--",
            label="With pumping",
        )
        axes[0].set_title(f"Hydrographs ({start_date} to {end_date})")
        axes[2].bar(
            monthly_depletion.index.to_timestamp() + pd.Timedelta(days=15),
            monthly_depletion.values,
            width=25,
            color="darkred",
            alpha=0.7,
        )
        axes[2].set_ylabel("Avg. depletion ($m^3$/day)")
        axes[2].set_title("Monthly Total Streamflow Depletion")
        axes[2].grid(alpha=0.3)

    _format_hydrograph_axis(axes[0])
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    axes[1].bar(
        recharge["date"],
        recharge["Recharge"],
        color="blue",
        alpha=0.5,
        width=1.5,
    )
    axes[1].set_ylabel("Recharge (mm/day)")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_depletion_timeseries_figure(
    depletion: pd.DataFrame,
    output_path: Path,
    *,
    start_date: str,
    end_date: str,
) -> None:
    """Save positive daily and cumulative depletion on logarithmic axes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(
        depletion["date"],
        depletion["total_streamflow_depletion_m3d"],
        color="darkred",
        lw=2,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Depletion ($m^3$/day; log scale)")
    axes[0].set_title(f"Daily Streamflow Depletion ({start_date} to {end_date})")
    axes[0].grid(alpha=0.3, which="both")
    axes[1].plot(
        depletion["date"],
        depletion["cumulative_streamflow_depletion_m3"],
        color="red",
        lw=2,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Cumulative depletion ($m^3$; log scale)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_outputs(
    results: dict[str, pd.DataFrame],
    snapshots: dict[str, dict[str, np.ndarray]],
    pumping_forcing: PumpingForcing | None,
    state: GroundwaterState,
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
) -> None:
    """Save explicit streamflow components, water-balance checks, and figures."""
    del state, pumping_forcing
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = f"{start_date}_to_{end_date}"
    natural = results["Unimpaired (Natural)"].copy()
    recharge = natural[["date", "recharge_mm_day"]].rename(
        columns={"recharge_mm_day": "Recharge"}
    )
    natural.to_csv(output_dir / f"simulation_unimpaired_{date_str}.csv", index=False)

    if "With Pumping" not in results:
        save_hydrograph_figure(
            natural,
            recharge,
            output_dir / f"hydrographs_{date_str}.png",
            start_date=start_date,
            end_date=end_date,
        )
    else:
        pumped = results["With Pumping"].copy()
        pumped.to_csv(output_dir / f"simulation_with_pumping_{date_str}.csv", index=False)
        depletion = build_depletion_table(natural, pumped)
        depletion_path = output_dir / f"streamflow_depletion_{date_str}.csv"
        depletion.to_csv(depletion_path, index=False)
        depletion.to_csv(output_dir / "streamflow_depletion_timeseries.csv", index=False)
        monthly_depletion = depletion.groupby(depletion["date"].dt.to_period("M"))[
            "total_streamflow_depletion_m3d"
        ].mean()
        save_hydrograph_figure(
            natural,
            recharge,
            output_dir / f"hydrographs_{date_str}.png",
            start_date=start_date,
            end_date=end_date,
            pumped=pumped,
        )

        response_rates = depletion.set_index("date")[[
            "pumping_m3d",
            "total_streamflow_depletion_m3d",
            "daily_aquifer_storage_depletion_m3",
        ]]
        rolling_response = response_rates.rolling(30, min_periods=1).mean()
        monthly_response = depletion.assign(month=depletion["date"].dt.month).groupby(
            "month"
        )[
            [
                "pumping_m3d",
                "total_streamflow_depletion_m3d",
                "daily_aquifer_storage_depletion_m3",
            ]
        ].sum()
        monthly_stream_fraction = np.full(12, np.nan, dtype=float)
        monthly_storage_fraction = np.full(12, np.nan, dtype=float)
        monthly_pumping = monthly_response["pumping_m3d"].reindex(range(1, 13)).to_numpy()
        np.divide(
            100.0
            * monthly_response["total_streamflow_depletion_m3d"]
            .reindex(range(1, 13))
            .to_numpy(),
            monthly_pumping,
            out=monthly_stream_fraction,
            where=monthly_pumping > 0.0,
        )
        np.divide(
            100.0
            * monthly_response["daily_aquifer_storage_depletion_m3"]
            .reindex(range(1, 13))
            .to_numpy(),
            monthly_pumping,
            out=monthly_storage_fraction,
            where=monthly_pumping > 0.0,
        )

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(12, 18),
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.25, 1.0]},
        )
        axes[0].plot(
            depletion["date"],
            depletion["impaired_streamflow_fraction_pct"],
            color="darkred",
            lw=2,
            label="With-pumping streamflow / unimpaired streamflow",
        )
        axes[0].axhline(
            100.0,
            color="black",
            linestyle=":",
            lw=1.5,
            label="No-impairment reference",
        )
        axes[0].set_ylabel("Impaired / unimpaired flow (%)")
        axes[0].set_title(
            f"Daily Impaired Streamflow Relative to Unimpaired "
            f"({start_date} to {end_date})"
        )
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)
        axes[1].bar(
            monthly_depletion.index.to_timestamp() + pd.Timedelta(days=15),
            monthly_depletion.values,
            width=25,
            color="darkred",
            alpha=0.7,
        )
        axes[1].set_ylabel("Avg. depletion ($m^3$/day)")
        axes[1].set_title("Monthly Total Streamflow Loss")
        axes[1].grid(alpha=0.3)
        for axis in axes[:3]:
            axis.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        axes[0].tick_params(labelbottom=False)
        axes[1].tick_params(labelbottom=False)

        axes[2].plot(
            response_rates.index,
            response_rates["total_streamflow_depletion_m3d"],
            color="red",
            lw=0.7,
            alpha=0.12,
        )
        axes[2].plot(
            response_rates.index,
            response_rates["daily_aquifer_storage_depletion_m3"],
            color="royalblue",
            lw=0.7,
            alpha=0.12,
        )
        axes[2].plot(
            rolling_response.index,
            rolling_response["pumping_m3d"],
            color="black",
            linestyle=":",
            lw=2.5,
            label="Applied pumping (30-day mean)",
        )
        axes[2].plot(
            rolling_response.index,
            rolling_response["total_streamflow_depletion_m3d"],
            color="red",
            lw=2,
            label="Streamflow-depletion response (30-day mean)",
        )
        axes[2].plot(
            rolling_response.index,
            rolling_response["daily_aquifer_storage_depletion_m3"],
            color="royalblue",
            lw=2,
            label="Storage-change response (30-day mean)",
        )
        axes[2].axhline(0.0, color="black", lw=1.0)
        axes[2].set_ylabel("Response rate ($m^3$/day)")
        axes[2].set_title(
            "Instantaneous Pumping Response (daily values faint; storage recovery < 0)"
        )
        axes[2].legend(loc="upper left")
        axes[2].grid(alpha=0.3)

        month_numbers = np.arange(1, 13)
        width = 0.38
        axes[3].bar(
            month_numbers - width / 2,
            monthly_stream_fraction,
            width=width,
            color="red",
            alpha=0.75,
            label="Streamflow-depletion response / pumping",
        )
        axes[3].bar(
            month_numbers + width / 2,
            monthly_storage_fraction,
            width=width,
            color="royalblue",
            alpha=0.75,
            label="Storage-change response / pumping",
        )
        axes[3].axhline(100.0, color="black", linestyle=":", lw=1.25)
        axes[3].axhline(0.0, color="black", lw=1.0)
        axes[3].set_xticks(month_numbers)
        axes[3].set_xticklabels(
            [pd.Timestamp(2000, month, 1).strftime("%b") for month in month_numbers]
        )
        axes[3].set_ylabel("Response / pumping (%)")
        axes[3].set_title(
            "Seasonal Pumping-Response Fractions "
            "(volume-weighted water balance; not tracer source)"
        )
        axes[3].legend(loc="lower center", ncol=2)
        axes[3].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"capture_{date_str}.png", dpi=300)
        fig.savefig(output_dir / f"pumping_response_{date_str}.png", dpi=300)
        plt.close(fig)

        save_depletion_timeseries_figure(
            depletion,
            output_dir / f"depletion_timeseries_{date_str}.png",
            start_date=start_date,
            end_date=end_date,
        )

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
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Topography & Stream Network
    plt.sca(ax[0])
    imshow_grid(grid, "topographic__elevation", plot_name="Topography & Streams", cmap="terrain")
    y_locs = grid.y_of_node[state.stream_indices]
    x_locs = grid.x_of_node[state.stream_indices]
    ax[0].scatter(x_locs, y_locs, s=1, c="blue", alpha=0.5, label="Streams")
    ax[0].scatter(
        grid.x_of_node[state.outlet_node],
        grid.y_of_node[state.outlet_node],
        s=25,
        c="red",
        label="Outlet",
    )
    ax[0].legend()

    # Plot 2: Boundary Conditions
    plt.sca(ax[1])
    status_grid = grid.status_at_node.reshape(grid.shape)
    cmap_bc = plt.get_cmap("viridis", 5)
    extent = [
        grid.x_of_node.min() - state.dx / 2.0,
        grid.x_of_node.max() + state.dx / 2.0,
        grid.y_of_node.min() - state.dy / 2.0,
        grid.y_of_node.max() + state.dy / 2.0,
    ]
    ax[1].imshow(status_grid, cmap=cmap_bc, origin="lower", extent=extent)
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
