# gw_simulator

`gw_simulator` is a Python package for simulating catchment-scale groundwater flow and streamflow depletion using the Landlab Dupuit-Boussinesq component. It extracts topography from Earth Engine, calculates recharge, and runs paired unimpaired and pumped scenarios for a watershed boundary.

## Reproducible Workflow

The recommended entry point is a versioned YAML config. It keeps the watershed,
forcing period, hydrogeology, pumping assumptions, spinup, and requested outputs
together and runs a preflight check before the expensive model stage:

```bash
eval "$(mamba shell hook --shell zsh)"
mamba activate lab
python scripts/run_workflow.py --config configs/green_valley.yml --stage all
```

Individual stages can be resumed with `--stage dem`, `hydrogeology`, `recharge`,
`preflight`, `groundwater`, or `plots`. PML/PRISM requests are cached by watershed
geometry and year. Use `--refresh-forcing` only when the remote products must be
queried again.

For the current Green Valley dataset choice and a concise collaborator-facing
description of recharge, groundwater, pumping, dry-cell handling, flow accounting,
outputs, and limitations, see `COLLABORATOR_METHODS.md`. Named S/T/depth alternatives
and file hashes are listed in `configs/HYDROGEOLOGY_OPTIONS.md`.

## Overview of the Workflow

To run a simulation for a new site, you must complete four main steps using the scripts provided in the `scripts/` directory:

1. **Extract DEM**: Download and clip a 3DEP DEM for your catchment.
2. **Compute Recharge**: Generate a daily time series of recharge forcing.
3. **Run Groundwater**: Simulate the physical groundwater system.
4. **Plot Cross Sections**: (Optional) Visualize the spatial distribution of the water table and depletion cone.

---

## Input Data Formats

To apply this model to your own watershed, you will need to prepare specific inputs.

### 1. Catchment Boundary
- **Format**: Any vector format supported by GeoPandas (e.g., `.gpkg`, `.shp`, `.geojson`, `.kml`).
- **Usage**: Used to clip the DEM, define the active modeling domain, and mask well locations.

### 2. Hydrogeology Data
- **Format**: `.tif` rasters.
- **Usage**: The groundwater model consumes three aligned rasters:
  - Transmissivity (`transmissivity_m2d.tif`)
  - Depth to Bedrock (`depthToBedrock_m.tif`)
  - Drainable porosity / specific yield (`storativity.tif` in the Green Valley data)
- **Official baseline**: `scripts/00_prepare_hydrogeology.py` rasterizes GLHYMPS
  2.0 total porosity and permeability, then combines its hydraulic conductivity
  with Pelletier et al. (2016) landform-specific thickness. Upland pixels use
  regolith thickness to unweathered bedrock; lowland pixels use sedimentary-deposit
  thickness. GLHYMPS total porosity is used as a screening-level specific-yield
  proxy, not as a measured drainable porosity.
- **Sensitivity data**: Optional Shangguan et al. (2017) SoilGrids depth and legacy
  hydrogeology rasters can be supplied for comparison, but they are not required
  and never affect the official baseline. Pelletier is about 1 km and its regolith
  layer is explicitly experimental, so thickness and conductivity remain
  first-order model uncertainties.

### 3. Recharge Time Series
- **Default**: omit `recharge.source` (or set it to
  `earth_engine_deficit`) to build watershed-mean recharge from PML V2.2a ET,
  PRISM precipitation, and the storage-deficit method through Earth Engine.
- **Basin time series**: set `recharge.source: csv` and provide a complete daily
  `.csv` with `date` and `Recharge` in **mm/day**.
- **Spatial time series**: set `recharge.source: raster_manifest` and provide a
  daily manifest of georeferenced raster paths, with optional band and units.
  This applies spatial recharge directly to groundwater-grid nodes and is the
  recommended external-input structure for larger basins.
- **Full contract**: see [docs/RECHARGE_INPUTS.md](docs/RECHARGE_INPUTS.md) for
  formats, validation, examples, and the planned Earth Engine spatial-deficit
  pathway.

### 4. Pumping Data (Optional)
Pumping is fully optional. If omitted, the model will run a purely natural (unimpaired) scenario. If included, you must provide two files:

**Well Locations:**
- **Format**: `.gpkg` or `.shp` point vector file.
- **Required Columns**: Must contain a unique identifier column, typically `APN` (Assessor's Parcel Number).

**Pumping Schedule:**
- **Format**: `.csv` file.
- **Required Columns**:
  - `Month`: Integer from `1` to `12` representing the calendar month.
  - `APN`: The unique identifier matching the Well Locations file.
  - *Either* `waterUse_m3Day` (average cubic meters per day for that month) or `waterUse_m3Month` (total cubic meters per month).
  - For the default `timeseries` mode, `Date` or `Year` is also required. The actual year-month records are used without averaging across years. Pumping is zero outside the schedule coverage, while missing months inside the coverage are errors.

---

## Step-by-Step Walkthrough

### Step 1: Extract the DEM
Run the DEM extraction script to pull high-resolution topography from Google Earth Engine. Note that you must have your Earth Engine environment initialized (`earthengine authenticate`).

```bash
python scripts/01_extract_dem.py \
    --boundary path/to/your_catchment.gpkg \
    --output-tif data/my_site_dem.tif
```

### Step 2: Compute Recharge
Generate the daily recharge CSV for your simulation period.

```bash
python scripts/02_compute_recharge.py \
    --boundary path/to/your_catchment.gpkg \
    --output-csv data/my_site_recharge.csv \
    --start-year 2000 \
    --end-year 2024 \
    --cache-dir data/forcing/my_site/pml_v22a_prism
```

This uses PML V2.2a eight-day mean daily ET, PRISM daily precipitation, and the
unbounded storage-deficit method. Composite mean ET is held over each documented
eight-day interval, preserving its water volume. Source chunks, daily forcing,
the water-balance table, and a provenance JSON are all written locally.
The YAML workflow runs this Earth Engine deficit method by default. To bypass
Earth Engine, select `csv` or `raster_manifest` and point the config at the
user-supplied forcing described in [the recharge input contract](docs/RECHARGE_INPUTS.md).

### Step 3: Run the Groundwater Simulation
Run the Dupuit-Boussinesq model. The script aligns the hydrogeologic rasters, extracts fixed-head stream nodes, performs a two-year transient spin-up immediately before the requested simulation by default, and uses those spin-up heads for the main run.

**For an Unimpaired (Natural) Run:**
```bash
python scripts/03_run_groundwater.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --recharge-csv data/my_site_recharge.csv \
    --start-date 2022-10-01 \
    --end-date 2023-09-30
```
Pass `--transmissivity`, `--depth-to-bedrock`, and `--porosity` explicitly, or use
the YAML runner so a production run cannot silently fall back to the legacy files.

**For a Run with Pumping:**
To evaluate streamflow depletion, pass the optional `--wells` and `--pumping-schedule` arguments. The script will automatically run *both* scenarios and generate comparative hydrographs.

```bash
python scripts/03_run_groundwater.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --recharge-csv data/my_site_recharge.csv \
    --wells path/to/my_wells.gpkg \
    --pumping-schedule path/to/my_pumping.csv \
    --pumping-mode timeseries \
    --pumping-source-mode topographic \
    --pumping-source-area-threshold 500000 \
    --start-date 2022-10-01 \
    --end-date 2023-09-30
```

Use `--pumping-mode climatology` only when intentionally repeating the mean rate for each calendar month across every simulation year.

Topographic pumping allocates each zone's daily demand across its currently
saturated cells, weighted by transmissive capacity and capped at half of each
cell's drainable storage by default. The zones are disjoint catchments of reaches
on a coarser D8 channel network. `--pumping-source-area-threshold` controls that
source network independently of the finer fixed-head stream boundary; it must be
at least `--stream-area-threshold`. If a zone cannot provide all reported demand,
the normal run clips pumping to available capacity and records the omitted volume.
`--strict-pumping-supply` instead stops at the first shortfall.

For a localized fractured-rock sensitivity, `--well-additional-depth` lowers the
aquifer base only at exact 50 m cells containing mapped pumping wells. It does not
spread pumping to neighboring cells and leaves the basin-wide link conductivity
field unchanged. Use it with `--strict-pumping-supply` when testing whether a local
well column can supply the complete schedule:

```bash
python scripts/03_run_groundwater.py \
    ... \
    --pumping-mode climatology \
    --well-additional-depth 25 \
    --strict-pumping-supply
```

### Streamflow Accounting

The canonical daily streamflow is the net water contribution from the modeled catchment:

```text
total_streamflow_m3d = groundwater_to_stream_m3d + saturation_excess_m3d
```

- `groundwater_to_stream_m3d` is the signed groundwater exchange integrated over all adaptive solver substeps. Positive values discharge to streams; negative values indicate stream loss to the aquifer.
- In the standard `routed_volume_limited` mode, `saturation_excess_m3d` is the
  solver-integrated local surface-water generation plus any sub-milliliter-scale
  routing correction needed to remove accumulated roundoff. In the explicit
  legacy `unlimited_fixed_head` mode it remains the balance-closing remainder.
- `landlab_saturation_excess_m3d` retains Landlab's adaptive-substep surface-flux
  estimate. Its difference from the closed term is reported in
  `landlab_surface_flux_integration_error_m3d` as a numerical diagnostic.
- `groundwater_discharge_m3d` and `stream_loss_to_groundwater_m3d` store the two nonnegative directions of groundwater exchange for convenience.
- `mass_balance_streamflow_m3d` and `mass_balance_error_m3d` are diagnostics. They are not substituted for the explicitly calculated streamflow.

Natural and pumped scenario CSVs retain internal pathway terms for water-balance
auditing. All depletion tables, figures, maps, videos, and headline metrics report
only total flow depletion. The standard solver prevents physical routed flow from
becoming negative; depletion itself is not clipped.

At every adaptive groundwater substep, local surface generation and signed
groundwater/stream exchange are aggregated by reach and routed headwater to outlet.
If potential losing-stream exchange exceeds incoming plus locally generated water,
that unavailable loss is rejected before aquifer storage is updated. Gaining-stream
exchange is unrestricted. Diagnostics record rejected potential loss, limiter
iterations, dry reaches, solver surface-flux clipping, and the final daily routing
roundoff correction. `--stream-loss-mode unlimited_fixed_head` is retained only for
controlled legacy comparisons.

Every run also performs reach accounting on the fine fixed-head stream network.
Each active aquifer cell is assigned to the reach containing its first downstream
stream node. The compressed `reach_daily.parquet` stores both the total flow
generated locally and the total integrated through that link from all upstream
reaches; `reaches.gpkg` supplies the joinable network geometry. This locates where
the model expresses depletion but does not attribute it to wells or pumping
zones. See [docs/REACH_OUTPUTS.md](docs/REACH_OUTPUTS.md) for the schema, spatial
definition, and enforced local-sum and routed-outlet checks.

The standard `plots` stage also makes one four-panel network summary and two
short water-year MP4s of trailing 30-day upstream-integrated total-flow depletion
fraction. By default it
chooses the driest and wettest complete water years by modeled recharge; specify
`outputs.reach_video_water_years` in YAML to pin particular years. These are
network-wide products, not a separate outlet-style figure for every reach.

With `--pumping-source-mode well_cell`, pumping is imposed as negative recharge at
mapped well nodes. Topographic mode instead imposes the allocated sink across each
bounded source zone. The depletion table distinguishes reported schedule,
source-zone allocation, and the extraction recovered from the paired model water
balances. Any source-capacity shortfall makes the simulated depletion a lower-bound
estimate for the full reported schedule, conditional on the model structure. It is
not a simulated deep-aquifer flux.

Each run also writes a dated `simulation_metadata_*.json` containing the grid
configuration, input paths and hashes, spin-up period, initial condition, pumping
coverage, outlet location, canonical streamflow definition, reach schema and
definition versions, reach-assignment digest, output hashes, and distributed-to-
basin validation errors. It also records the Git commit/dirty flag when available
and SHA-256 hashes of the groundwater module and workflow entry points, so an
uncommitted development run can still be tied to the exact source files used.

### Modeling Assumptions

- Stream stage is still represented by internal fixed-head nodes, but standard
  losing exchange is capped by routed stream-water availability at every adaptive
  substep. Surface water is delivered immediately, without channel travel time,
  channel storage, or reinfiltration.
- The groundwater component uses the Dupuit approximation for an unconfined aquifer. The supplied porosity raster is interpreted as drainable porosity (specific yield), not confined elastic storativity.
- GLHYMPS porosity is a total-porosity proxy rather than an independently calibrated specific-yield field. `--specific-yield-floor` is available only as an explicit whole-domain sensitivity and leaves the input raster preserved in a separate model field.
- Hydraulic conductivity is inferred as transmissivity divided by depth to bedrock. The depth-to-bedrock raster therefore defines the modeled aquifer thickness. `--additional-aquifer-depth` is zero by default and should be treated as a basin-wide sensitivity test because changing thickness also changes storage and, depending on `--deep-aquifer-hydraulics`, conductivity or transmissivity.
- `--well-additional-depth` is a separate local sensitivity. It extends only mapped pumping cells and preserves their existing conductivity and specific yield. A successful strict run therefore means the existing hydrogeologic properties can transmit and store the full scheduled demand within those deeper columns; it does not validate those properties independently.
- The Green Valley example uses spatially uniform basin-mean recharge. Other
  runs may supply a daily raster manifest; the solver then applies the aligned
  spatial field directly. The default storage-deficit calculation is a climatic
  cumulative deficit with no imposed maximum bucket depth.
- Production run metadata includes SHA-256 hashes of the model rasters, boundary,
  recharge, pumping inputs, and YAML config so results can be tied to exact inputs.
- `--snapshot-date` may be repeated to save only selected head arrays. This avoids
  unnecessary storage when scaling to a larger watershed.
- Negative `groundwater_to_stream_m3d` values remain valid internal signed exchange,
  but the routed total flow through every reach is nonnegative in the standard mode.

### Step 4: Visualize Depletion Cross Sections
If you want to view a 2D cross section of the aquifer showing the physical depletion cone at the end of the dry season, run the plotting script:

```bash
python scripts/04_plot_cross_sections.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --date 2023-09-30
```
*(This script will automatically detect if a pumped scenario was run. If no pumping was provided, it will simply plot the natural groundwater table).*
