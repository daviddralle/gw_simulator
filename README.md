# gw_simulator

`gw_simulator` runs catchment-scale, transient Dupuit–Boussinesq groundwater
simulations with optional pumping. A paired run compares an unimpaired branch
with an otherwise identical pumped branch and reports changes in total modeled
streamflow, aquifer storage, and flow through each extracted stream reach.

The model is intended for screening and sensitivity analysis. It is not a
calibrated groundwater model unless the user supplies and evaluates calibrated
inputs for a particular site.

## Try the model

The synthetic example is self-contained and does not require Earth Engine or
Green Valley data. It generates deterministic inputs and runs a one-year paired
simulation after a one-year unpumped spin-up.

```bash
git clone https://github.com/daviddralle/gw_simulator.git
cd gw_simulator
mamba env create -f environment.yml
mamba activate boussinesq-pumping
bash examples/synthetic_basin/run.sh
```

Results are written to `examples/synthetic_basin/outputs/`. See
[`examples/synthetic_basin/README.md`](examples/synthetic_basin/README.md) for
the input definitions and the parts of the workflow exercised by the run.

The test suite is independent of Earth Engine:

```bash
python -m pytest -q
```

## Green Valley case study

[`examples/green_valley/README.md`](examples/green_valley/README.md) documents
the current 39-reach Green Valley application and provides selected figures and
machine-readable results. That run used the project's continental/legacy
transmissivity, depth-to-bedrock, and specific-yield rasters. The exact site
inputs are not distributed in this repository, so the case study is not the
public execution example.

## Configured workflow

Site applications are defined by a versioned YAML file. Preflight checks verify
input existence, spatial coverage, recharge completeness, requested dates, and
estimated grid size before the model runs.

```bash
python scripts/run_workflow.py --config path/to/site.yml --stage preflight
python scripts/run_workflow.py --config path/to/site.yml --stage groundwater
python scripts/run_workflow.py --config path/to/site.yml --stage plots
```

`--stage all` prepares any configured remote inputs, runs preflight, executes the
model, and makes the configured plots. Existing DEM and hydrogeology rasters are
used without rebuilding them. Earth Engine stages require prior authentication.
The Green Valley configurations require local, untracked site inputs; see
[`configs/README.md`](configs/README.md) before using them.

## Required model inputs

### Watershed and topography

- A watershed polygon readable by GeoPandas, with a defined coordinate system.
- A georeferenced DEM that covers the watershed.

The DEM defines land-surface elevation, D8 flow routing, the stream network, and
the aquifer top. `scripts/01_extract_dem.py` can obtain a 3DEP DEM through Earth
Engine, but a local DEM can be supplied directly.

### Hydrogeology

The groundwater model consumes three georeferenced rasters:

- transmissivity in m²/day;
- depth to bedrock, interpreted as aquifer thickness, in metres; and
- drainable porosity, interpreted as specific yield, as a fraction from zero to
  one.

Pass these paths explicitly in YAML or on the command line. The model infers
hydraulic conductivity as transmissivity divided by modeled aquifer thickness.
Named datasets evaluated for Green Valley are listed in
[`configs/HYDROGEOLOGY_OPTIONS.md`](configs/HYDROGEOLOGY_OPTIONS.md).

### Recharge

Three input modes are supported:

- `earth_engine_deficit`: watershed-mean recharge calculated from PML V2.2a ET,
  PRISM precipitation, and an unbounded storage deficit;
- `csv`: one basin-mean recharge value in mm/day for every modeled and spin-up
  day; or
- `raster_manifest`: one georeferenced spatial recharge field for every day.

Formats and validation rules are in
[`docs/RECHARGE_INPUTS.md`](docs/RECHARGE_INPUTS.md). Pixelwise Earth Engine
deficit extraction is reserved for future implementation; it is not represented
as an available method.

### Pumping

Pumping is optional. A pumping run requires:

- a point layer with a unique identifier such as `APN`; and
- a CSV that joins on that identifier and contains `waterUse_m3Day` or
  `waterUse_m3Month`.

`timeseries` mode uses dated year-month records. `climatology` mode intentionally
repeats mean calendar-month rates in every modeled year. Pumping can be applied
at mapped well cells or allocated within disjoint topographic source zones.

## Flow and depletion accounting

Daily total modeled streamflow is

```text
total_streamflow = groundwater_to_stream + saturation_excess
```

The standard `routed_volume_limited` method routes surface generation and signed
groundwater exchange through the reach network at every adaptive solver substep.
Potential losing-stream exchange is limited by the water available from upstream
flow and local generation. Channel travel time, storage, and reinfiltration are
not represented.

Streamflow depletion is unimpaired total flow minus pumped total flow. Pumping
response is also tracked as the change in aquifer-storage depletion. Reported
tables distinguish scheduled pumping, storage-limited allocated pumping, modeled
extraction recovered from paired balances, and source-capacity shortfall.

Every run writes `reach_daily.parquet` and `reaches.gpkg`. Local reach values
refer to disjoint incremental catchments and may be summed to recover the basin
total. Routed values include upstream contributions and must not be summed across
reaches. The outlet routed value equals basin flow. Full definitions and enforced
checks are in [`docs/REACH_OUTPUTS.md`](docs/REACH_OUTPUTS.md).

## Principal outputs

- daily unimpaired and pumped water-balance CSVs;
- a daily depletion and pumping-response table;
- hydrograph, depletion, pumping-response, and grid figures;
- requested natural and pumped water-table snapshots;
- reach geometry and daily local/routed reach results; and
- JSON metadata containing configuration, input hashes, code revision, output
  hashes, and numerical validation results.

## Main limitations

- The Dupuit approximation represents an unconfined aquifer with predominantly
  horizontal groundwater flow.
- Stream cells are fixed-head boundaries, subject to the routed water-availability
  limit for losing exchange.
- Reach routing is instantaneous.
- Results depend strongly on transmissivity, aquifer thickness, specific yield,
  recharge, and the representation of pumping sources.
- Reach accounting identifies where the model expresses a flow change; it does
  not attribute that change to a particular well.
- A completed numerical run is not evidence that site parameters are calibrated.

## Development

Python 3.11 or newer is required. Runtime dependencies are declared in
`pyproject.toml`; `environment.yml` also includes Earth Engine, plotting, testing,
and video dependencies used by the complete workflow.
