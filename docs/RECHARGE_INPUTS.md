# Recharge inputs

The groundwater model accepts either one basin-mean recharge rate per day or a
spatial recharge raster for every modeled day. Recharge is always applied as a
nonnegative rate over active aquifer cells. Stream cells and closed cells do not
receive recharge.

## Default: Earth Engine storage-deficit recharge

If `recharge.source` is omitted, the workflow uses
`earth_engine_deficit`. It extracts watershed-mean PML V2.2a
evapotranspiration and PRISM precipitation from Google Earth Engine, expands the
PML composites to daily rates, and applies the unbounded storage-deficit
calculation locally. The default forcing period covers the configured spin-up
and simulation dates.

```yaml
recharge:
  # source: earth_engine_deficit  # optional; this is the default
  initial_deficit_mm: 0.0

earth_engine:
  project: your-ee-project
```

When output paths are omitted, the workflow writes the daily table and its
cache below `data/forcing/<workflow name>/`. Run the normal workflow after
authenticating Earth Engine once:

```bash
python scripts/run_workflow.py --config configs/my_basin.yml --stage all
```

The generated CSV contains `date`, `ET`, `P`, `Deficit`, and `Recharge`; all
water-balance quantities are in millimeters or millimeters per day as
appropriate.

## User-supplied basin-mean time series

Set the source to `csv` and provide a daily file:

```yaml
recharge_csv: data/my_basin/recharge.csv

recharge:
  source: csv
```

The CSV must contain exactly one record per requested day, including spin-up,
with these columns:

```csv
date,Recharge
2020-01-01,0.0
2020-01-02,1.37
```

`Recharge` is a finite, nonnegative basin-mean rate in mm/day. Dates must be
unique and complete; the workflow never fills or carries missing values.

## User-supplied spatial raster time series

Set the source to `raster_manifest`:

```yaml
recharge_raster_manifest: data/my_basin/recharge_manifest.csv

recharge:
  source: raster_manifest
```

The manifest has one row per day:

```csv
date,raster_path,band,units
2020-01-01,recharge_2020.tif,1,mm/day
2020-01-02,recharge_2020.tif,2,mm/day
```

- `date` and `raster_path` are required.
- `raster_path` may be absolute or relative to the manifest.
- `band` is one-based and defaults to 1. Repeating one GeoTIFF path with
  successive bands is a compact way to supply a single multiband raster time
  series; separate daily rasters also work.
- `units` defaults to `mm/day`. Supported values are `mm/day`, `mm/d`, `m/day`,
  `m/d`, and `m/s`.
- Every raster must have a CRS and cover the modeled domain. Values are
  area-average reprojected to the groundwater grid. Missing, infinite, or
  negative values at active aquifer cells stop the run.

The daily output column `recharge_mm_day` remains the area-weighted basin mean,
and `recharge_m3d` is the volume obtained by integrating the spatial field over
active cells. The groundwater solver, reach accounting, and mass balance use
the spatial node field itself.

## Future Earth Engine spatial-deficit extraction

`earth_engine_spatial_deficit` is reserved as a future source name but is not
implemented. A scientifically consistent implementation must retain the
storage-deficit state independently at every pixel through time; calculating a
deficit from basin-averaged precipitation and ET and then redistributing it is
not equivalent. A future extractor should:

1. reconcile the different PML and PRISM spatial supports on a documented grid;
2. carry the pixelwise antecedent deficit and initial-condition field across
   export chunks;
3. export chunked multiband GeoTIFFs or another georeferenced time-cube format
   plus the same daily manifest used by the current solver;
4. cache source and derived chunks with dates, CRS, resolution, masks, and
   hashes; and
5. compare the spatially integrated forcing with an independently reduced
   basin water balance before a groundwater run begins.

This keeps the solver-facing contract stable: future Earth Engine spatial
products can use `raster_manifest` without another change to the groundwater
model.
