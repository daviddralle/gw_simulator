# Green Valley example data

These files are the public, collaborator-facing products from the paired Green
Valley simulation for 2010-10-01 through 2024-09-30. The run uses the current
`routed_volume_limited` stream-loss method, a 50 m grid, and a 39-reach network.

## Primary products

- `streamflow_depletion_2010-10-01_to_2024-09-30.csv`: daily outlet comparison
  with unimpaired and pumped streamflow, depletion, pumping, aquifer-storage
  response, and cumulative fields.
- `reach_daily.parquet`: one row per date and reach (199,446 rows), with local
  and upstream-routed unimpaired flow, pumped flow, depletion, and depletion
  fraction.
- `reaches.gpkg`: network geometry and summary attributes, joined to the Parquet
  by `reach_id`; `reach_id = 39` is the outlet.

## Summaries and provenance

- `basin_dry_season_metrics_by_year.csv`: June–October basin metrics by year.
- `reach_dry_season_metrics_by_year.csv`: June–October metrics by reach and year.
- `reach_dry_season_summary.csv`: full-period reach summary.
- `simulation_metadata_2010-10-01_to_2024-09-30.json`: run configuration, input
  and output hashes, dates, definitions, and validation results.
- `preflight.json`: pre-run configuration checks.
- `reach_visualization_metadata.json` and `dry_season_metrics_metadata.json`:
  definitions and provenance for the mapped and seasonal summaries.

See [`../REACH_OUTPUTS.md`](../REACH_OUTPUTS.md) for the spatial meaning of local
and routed reach quantities. The values are screening-model results and have not
been calibrated to observed groundwater levels or streamflow.
