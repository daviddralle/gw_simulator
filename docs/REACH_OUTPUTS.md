# Reach flow-depletion outputs

Every groundwater run builds a fine D8 reach network from the same stream
threshold used for the fixed-head groundwater boundary. The network and daily
values are standard outputs; there is no separate distributed-analysis mode.

## Spatial meaning

A reach is a maximal stream chain between a headwater, confluence, and the
outlet. Each active aquifer cell belongs to exactly one incremental catchment:
the reach containing that cell's first downstream stream node.

The public reach products report **total modeled flow only**. They do not split
depletion into groundwater and saturation-excess pathways.

- **Local total flow** is the stream contribution generated along a reach and
  within its disjoint incremental catchment. It excludes upstream inflow.
- **Routed total flow** is that local contribution plus the contributions from
  every upstream reach. In the standard limited solver it is constrained to be
  nonnegative at every adaptive substep and again checked when daily products are
  written. Routing is instantaneous and does not simulate channel lag or storage.
- **Flow depletion** is unimpaired total flow minus pumped total flow. Values are
  not clipped, so a negative local value remains a valid redistribution signal.

Summing local values across all reaches reproduces the basin daily flow. The
routed value at the single outlet also reproduces the basin daily flow.

This is receiving-reach accounting, not tracer or pumping-source attribution. It
does not claim that pumping in an incremental catchment caused depletion in that
same reach, and groundwater may cross surface-drainage boundaries.

## Files and schema

`reach_daily.parquet` contains one row per date and reach, compressed with
Zstandard. Schema version `2.0.0` contains:

- `date`, `reach_id`
- `unimpaired_local_total_streamflow_m3d`
- `pumped_local_total_streamflow_m3d`
- `local_total_streamflow_depletion_m3d`
- `local_streamflow_depletion_fraction_pct`
- `routed_unimpaired_total_streamflow_m3d`
- `routed_pumped_total_streamflow_m3d`
- `routed_total_streamflow_depletion_m3d`
- `routed_streamflow_depletion_fraction_pct`

`reaches.gpkg`, layer `reaches`, contains reach lines, downstream reach ID,
outlet flag, stream-node count, reach length, incremental and upstream area, and
total-flow summary metrics. It joins to the Parquet on `reach_id`.

The Parquet metadata and run `simulation_metadata_*.json` record the schema and
reach-definition versions, node/reach assignment digest, file sizes and hashes,
and basin-closure errors. The reach algorithm version is
`d8-maximal-stream-chain-v1`.

## Invariants

The workflow stops rather than writing ambiguous distributed results if:

- a core aquifer cell or open groundwater boundary cannot be assigned;
- the reach graph cycles or has anything other than one outlet;
- incremental reach areas do not sum to the modeled aquifer area;
- daily local totals do not sum to the basin result; or
- daily routed outlet totals do not equal the basin result.

In availability-limited mode it also stops if post-integration routing needs more
than 0.001 m3/day of correction or if any routed daily flow is below numerical
tolerance. Floating-point negatives smaller than 1e-10 m3/day are written as zero.

No daily GeoTIFFs are written. The 37-reach, 5,114-day continental Green Valley
run contains 189,218 rows.

## Standard maps and videos

The `plots` stage reads the compact reach files and does not rerun the solver. It
writes:

- `reach_depletion_summary.png`: cumulative local depletion, cumulative routed
  depletion, full-period routed depletion fraction, and June–October routed
  depletion fraction;
- `reach_routed_depletion_fraction_wy*.mp4`: weekly frames for the driest and
  wettest complete modeled water years, unless years are pinned in the YAML; and
- `reach_visualization_metadata.json`: definitions, parameters, hashes, selected
  years, and display limits.

Each animated link shows the trailing 30-day volume ratio

```text
100 * sum(routed unimpaired total flow - routed pumped total flow)
    / sum(routed unimpaired total flow)
```

Thus every link displays the depletion fraction of the total flow passing that
link under this instantaneous routing convention. The basin value in the title
is the outlet link, not a sum of routed links (which would double count upstream
water).

Fractions are undefined where the rolling unimpaired routed volume is at most
1 m3. Full values remain in the Parquet; animation colors alone are capped at
plus/minus 100 percent. Rolling window, frame step, frame rate, and optional water
years are configurable under `outputs` in the YAML.
