# Hydrogeology input options

The model consumes three independent raster inputs. Keep each experiment's exact
combination in a named workflow YAML rather than overwriting shared rasters.

| Variable | Option | Path | Model units |
| --- | --- | --- | --- |
| Transmissivity | GLHYMPS 2.0 + Pelletier depth (constructed alternative) | `data/derived/green_valley/glhymps2/transmissivity_glhymps2_pelletier2016_m2day.tif` | m2/day |
| Transmissivity | Sonoma County merged | `data/hydrogeology/sonoma_county/transmissivity_sonoma_county_m2day.tif` | assumed m2/day |
| Transmissivity | Continental (current starting option) | `data/hydrogeology/continental/transmissivity_continental_m2day.tif` | m2/day |
| Specific yield | GLHYMPS 2.0 total-porosity proxy (constructed alternative) | `data/derived/green_valley/glhymps2/specific_yield_glhymps2_total_porosity.tif` | dimensionless |
| Specific yield | Sonoma County merged (`S`) | `data/hydrogeology/sonoma_county/specific_yield_sonoma_county.tif` | dimensionless |
| Specific yield | Continental (current starting option) | `data/hydrogeology/continental/specific_yield_continental.tif` | dimensionless |
| Depth to bedrock | Pelletier 2016 landform mosaic (constructed alternative) | `data/derived/green_valley/glhymps2/depth_to_unweathered_bedrock_pelletier2016_m.tif` | m |
| Depth to bedrock | Shangguan 2017 | `data/derived/green_valley/glhymps2/depth_to_bedrock_shangguan2017_m.tif` | m |
| Depth to bedrock | Continental (current starting option) | `data/hydrogeology/continental/depth_to_bedrock_continental_m.tif` | m |

The command-line/config key remains `porosity` for compatibility, but the solver
interprets that raster as drainable porosity (specific yield). The Sonoma source
files contain no provenance or unit metadata beyond their filenames, so confirm
the transmissivity units and intended meaning of `S` before treating them as more
than sensitivity inputs.

The source file hashes copied on 2026-08-14 are:

- `specific_yield_sonoma_county.tif`: `7c46e63044adb065fa61171db6f1d10d5539a16741825b39cc5ac5866ebc7e78`
- `transmissivity_sonoma_county_m2day.tif`: `11de7841352f9bde55e8618f9ca06ffecc31e90049ef74393b2ba1ddf8a4dde8`

The continental files supplied on 2026-08-14 are byte-identical to the former
`data/GLYMPHS/` inputs. Their hashes are:

- `depth_to_bedrock_continental_m.tif`: `1e23c7ea9eca353515918d8240fbbcda650f30202c46e11c71279817b75b6820`
- `specific_yield_continental.tif`: `69c1952f071e81c76d165549ace0e28439444c2e0bd07d4b0bc5ab07b37baf23`
- `transmissivity_continental_m2day.tif`: `e466dfc4d0596f155d34854dcfc1dec86e43c8edea89833083dc395cefb7cfed`

Run the short, initially dry smoke test with:

```bash
python scripts/run_workflow.py \
  --config configs/green_valley_sonoma_county_smoke.yml \
  --stage preflight
python scripts/run_workflow.py \
  --config configs/green_valley_sonoma_county_smoke.yml \
  --stage groundwater
```

## Smoke-test result (2026-08-14)

The three-day paired run completed from an initially dry aquifer without a solver
or dry-cell failure. Its water-balance error was zero at reported precision. The
very small Sonoma S values constrained available drainable storage: 191.42 m3 of
437.71 m3 scheduled pumping was allocated (43.73%), and 246.29 m3 was recorded as
source-capacity shortfall. Results are in
`runs/green_valley/sonoma_county_st_smoke/groundwater/`.

The equivalent continental S/T/depth smoke test also completed without a solver
or dry-cell failure and closed its paired water balance to machine precision. It
allocated the full 145.90 m3/day after the initially dry first day: 291.81 m3 of
437.71 m3 over the three-day test (66.67%). Of that extraction, 87.17 m3 appeared
as streamflow depletion and 204.64 m3 remained as aquifer-storage depletion at the
end of the test. Results are in
`runs/green_valley/continental_st_depth_smoke/groundwater/`.

The full 2010-10-01 through 2024-09-30 continental run completed on 2026-08-14.
All 3,350,643 m3 of scheduled pumping was allocated, cumulative streamflow
depletion was 3,290,227 m3 (98.20% capture), and end-of-run storage depletion was
60,416 m3. No source-capacity shortfall occurred. See `COLLABORATOR_METHODS.md`
for the complete method, QA summary, results, and interpretation limits.
