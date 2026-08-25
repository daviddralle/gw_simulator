# Workflow configurations

The public, executable configuration is
[`../examples/synthetic_basin/config.yml`](../examples/synthetic_basin/config.yml).
Its inputs are generated locally by the example script.

The Green Valley YAML files in this directory are provenance records and working
site configurations. They refer to DEMs, hydrogeology rasters, well locations,
pumping schedules, recharge caches, or restart arrays that are intentionally not
tracked. They will not run from a clean clone.

- `green_valley.yml`: 15.836 km², 39-reach continental/legacy case study.
- `green_valley_domain.yml`: larger Green Valley groundwater-model domain.
- `green_valley_domain_plus50m_preserve_k.yml`: additional-depth sensitivity.
- `green_valley_continental_smoke.yml`: short continental-input solver test.
- `green_valley_sonoma_county_smoke.yml`: short Sonoma County-input solver test.
- `green_valley_stream_limited_test.yml`: routed stream-availability comparison.

Each site run should retain its exact configuration and the generated simulation
metadata. Do not replace rasters behind a shared filename; use a new named
configuration for each hydrogeologic combination.
