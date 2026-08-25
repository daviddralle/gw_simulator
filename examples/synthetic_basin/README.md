# Synthetic basin quickstart

This example runs the complete paired unimpaired/pumped workflow without Earth
Engine credentials or unpublished field data. The input builder creates a small
synthetic basin, four model rasters, one synthetic well, a monthly pumping
climatology, and daily recharge. These inputs are for software testing and have
no Green Valley interpretation.

From the repository root, with the project environment active, run:

```bash
bash examples/synthetic_basin/run.sh
```

The script regenerates the inputs, runs preflight checks, performs a one-year
unpumped spin-up, simulates paired conditions for calendar year 2020,
and writes figures and machine-readable results to `outputs/`. The run exercises:

- raster alignment and watershed masking;
- D8 stream extraction and reach construction;
- transient spin-up;
- monthly pumping forcing;
- paired flow and storage accounting;
- routed stream-water availability limits;
- reach-level Parquet and GeoPackage output; and
- a final water-table cross section.

The final check verifies complete dates, finite results, allocation of the
synthetic pumping schedule, paired flow/storage closure, daily mass balance, and
nonnegative routed flow.

`inputs/` and `outputs/` are ignored because they are regenerated locally. The
complete input definitions are in `build_inputs.py`; the numerical assumptions
are in `config.yml`.

This is an execution example, not a calibrated catchment model. Use the Green
Valley case study for the form of the scientific outputs produced by a longer
site application.
