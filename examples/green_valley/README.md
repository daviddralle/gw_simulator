# Green Valley 24-Year Simulation

This example demonstrates how to use `gw_simulator` to run a comprehensive 24-year transient groundwater simulation (from Water Year 2001 through Water Year 2024) for the Green Valley watershed.

## Requirements

Before running this example, ensure that the `data/` directory in the repository root is fully populated. Specifically, it must contain:

- `drainage_area_10m_dem_clipped.tif`: The clipped 3DEP topography.
- `comid_8273277.gpkg`: The catchment boundary.
- `daily_water_balance_full.csv`: The pre-computed 24-year daily recharge forcing time series.
- `well_locations.gpkg`: Point locations of groundwater wells in the catchment.
- `GVDB/pumpingSchedule.csv`: Monthly pumping schedules associated with the wells.
- `GLYMPHS/`: The folder containing `transmissivity_m2d.tif`, `depthToBedrock_m.tif`, and `storativity.tif`.

*(Note: These datasets are pre-included in the repository data folder by default for the Green Valley site).*

## Running the Example

You can execute the entire 24-year simulation by running the provided shell script:

```bash
bash run.sh
```

### What this script does:

1. **Executes `03_run_groundwater.py`**:
   - Routes the daily recharge from `2000-10-01` to `2024-09-30`.
   - Spins up the steady-state aquifer (since `--skip-spinup` is omitted).
   - Simulates two parallel scenarios: an **Unimpaired (Natural)** run, and a **With Pumping** run.
   - Calculates the streamflow depletion caused by the pumping by comparing basin-scale discharge between the two scenarios.
   - Saves daily mass balance summaries, depletion time series, and comprehensive diagnostic plots (hydrographs and capture fraction) to the `outputs/gv_24_year/` directory.

2. **Executes `04_plot_cross_sections.py`**:
   - Takes the final day's water table snapshot (`2024-09-30`).
   - Automatically finds the location of maximum depletion in the grid.
   - Generates a 2D East-West cross-section across the watershed, plotting the topography, bedrock, natural water table, and the final pumped depletion cone.

## Outputs

All outputs, including CSVs and `.png` plots, will be saved relative to the project root in:
`outputs/gv_24_year/`
