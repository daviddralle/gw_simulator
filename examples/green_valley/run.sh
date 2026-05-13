#!/bin/bash
# run.sh
# This script runs the full 24-year groundwater simulation for the Green Valley watershed.

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the simulation from the project root directory
cd ../..

echo "Starting 24-year Green Valley Simulation..."

python scripts/03_run_groundwater.py \
    --dem data/drainage_area_10m_dem_clipped.tif \
    --boundary data/comid_8273277.gpkg \
    --recharge-csv data/daily_water_balance_full.csv \
    --wells data/well_locations.gpkg \
    --pumping-schedule data/GVDB/pumpingSchedule.csv \
    --start-date 2000-10-01 \
    --end-date 2024-09-30 \
    --output-dir outputs/gv_24_year

echo "Simulation complete! Outputs are saved in 'outputs/gv_24_year/'"

echo "Generating depletion cross section..."
python scripts/04_plot_cross_sections.py \
    --dem data/drainage_area_10m_dem_clipped.tif \
    --boundary data/comid_8273277.gpkg \
    --output-dir outputs/gv_24_year \
    --date 2024-09-30

echo "Done!"
