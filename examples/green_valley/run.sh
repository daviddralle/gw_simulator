#!/bin/bash
# run.sh
# This script runs the full 24-year groundwater simulation for the Green Valley watershed.

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the simulation from the project root directory
cd ../..

mkdir -p examples/green_valley/outputs

echo "Extracting DEM from Earth Engine..."
python scripts/01_extract_dem.py \
    --boundary data/comid_8273277.gpkg \
    --output-tif examples/green_valley/outputs/dem.tif

echo "Computing Recharge..."
python scripts/02_compute_recharge.py \
    --boundary data/comid_8273277.gpkg \
    --output-csv examples/green_valley/outputs/recharge.csv \
    --start-year 2000 \
    --end-year 2024

echo "Starting 24-year Green Valley Simulation..."

python scripts/03_run_groundwater.py \
    --dem examples/green_valley/outputs/dem.tif \
    --boundary data/comid_8273277.gpkg \
    --recharge-csv examples/green_valley/outputs/recharge.csv \
    --wells data/well_locations.gpkg \
    --pumping-schedule data/GVDB/pumpingSchedule.csv \
    --start-date 2000-10-01 \
    --end-date 2024-09-30 \
    --output-dir examples/green_valley/outputs

echo "Simulation complete! Outputs are saved in 'examples/green_valley/outputs/'"

echo "Generating depletion cross section..."
python scripts/04_plot_cross_sections.py \
    --dem examples/green_valley/outputs/dem.tif \
    --boundary data/comid_8273277.gpkg \
    --output-dir examples/green_valley/outputs \
    --date 2024-09-30

echo "Done!"
