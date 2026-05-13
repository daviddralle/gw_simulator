# gw_simulator

`gw_simulator` is a Python package for simulating catchment-scale groundwater flow and streamflow depletion using the Landlab Dupuit-Boussinesq component. The package is designed to be highly generalizable: it can extract topography from Earth Engine, calculate recharge, and run both unimpaired (natural) and pumped groundwater scenarios for any defined watershed boundary.

## Overview of the Workflow

To run a simulation for a new site, you must complete four main steps using the scripts provided in the `scripts/` directory:

1. **Extract DEM**: Download and clip a 3DEP DEM for your catchment.
2. **Compute Recharge**: Generate a daily time series of recharge forcing.
3. **Run Groundwater**: Simulate the physical groundwater system.
4. **Plot Cross Sections**: (Optional) Visualize the spatial distribution of the water table and depletion cone.

---

## Input Data Formats

To apply this model to your own watershed, you will need to prepare specific inputs.

### 1. Catchment Boundary
- **Format**: Any vector format supported by GeoPandas (e.g., `.gpkg`, `.shp`, `.geojson`, `.kml`).
- **Usage**: Used to clip the DEM, define the active modeling domain, and mask well locations.

### 2. GLYMPHS Hydrogeology Data
- **Format**: `.tif` rasters.
- **Usage**: You must provide three rasters representing the subsurface properties:
  - Transmissivity (`transmissivity_m2d.tif`)
  - Depth to Bedrock (`depthToBedrock_m.tif`)
  - Porosity / Storativity (`storativity.tif`)
- **Note**: By default, the scripts point to local files in `data/GLYMPHS/`. These rasters should ideally cover your entire study extent.

### 3. Recharge Time Series
- **Format**: `.csv` file.
- **Required Columns**:
  - `date`: In a standard datetime format (e.g., `YYYY-MM-DD`).
  - `Recharge`: Daily recharge rate in **mm/day**.

### 4. Pumping Data (Optional)
Pumping is fully optional. If omitted, the model will run a purely natural (unimpaired) scenario. If included, you must provide two files:

**Well Locations:**
- **Format**: `.gpkg` or `.shp` point vector file.
- **Required Columns**: Must contain a unique identifier column, typically `APN` (Assessor's Parcel Number).

**Pumping Schedule:**
- **Format**: `.csv` file.
- **Required Columns**:
  - `Month`: Integer from `1` to `12` representing the calendar month.
  - `APN`: The unique identifier matching the Well Locations file.
  - *Either* `waterUse_m3Month` (total cubic meters per month) *or* `waterUse_m3Day` (average cubic meters per day for that month).

---

## Step-by-Step Walkthrough

### Step 1: Extract the DEM
Run the DEM extraction script to pull high-resolution topography from Google Earth Engine. Note that you must have your Earth Engine environment initialized (`earthengine authenticate`).

```bash
python scripts/01_extract_dem.py \
    --boundary path/to/your_catchment.gpkg \
    --output-tif data/my_site_dem.tif
```

### Step 2: Compute Recharge
Generate the daily recharge CSV for your simulation period.

```bash
python scripts/02_compute_recharge.py \
    --boundary path/to/your_catchment.gpkg \
    --output-csv data/my_site_recharge.csv \
    --start-year 2000 \
    --end-year 2024
```

### Step 3: Run the Groundwater Simulation
Run the Dupuit-Boussinesq model. This script automatically interpolates the GLYMPHS data, spins up the steady-state aquifer, and routes the daily recharge. 

**For an Unimpaired (Natural) Run:**
```bash
python scripts/03_run_groundwater.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --recharge-csv data/my_site_recharge.csv \
    --start-date 2022-10-01 \
    --end-date 2023-09-30
```
*(By default, this will look for GLYMPHS data in `data/GLYMPHS/`. Use `--transmissivity`, `--depth-to-bedrock`, and `--porosity` to override these paths if necessary).*

**For a Run with Pumping:**
To evaluate streamflow depletion, pass the optional `--wells` and `--pumping-schedule` arguments. The script will automatically run *both* scenarios and generate comparative hydrographs.

```bash
python scripts/03_run_groundwater.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --recharge-csv data/my_site_recharge.csv \
    --wells path/to/my_wells.gpkg \
    --pumping-schedule path/to/my_pumping.csv \
    --start-date 2022-10-01 \
    --end-date 2023-09-30
```

### Step 4: Visualize Depletion Cross Sections
If you want to view a 2D cross section of the aquifer showing the physical depletion cone at the end of the dry season, run the plotting script:

```bash
python scripts/04_plot_cross_sections.py \
    --boundary path/to/your_catchment.gpkg \
    --dem data/my_site_dem.tif \
    --date 2023-09-30
```
*(This script will automatically detect if a pumped scenario was run. If no pumping was provided, it will simply plot the natural groundwater table).*
