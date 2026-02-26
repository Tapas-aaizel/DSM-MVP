# AWS Data Extractor & Interpolator

This directory contains scripts responsible for fetching, processing, and interpolating ground-truth weather data. This data is primarily used for **Bias Correction** in the post-processing phase of the Climate Forte pipeline, ensuring that the model's predictions align with real-world observations from automatic weather stations (AWS).

## Core Scripts

### 1. `weather.py`
This is the main driver script used by the Airflow pipeline (`fetch_ground_truth_AWS_Data` task).

**Functionality:**
1.  **Scraping**: Fetches hourly weather data from `www.weather-india.in` for ~85 stations across Haryana, NCR, and Uttar Pradesh.
2.  **Parsing**:Extracts key variables:
    *   Temperature (`T2M`)
    *   Relative Humidity (`RH`)
    *   Pressure (`Pressure`)
    *   Wind Speed & Direction -> converted to `U10`, `V10` components.
3.  **Synchronization**: Filters data to match specific target timestamps (e.g., 00:00, 06:00, 12:00, 18:00 UTC) required for model validation.
4.  **Interpolation**: Uses **Gaussian Process Regression (Kriging)** to interpolate point station data onto the regular **MERRA-2 grid** (0.5° x 0.625°).
5.  **Output**: Saves the result as a NetCDF file (`upscaled_merra2_{YYYY-MM-DD}.nc`) representing the "Ground Truth" state of the atmosphere on that grid.

**Usage:**
```bash
python weather.py --start_date 2026-01-13 --end_date 2026-01-14 --slots 1,2,3,4 --output_dir /path/to/output
```
*   `--slots`: Maps to UTC hours (1=00, 2=03, 3=06, ..., 8=21).

### 2. `legacy_meteostat_app 1.py`
An alternative/backup script (likely legacy) that appears to use the `meteostat` library or a similar approach for historical data retrieval. It is currently not active in the main `v9` pipeline.

### 3. `forcast.py`
Contains logic related to forecast data handling, possibly for validation comparisons or legacy forecasting tests.

## Data Files
*   `upscaled_merra2_*.nc`: Generated NetCDF files containing the interpolated ground truth grids.
*   `upscaled_merra2_*.csv`: Raw station observation data before interpolation.

## Dependencies
*   `requests`, `beautifulsoup4`: For web scraping.
*   `scikit-learn`: For Gaussian Process interpolation (`GaussianProcessRegressor`).
*   `xarray`, `pandas`, `numpy`: For data manipulation and file I/O.

## Workflow Integration
In the `prithvi_pipeline_dag_optimized.py`:
1.  The `fetch_ground_truth_AWS_Data` task calls `weather.py`.
2.  The output `.nc` file allows the pipeline to calculate the `Delta` (Error) between the Model's Global Prediction and this Ground Truth.
3.  This `Delta` is then downscaled and subtracted from the prediction to correct systematic biases.
