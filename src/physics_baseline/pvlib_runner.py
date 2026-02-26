"""
Stage S2: PVLib Physics Baseline
Converts NWP weather variables to expected DC/AC power using physical PV model.
Following ClimateForte Solar DSM MVP Execution Directive v1.0
"""
import pandas as pd
import numpy as np
import pvlib
import os
import logging
from pvlib.irradiance import erbs, get_total_irradiance, get_extra_radiation
from pvlib.solarposition import get_solarposition
from pvlib.temperature import faiman
from src import config

# ============================================================================
# 1. MANDATORY CONFIGURATION (Directive Page 4)
# ============================================================================
LAT, LON, ALT = config.LATITUDE, config.LONGITUDE, config.ALTITUDE
TILT, AZIMUTH = config.FIXED_TILT, config.AZIMUTH
CAPACITY_MW = config.CAPACITY_MW
TEMP_COEFF = -0.0035 # User-provided override

# Bounding Box for Bhadla Region (Ensures regional stability)
LAT_MIN, LAT_MAX = 27.0, 28.0
LON_MIN, LON_MAX = 71.4, 72.4

def run_pvlib_baseline(input_path, output_dir=None):
    """
    S2: Run PVLib Physics Baseline using the mandated regional-mean approach.
    """
    print(f"🚀 Module S2: Processing Bhadla Baseline from {input_path}...")
    
    # --- STEP 1: LOAD & CLEAN DATA ---
    if not os.path.exists(input_path):
        print(f"❌ ERROR: File not found at {input_path}")
        return

    if input_path.endswith(('.nc', '.nc4')):
        import xarray as xr
        print(f"Opening NetCDF: {input_path}")
        ds = xr.open_dataset(input_path)
        # Convert to DF and reset index to match CSV structure for the BBox mask
        df_raw = ds.to_dataframe().reset_index()
    else:
        df_raw = pd.read_csv(input_path)
    
    # Handle mixed date formats (Optimized)
    time_col = 'time' if 'time' in df_raw.columns else 'timestamp_ist'
    if time_col not in df_raw.columns and 'timestamp_utc' in df_raw.columns:
        time_col = 'timestamp_utc'
    
    # Specify format for 100x speedup on large datasets
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # Fallback for any missed rows
    if df_raw[time_col].isna().any():
        df_raw[time_col] = df_raw[time_col].fillna(pd.to_datetime(df_raw[time_col], format='mixed', dayfirst=True))

    # --- STEP 2: GEOGRAPHY LOCK (BBox Selection) ---
    mask = (df_raw['lat'] >= LAT_MIN) & (df_raw['lat'] <= LAT_MAX) & \
           (df_raw['lon'] >= LON_MIN) & (df_raw['lon'] <= LON_MAX)
    df_site = df_raw[mask].copy()

    if df_site.empty:
        print("❌ ERROR: No data found in Bhadla BBox coordinates.")
        return

    # --- STEP 3: AGGREGATE & IST ALIGNMENT ---
    # Take the mean of the region to fill any spatial NaNs
    df_hourly = df_site.groupby(time_col).mean(numeric_only=True)
    
    # Ensure timezone is IST
    if df_hourly.index.tz is None:
        # Assume input is UTC if no TZ (standard for NWP)
        df_hourly.index = df_hourly.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df_hourly.index = df_hourly.index.tz_convert('Asia/Kolkata')

    # --- STEP 4: SMOOTH INTERPOLATION (96 BLOCKS) ---
    print("🪄 Applying 15-minute linear ramping to eliminate staircase...")
    start_time = df_hourly.index.min().floor('D')
    end_time = df_hourly.index.max().ceil('D')
    clean_index = pd.date_range(start=start_time, end=end_time, freq='15min', tz='Asia/Kolkata')
    
    # Reindex and interpolate hourly NWP to 15-min intervals
    df_96 = df_hourly.reindex(clean_index).interpolate(method='linear')
    df_96 = df_96.fillna(0)
    times = df_96.index

    # --- STEP 5: PHYSICS ENGINE (Perez + Faiman) ---
    print("🧠 Executing Physics Engine...")
    
    # A. Solar Position
    solpos = get_solarposition(times, LAT, LON, altitude=ALT)
    zenith = solpos['apparent_zenith']
    azimuth = solpos['azimuth']
    dni_extra = get_extra_radiation(times)
    
    # B. Wind & Temp Setup
    if 'wspd' not in df_96.columns:
        if 'WS10' in df_96.columns:
            df_96['wspd'] = df_96['WS10']
        elif 'U10' in df_96.columns and 'V10' in df_96.columns:
            df_96['wspd'] = np.sqrt(df_96['U10']**2 + df_96['V10']**2)
        else:
            df_96['wspd'] = 2.0
    
    # T2M handling (Kelvin to Celsius)
    temp_c = df_96['T2M'] - 273.15 if df_96['T2M'].mean() > 200 else df_96['T2M']

    # C. GHI Cleaning (Crucial Fix for Horizon Stability)
    ghi_cleaned = df_96['SWGDN'].copy()
    ghi_cleaned[zenith > 88] = 0

    # D. Irradiance Decomposition (Erbs)
    dni_dhi = erbs(ghi_cleaned, zenith, times)
    
    # E. Transposition (Perez Model)
    total_irrad = get_total_irradiance(
        surface_tilt=TILT, surface_azimuth=AZIMUTH,
        solar_zenith=zenith, solar_azimuth=azimuth,
        dni=dni_dhi['dni'], ghi=ghi_cleaned, dhi=dni_dhi['dhi'],
        dni_extra=dni_extra, model='perez'
    )
    poa = total_irrad['poa_global']
    
    # F. Cell Temperature (Faiman Model)
    cell_temp = faiman(poa, temp_c, df_96['wspd'])
    
    # G. Temperature Derating
    temp_eff = 1 + (TEMP_COEFF * (cell_temp - 25))
    
    # H. Final MW Calculation
    phys_mw = (poa / 1000) * CAPACITY_MW * temp_eff
    
    # --- STEP 6: MANDATORY SAFETY GATES & CLEANUP ---
    # Apply Night Gate
    phys_mw[zenith > 90] = 0  
    
    # Limit to reasonable capacity (110% max)
    phys_mw = phys_mw.clip(0, CAPACITY_MW * 1.1) 

    # --- STEP 7: OUTPUT GENERATION ---
    physics_baseline = pd.DataFrame({
        'timestamp_ist': times.strftime('%Y-%m-%d %H:%M:%S'),
        'block_number': [(i % 96) + 1 for i in range(len(times))],
        'ghi_forecast': ghi_cleaned.values.round(2),
        't2m_forecast': temp_c.values.round(2),
        'wind_speed_forecast': df_96['wspd'].values.round(2),
        'tcc_forecast': (df_96['TCC'].values / 8.0).round(3) if 'TCC' in df_96.columns else 0.0,
        'solar_zenith': zenith.values.round(2),
        'solar_azimuth': azimuth.values.round(2),
        'hour_of_day': times.hour,
        'day_of_year': times.dayofyear,
        'pvlib_predicted_mw': phys_mw.values.round(4)
    })
    
    # THE FINAL FIX: Fill horizon NaNs with 0 to ensure clean ML training
    physics_baseline['pvlib_predicted_mw'] = physics_baseline['pvlib_predicted_mw'].fillna(0)

    # Save output
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "physics_baseline.csv")
    physics_baseline.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: 'physics_baseline.csv' generated ({len(physics_baseline)} blocks).")
    print(f"Preview of Daytime Blocks:\n{physics_baseline.iloc[35:40]}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        run_pvlib_baseline(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        run_pvlib_baseline(sys.argv[1])
    else:
        print("Usage: python -m src.physics_baseline.pvlib_runner <input_path> [<output_dir>]")
