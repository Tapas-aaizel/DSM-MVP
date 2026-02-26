#!/usr/bin/env python3
"""
GEOS RAD (GHI) Interpolation Script (Pipeline Version)
------------------------------------------------------
• Reads 24-hour downscaled radiation files from Input Directory
• Concatenates and Interpolates to 15-minute resolution (Forward Fill)
• Saves output to Output Directory
"""

import os
import argparse
import glob
import re
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Interpolate Radiation Data to 15-min")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing downscaled .nc files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save interpolated output")
    parser.add_argument("--date", type=str, required=True, help="Target Date (YYYY-MM-DD)")
    return parser.parse_args()

def extract_time_from_filename(filename):
    # Regex for ...YYYYMMDD_HHMM...
    # Matches both +YYYYMMDD_HHMM and .YYYYMMDD_HHMM
    match = re.search(r'[+.](\d{8})_(\d{4})\.', os.path.basename(filename))
    if match:
        dt_str = match.group(1) + match.group(2)
        return datetime.strptime(dt_str, "%Y%m%d%H%M")
    return None

def main():
    args = parse_args() 
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_date_str = args.date
    
    print(f"[RAD-INTERP] Starting for {target_date_str}")
    print(f"[RAD-INTERP] Input: {input_dir}")
    print(f"[RAD-INTERP] Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all downscaled NC files
    # Pattern: *downscaled.nc or *.nc4 or *.nc
    all_files = glob.glob(os.path.join(input_dir, "*.nc*"))
    
    if not all_files:
        print("[RAD-INTERP] No downscaled files found in input directory.")
        raise RuntimeError("No downscaled files found in input directory.")

    # 2. Filter, Sort, and Deduplicate by Time
    # We sort all_files first to ensure we process them in lexicographical order.
    # If multiple files exist for the same validity time (e.g. from different forecast initializations),
    # the one with the "larger" filename (later initialization date) will be processed last.
    all_files.sort()
    
    files_map = {}
    
    # Target date object
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    for f in all_files:
        dt = extract_time_from_filename(f)
        if dt:
            # Overwrite entry for this timestamp -> Last one wins (latest init)
            files_map[dt] = f
            
    # Convert back to list
    valid_files = list(files_map.items())
            
    # Sort by validity time
    valid_files.sort(key=lambda x: x[0])
    
    if not valid_files:
        print("[RAD-INTERP] No valid timestamped files found.")
        raise RuntimeError("No valid timestamped files found.")

    print(f"[RAD-INTERP] Found {len(valid_files)} valid files.")
    
    # 3. Load Datasets
    datasets = []
    for dt, f in valid_files:
        try:
            ds = xr.open_dataset(f)
            # Ensure time dimension exists or add it
            if 'time' not in ds.dims:
                # Add time dim
                # Try to get single time value if exists as scalar coords
                if 'time' in ds.coords and ds.coords['time'].size == 1:
                     # It's a scalar coord, expand dims
                     ds = ds.expand_dims('time')
                else:
                    # Create time dim from filename timestamp
                    ds = ds.expand_dims(time=[pd.to_datetime(dt)])
            
            # ROI Slicing (Rajasthan + Gujarat)
            # Approx Bounds: Lat 20.0 to 31.0, Lon 68.0 to 79.0
            # Handle potential descending lat in GEOS
            if 'lat' in ds.coords and 'lon' in ds.coords:
                lats = ds.lat
                roi_lat = slice(20.0, 31.0)
                roi_lon = slice(68.0, 79.0)
                
                if lats[0] > lats[-1]:
                    # Descending
                    ds = ds.sel(lat=slice(31.0, 20.0), lon=roi_lon)
                else:
                    ds = ds.sel(lat=roi_lat, lon=roi_lon)
            
            datasets.append(ds)
        except Exception as e:
            print(f"[RAD-INTERP] Failed to open {f}: {e}")

    if not datasets:
        print("[RAD-INTERP] No datasets could be loaded.")
        raise RuntimeError("No datasets could be loaded.")

    # 4. Concatenate
    print("[RAD-INTERP] Concatenating datasets...")
    try:
        ds_combined = xr.concat(datasets, dim="time").sortby("time")
    except Exception as e:
        print(f"[RAD-INTERP] Concatenation failed: {e}")
        raise RuntimeError(f"Concatenation failed: {e}")

    # 5. Interpolate (Linear Interpolation for smooth curves)
    print("[RAD-INTERP] Resampling to 15min (Linear Interpolation)...")
    # Create the target 15-min time index
    time_15min = pd.date_range(
        start=ds_combined.time.values[0],
        end=ds_combined.time.values[-1],
        freq="15min"
    )
    ds_15min = ds_combined.interp(time=time_15min, method="linear")
    
    # Clamp: Radiation cannot be negative (interpolation at sunrise/sunset edges)
    if 'SWGDN' in ds_15min:
        ds_15min['SWGDN'] = ds_15min['SWGDN'].clip(min=0)
    
    # Filter to Target Date Only (Optional, but good for cleanliness)
    # ds_15min = ds_15min.sel(time=slice(target_date_str, target_date_str))

    # 6. Save
    out_filename = f"rad_15min_{target_date_str}.nc"
    out_path = os.path.join(output_dir, out_filename)
    
    print(f"[RAD-INTERP] Saving to {out_path}")
    ds_15min.to_netcdf(out_path)
    
    print("[RAD-INTERP] Done.")

if __name__ == "__main__":
    main()
