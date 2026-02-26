#!/usr/bin/env python3
"""
Final Merge Script
------------------
Merges interpolated Terrain variables, Radiation variables, and AWS Cloud Cover
into a single 5km 15-minute resolution NetCDF file.

Inputs:
- Terrain NC: T2M, U10M, V10M
- Radiation NC: SWGDN
- AWS NC: TCC
"""

import os
import argparse
import xarray as xr
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Merge Final Outputs")
    parser.add_argument("--terrain_dir", type=str, required=False, help="Directory containing interpolated terrain .nc file (Optional)")
    parser.add_argument("--rad_dir", type=str, required=True, help="Directory containing interpolated radiation .nc file")
    parser.add_argument("--aws_dir", type=str, required=True, help="Directory containing AWS Ground Truth .nc file")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the final merged output file")
    return parser.parse_args()

def find_nc_file(directory):
    """Find the first .nc file in a directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = [f for f in os.listdir(directory) if f.endswith(".nc")]
    if not files:
        raise FileNotFoundError(f"No .nc files found in {directory}")
    
    return os.path.join(directory, sorted(files)[-1]) # Use the latest/sorted one

def main():
    args = parse_args()
    
    print("="*60)
    print("FINAL MERGE: STARTING")
    print(f"Terrain Dir: {args.terrain_dir}")
    print(f"Rad Dir:     {args.rad_dir}")
    print(f"AWS Dir:     {args.aws_dir}")
    print("="*60)
    
    # 1. Locate Files
    terrain_path = None
    if args.terrain_dir:
        try:
            terrain_path = find_nc_file(args.terrain_dir)
            print(f"[MERGE] Terrain File: {terrain_path}")
        except FileNotFoundError:
            print("[MERGE] Terrain directory provided but empty/missing. Proceeding without Terrain.")

    try:
        rad_path = find_nc_file(args.rad_dir)
    except FileNotFoundError as e:
        print(f"[MERGE] Error locating Rad input files: {e}")
        raise e

    print(f"[MERGE] Rad File:     {rad_path}")

    try:
        aws_path = find_nc_file(args.aws_dir)
    except FileNotFoundError as e:
        print(f"[MERGE] Error locating AWS input files: {e}")
        raise e

    print(f"[MERGE] AWS File:     {aws_path}")
    
    # 2. Load Datasets
    ds_terr = xr.open_dataset(terrain_path) if terrain_path else None
    ds_rad = xr.open_dataset(rad_path)
    ds_aws = xr.open_dataset(aws_path)
    
    # ROI: Bhadla Solar Park Focus (Tight)
    # Mandated MVP Geography: 27.5°N, 71.9°E
    ROI_LAT = slice(27.0, 28.0) 
    ROI_LON = slice(71.4, 72.4)
    
    # 3. Select & Rename Variables
    print("[MERGE] Selecting variables...")
    
    # Terrain: T2M, U10M, V10M
    ds_final_terr = xr.Dataset()
    if ds_terr:
        terr_vars = {}
        if 'T2M' in ds_terr: terr_vars['T2M'] = 'T2M'
        
        if 'U10M' in ds_terr: terr_vars['U10M'] = 'U10'
        elif 'U10' in ds_terr: terr_vars['U10'] = 'U10'
        
        if 'V10M' in ds_terr: terr_vars['V10M'] = 'V10'
        elif 'V10' in ds_terr: terr_vars['V10'] = 'V10'
        
        ds_final_terr = ds_terr[list(terr_vars.keys())].rename(terr_vars)
        
        # Standardize Grid Coords: y/x -> lat/lon
        if 'y' in ds_final_terr.dims and 'x' in ds_final_terr.dims:
            print("[MERGE] Renaming y/x to lat/lon for alignment")
            ds_final_terr = ds_final_terr.rename({'y': 'lat', 'x': 'lon'})
            
        # Select ROI
        # Check if lat/lon are coords
        if 'lat' in ds_final_terr.coords and 'lon' in ds_final_terr.coords:
             print("[MERGE] Slicing Terrain to ROI (Raj+Guj)...")
             ds_final_terr = ds_final_terr.sel(lat=ROI_LAT, lon=ROI_LON)

    # Rad: SWGDN
    if 'SWGDN' not in ds_rad:
        print("[MERGE] CAUTION: SWGDN not found in Radiation file!")
    
    ds_final_rad = ds_rad[['SWGDN']] if 'SWGDN' in ds_rad else xr.Dataset()
    
    # AWS: TCC, T2M, WS10, U10, V10
    # The AWS file contains: T2M, WS10, U10, V10, TCC
    aws_vars_to_keep = []
    for var in ['TCC', 'T2M', 'WS10', 'U10', 'V10']:
        if var in ds_aws:
            aws_vars_to_keep.append(var)
        else:
            print(f"[MERGE] CAUTION: {var} not found in AWS file!")

    ds_final_aws = ds_aws[aws_vars_to_keep] if aws_vars_to_keep else xr.Dataset()
    
    # Clip TCC if likely Cloud Cover
    if 'TCC' in ds_final_aws:
         ds_final_aws['TCC'] = ds_final_aws['TCC'].clip(0, 8) 
         
    # Slice Rad to ROI if not already aligned via Terrain
    if not args.terrain_dir:
         print("[MERGE] Slicing Radiation to ROI (Raj+Guj) as Master Grid...")
         # Ensure lat/lon exist
         if 'lat' in ds_final_rad.coords and 'lon' in ds_final_rad.coords:
              # Handle potential descending lat in GEOS
              lats = ds_final_rad.lat
              if lats[0] > lats[-1]:
                   # Descending, swap slice
                   ds_final_rad = ds_final_rad.sel(lat=slice(ROI_LAT.stop, ROI_LAT.start), lon=ROI_LON)
              else:
                   ds_final_rad = ds_final_rad.sel(lat=ROI_LAT, lon=ROI_LON)

    # 4. Alignment
    # Use Terrain Grid as Master if exists, else Rad
    print("[MERGE] Aligning grids and time...")
    
    master_grid = ds_final_terr if args.terrain_dir and 'lat' in ds_final_terr.coords else ds_final_rad
    
    # Align Rad to Master (if different)
    if args.terrain_dir:
        try:
            ds_final_rad = ds_final_rad.interp_like(master_grid, method='nearest')
        except Exception as e:
            print(f"[MERGE] Rad alignment failed: {e}")
    
    # Align AWS to Master
    try:
        # Interpolate AWS vars to the exact grid points
        ds_final_aws = ds_final_aws.interp_like(master_grid, method='nearest')
    except Exception as e:
        print(f"[MERGE] AWS alignment failed: {e}")
        
    # 5. Merge
    print("[MERGE] Merging datasets...")
    
    # Filter out empty datasets
    datasets_to_merge = [d for d in [ds_final_terr, ds_final_rad, ds_final_aws] if len(d.data_vars) > 0]
    
    if not datasets_to_merge:
        raise ValueError("No datasets to merge!")

    merged = xr.merge(datasets_to_merge)
    
    # 5.5 Keep in UTC for now; let downstream tasks handle localization
    # (Removed manual shift to prevent double-shifting if pvlib_runner also localizes)
    # print("[MERGE] Converting Timezone: UTC -> IST (+05:30)")
    # if 'time' in merged.coords:
    #     merged['time'] = merged.indexes['time'] + pd.Timedelta(hours=5, minutes=30)
    #     merged.attrs['timezone'] = 'IST (UTC+05:30)'
    merged.attrs['timezone'] = 'UTC'
    
    # 6. Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # derived speed/dir (if not in AWS or need recalc)
    if 'U10' in merged and 'V10' in merged and 'WD10' not in merged:
        # Calculate Wind Direction
        merged['WD10'] = (270 - np.degrees(np.arctan2(merged['V10'], merged['U10']))) % 360
        
    merged.attrs['description'] = "Final Merged Forecast: Rad(SWGDN) + AWS(TCC, T2M, WS10, U10, V10) [Solar MVP-DSM]"
    merged.attrs['created'] = pd.Timestamp.now().isoformat()
    
    print(f"[MERGE] Saving to {args.output_file} (IST)")
    merged.to_netcdf(args.output_file)
    
    # 7. Save to CSV
    csv_file = args.output_file.replace(".nc", ".csv")
    print(f"[MERGE] Generating CSV: {csv_file}")
    
    try:
        # Convert to DataFrame
        # This can be large, so we drop NaNs immediately (Masked Terrain)
        # We assume T2M defines the valid domain
        print("[MERGE] Converting to DataFrame (dropping NaNs)...")
        
        # Flatten
        df = merged.to_dataframe().reset_index()
        
        # Standardize time format to prevent redundancies (e.g., midnight formatting)
        if 'time' in df.columns:
             df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        if 'T2M' in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=['T2M'])
            print(f"[MERGE] Reduced rows from {initial_len} to {len(df)}")
            
        # Select only requested columns
        wanted_cols = ['time', 'lat', 'lon', 'T2M', 'U10', 'V10', 'SWGDN', 'TCC']
        final_cols = [c for c in wanted_cols if c in df.columns]
        
        df = df[final_cols]
        
        # Round floats to 3 decimal places to reduce file size
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(3)
        
        df.to_csv(csv_file, index=False)
        print("[MERGE] CSV saved successfully.")
        
    except Exception as e:
        print(f"[MERGE] Failed to generate CSV: {e}")

    print("[MERGE] Success.")

if __name__ == "__main__":
    main()
