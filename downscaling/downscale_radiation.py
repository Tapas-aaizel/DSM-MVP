import xarray as xr
import numpy as np
import os
import glob
import argparse

def downscale_file(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        # ----------------------------
        # Load
        # ----------------------------
        ds = xr.open_dataset(input_path)

        # ----------------------------
        # Select Variable (SWGDN)
        # ----------------------------
        var_name = "SWGDN"
        if var_name not in ds:
            print(f"Warning: {var_name} not found in {input_path}. Skipping file.")
            return

        # Sort by lat/lon to ensure monotonic increasing for interpolation
        ds = ds.sortby(["lat", "lon"])

        # ----------------------------
        # Crop to ROI (Rajasthan + Gujarat)
        # Lat: 20.0 to 30.5
        # Lon: 68.0 to 78.5
        # ----------------------------
        da = ds[var_name].sel(
            lat=slice(26.5, 28.5), # Expanded buffer for interpolation
            lon=slice(70.5, 73.0)
        )

        
        # Note: We do NOT drop the time dimension here.

        # ----------------------------
        # Target ~1 km grid
        # ----------------------------
        # 1 km is approx 0.009 degrees
        target_res = 0.009
        
        new_lats = np.arange(27.0, 28.0 + target_res, target_res)
        new_lons = np.arange(71.4, 72.4 + target_res, target_res)

        # ----------------------------
        # Downscale (Interpolation)
        # ----------------------------
        # method="linear" (bilinear) is standard for spatial downscaling
        da_out = da.interp(
            lat=new_lats,
            lon=new_lons,
            method="linear"
        )

        # Restore/Update Attributes
        da_out.attrs = da.attrs
        da_out.attrs["description"] = f"Downscaled to ~5km ({target_res} deg) using linear interpolation"
        
        # Save as Dataset to preserve coord info properly
        ds_out = da_out.to_dataset()
        ds_out.to_netcdf(output_path)
        print(f"✅ Downscaled {var_name} saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downscale radiation data")
    parser.add_argument("--input_dir", required=True, help="Directory containing input .nc4 files")
    parser.add_argument("--output_dir", required=True, help="Directory to save downscaled files")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Match .nc and .nc4
    files = glob.glob(os.path.join(args.input_dir, "*.nc*"))
    if not files:
        print(f"No .nc/.nc4 files found in {args.input_dir}")
    
    print(f"Found {len(files)} files in {args.input_dir}")

    for f in files:
        if "_downscaled" in f: continue # Avoid re-processing output if in same dir
        
        basename = os.path.basename(f)
        # Ensure we keep the timestamp part intact for regex later
        # Just append _downscaled before extension
        if f.endswith(".nc4"):
            output_name = basename.replace(".nc4", "_downscaled.nc")
        elif f.endswith(".nc"):
             output_name = basename.replace(".nc", "_downscaled.nc")
        else:
             output_name = basename + "_downscaled.nc"
             
        output_path = os.path.join(args.output_dir, output_name)
        downscale_file(f, output_path)
