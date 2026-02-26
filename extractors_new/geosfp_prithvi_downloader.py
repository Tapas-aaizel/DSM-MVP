#!/usr/bin/env python3
"""
GEOS-FP Downloader for Prithvi-WxC-2.3B Rollout Model

Downloads GEOS-FP files for near real-time weather prediction pipeline.
Data is 3-hourly to match model requirements.

Pipeline Context (from NEW_IMPLE.txt):
======================================
Phase 1: Ingest - Download GEOS-FP (Global) → MinIO scratch-space/raw
Phase 2: Pre-process - Regrid to MERRA-2 (0.5°) → Assemble → scratch-space/ready
Phase 3: Inference - Run Prithvi → Global NetCDF → scratch-space/global-pred
Phase 4: Downscale & Crop - Interpolate to 5km, crop to India
Phase 5: Bias Correction - Apply ground truth correction
Phase 6: Formatter - Split Surface/Pressure, convert to Zarr/COG

Required Variables for Prithvi-WxC-2.3B:
========================================
Surface Variables (20):
  EFLUX, GWETROOT, HFLUX, LAI, LWGAB, LWGEM, LWTUP, PS, QV2M, SLP,
  SWGNT, SWTNT, T2M, TQI, TQL, TQV, TS, U10M, V10M, Z0M

Static Surface Variables (4):
  FRACI, FRLAND, FROCEAN, PHIS

Vertical Variables (10 vars × 13 model levels = 130 channels):
  CLOUD, H, OMEGA, PL, QI, QL, QV, T, U, V

Total: 154 channels (20 surface + 4 static + 130 vertical)

GEOS-FP Products:
=================
| Collection        | Variables                                    | Notes                |
|-------------------|----------------------------------------------|----------------------|
| inst3_2d_asm_Nx   | PS, QV2M, SLP, T2M, TQI, TQL, TQV, TS,       | 3-hourly surface     |
|                   | U10M, V10M                                   |                      |
| tavg1_2d_flx_Nx   | EFLUX, HFLUX, Z0M                            | 1-hr→3-hr flux       |
| tavg1_2d_lnd_Nx   | GWETROOT, LAI                                | 1-hr→3-hr land       |
| tavg1_2d_rad_Nx   | LWGAB, LWGEM, LWTUP, SWGNT, SWTNT            | 1-hr→3-hr radiation  |
| tavg1_2d_slv_Nx   | FRSEAICE→FRACI                               | 1-hr→3-hr sea ice    |
| const_2d_asm_Nx   | FRLAND, FROCEAN, PHIS                        | Static constants     |
| inst3_3d_asm_Nv   | CLOUD, H, OMEGA, PL, QI, QL, QV, T, U, V     | 3-hourly model levels|

Usage:
    python geosfp_prithvi_downloader.py --date 2025-12-03
    python geosfp_prithvi_downloader.py --date 2025-12-03 --dry-run
    python geosfp_prithvi_downloader.py --date 2025-12-03 --out ./my_data
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import List, Dict

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================================================================
# GEOS-FP Products required for Prithvi-WxC-1.0-2300M Rollout (3-hourly)
# =============================================================================

PRITHVI_GEOSFP_PRODUCTS = {
    # === 2D Surface Products (20 variables) ===
    'inst3_2d_asm_Nx': {
        'description': '2D Instantaneous 3-hr surface atmospheric variables',
        'variables': ['PS', 'QV2M', 'SLP', 'T2M', 'TQI', 'TQL', 'TQV', 'TS', 'U10M', 'V10M'],
        'prithvi_vars': ['PS', 'QV2M', 'SLP', 'T2M', 'TQI', 'TQL', 'TQV', 'TS', 'U10M', 'V10M'],
    },
    'tavg1_2d_flx_Nx': {
        'description': '2D Flux 1-hr -> select 3-hr (surface flux)',
        'variables': ['EFLUX', 'HFLUX', 'Z0M'],
        'prithvi_vars': ['EFLUX', 'HFLUX', 'Z0M'],
    },
    'tavg1_2d_lnd_Nx': {
        'description': '2D Land 1-hr -> select 3-hr (land surface)',
        'variables': ['GWETROOT', 'LAI'],
        'prithvi_vars': ['GWETROOT', 'LAI'],
    },
    'tavg1_2d_rad_Nx': {
        'description': '2D Radiation 1-hr -> select 3-hr (radiation)',
        'variables': ['LWGAB', 'LWGEM', 'LWTUP', 'SWGNT', 'SWTNT'],
        'prithvi_vars': ['LWGAB', 'LWGEM', 'LWTUP', 'SWGNT', 'SWTNT'],
    },
    'tavg1_2d_slv_Nx': {
        'description': '2D Single-level 1-hr -> select 3-hr (sea ice fraction)',
        'variables': ['FRSEAICE'],
        'prithvi_vars': ['FRACI'],  # FRSEAICE maps to FRACI for Prithvi
    },
    # === Static/Constant Products (3 variables) ===
    'const_2d_asm_Nx': {
        'description': '2D Constants - static surface properties',
        'variables': ['FRLAND', 'FROCEAN', 'PHIS'],
        'prithvi_vars': ['FRLAND', 'FROCEAN', 'PHIS'],
        'is_static': True,
    },
    # === 3D Model Level Products (10 variables × 13 levels) ===
    # NOTE: Prithvi-WxC uses MODEL LEVELS (Nv), not pressure levels (Np)
    # All 10 vertical variables come from inst3_3d_asm_Nv
    'inst3_3d_asm_Nv': {
        'description': '3D Model levels 3-hr (all vertical variables for Prithvi)',
        'variables': ['CLOUD', 'H', 'OMEGA', 'PL', 'QI', 'QL', 'QV', 'T', 'U', 'V'],
        'prithvi_vars': ['CLOUD', 'H', 'OMEGA', 'PL', 'QI', 'QL', 'QV', 'T', 'U', 'V'],
    },
}

# 3-hourly times (00, 03, 06, 09, 12, 15, 18, 21 UTC)
HOURS_3HOURLY = [0, 3, 6, 9, 12, 15, 18, 21]


class LinkParser(HTMLParser):
    """Parse HTML to extract file links."""
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            for attr, val in attrs:
                if attr.lower() == 'href' and val:
                    self.links.append(val)


def build_forecast_url(year: int, month: int, day: int) -> str:
    """
    Build NCCS datashare Forecast URL for a specific date.
    Forecast logic: /forecast/ path, Date - 1 Day, H00 subdirectory.
    """
    from datetime import date, timedelta
    target_date = date(year, month, day)
    folder_date = target_date - timedelta(days=1)
    return f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y{folder_date.year:04d}/M{folder_date.month:02d}/D{folder_date.day:02d}/H00/"


def build_directory_url(year: int, month: int, day: int) -> str:
    """Build NCCS datashare URL for a specific date (defaults to Forecast)."""
    return build_forecast_url(year, month, day)


def fetch_directory_listing(url: str, timeout: int = 60) -> List[str]:
    """Fetch and parse directory listing from NCCS portal."""
    print(f"[DIR] Fetching directory: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    parser = LinkParser()
    parser.feed(resp.text)

    # Filter to only .nc4 files
    files = [href for href in parser.links if href.endswith('.nc4')]
    
    # Remove duplicates
    return list(dict.fromkeys(files))


def fetch_directory_listing_with_fallback(year: int, month: int, day: int, timeout: int = 60):
    """
    Fetch directory listing using the stable D-1 Forecast folder.
    """
    from datetime import date, timedelta, timezone as tz

    target_date = date(year, month, day)
    today_utc = datetime.now(tz.utc).date()
    
    # Logic: For future or today, use Yesterday (relative to Now) as it contains 10-day predictions.
    if target_date >= today_utc:
        forecast_base_date = today_utc - timedelta(days=1)
    else:
        forecast_base_date = target_date - timedelta(days=1)

    forecast_url = f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y{forecast_base_date.year:04d}/M{forecast_base_date.month:02d}/D{forecast_base_date.day:02d}/H00/"
    
    print(f"[FORECAST-STABLE] Using Base Folder (D-1): {forecast_url}")
    try:
        files = fetch_directory_listing(forecast_url, timeout)
        if files:
            # Verify target date exists in this run
            target_str = target_date.strftime('%Y%m%d')
            matching = [f for f in files if target_str in f]
            if matching:
                print(f"[SUCCESS] Found {len(matching)} files in Forecast Run {forecast_base_date}")
                return matching, forecast_url
            else:
                raise RuntimeError(f"Forecast run {forecast_base_date} does not cover {target_date}")
        else:
            raise RuntimeError("Forecast URL returned empty directory")
    except Exception as e:
        error_msg = f"Forecast fetch failed: {e}"
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)



def filter_prithvi_files(files: List[str]) -> Dict[str, List[str]]:
    """
    Filter files to only include Prithvi-WxC required products at 3-hourly intervals.
    
    File naming conventions:
    - inst3_*: instantaneous 3-hourly, timestamps 0000, 0300, 0600, ...
    - tavg1_*: time-averaged 1-hourly, timestamps 0030, 0130, 0230, ... -> select 0030, 0330, 0630, ...
    """
    result = {}
    
    # tavg1 files use centered timestamps: 0030, 0130, 0230, ...
    # We select 3-hourly: 0030, 0330, 0630, 0930, 1230, 1530, 1830, 2130
    TAVG1_3HR_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]  # hour part of 0030, 0330, etc.
    
    for product in PRITHVI_GEOSFP_PRODUCTS.keys():
        matching = []
        
        for f in files:
            if product not in f:
                continue
            
            # Extract hour and minute from filename (e.g., 20251203_0030 -> 00, 30)
            match = re.search(r'\.(\d{8})_(\d{2})(\d{2})\.', f)
            if match:
                file_hour = int(match.group(2))
                file_min = int(match.group(3))
                
                if 'inst3' in product:
                    # inst3 files: exact 3-hourly (0000, 0300, 0600, ...)
                    if file_hour in HOURS_3HOURLY and file_min == 0:
                        matching.append(f)
                elif 'tavg1' in product:
                    # tavg1 files: select 3-hourly from 1-hourly (0030, 0330, 0630, ...)
                    if file_hour in TAVG1_3HR_HOURS and file_min == 30:
                        matching.append(f)
        
        if matching:
            result[product] = sorted(matching)
    
    return result


def download_file(session: requests.Session, base_url: str, filename: str, out_dir: str,
                  retries: int = 5, chunk_size: int = 1024 * 64, skip_existing: bool = True) -> str:
    """Download a single file with retries (no resume)."""
    url = base_url + filename
    out_path = os.path.join(out_dir, filename)
    
    # Check if file exists and get expected size from server
    if skip_existing and os.path.exists(out_path):
        # Get expected file size from server (HEAD request)
        try:
            head_resp = session.head(url, timeout=30)
            expected_size = int(head_resp.headers.get('Content-Length') or 0)
            actual_size = os.path.getsize(out_path)
            
            if expected_size > 0 and actual_size >= expected_size:
                print(f"[SKIP] Already complete: {filename} ({actual_size / 1e6:.1f} MB)")
                return out_path
            elif actual_size > 0:
                print(f"[INCOMPLETE] {filename}: {actual_size / 1e6:.1f} MB / {expected_size / 1e6:.1f} MB - re-downloading")
        except Exception as e:
            # If we can't check size, just skip if file exists
            print(f"[SKIP] Already exists (size check failed): {filename}")
            return out_path

    attempt = 0
    while attempt < retries:
        attempt += 1
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                
                # Get total size
                total = int(r.headers.get('Content-Length') or 0)
                
                # Progress bar
                if tqdm:
                    pbar = tqdm(total=total, unit='B', unit_scale=True,
                               desc=filename[:45], leave=True)
                else:
                    print(f"[DOWNLOAD] {filename} ({total / 1e6:.1f} MB)")
                    pbar = None

                # Write file
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))

                if pbar:
                    pbar.close()
                
                print(f"[DONE] Complete: {filename}")
                return out_path

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Cancelled: {filename}")
            # Keep partial file - can resume later or will be overwritten
            print(f"[INFO] Partial file kept: {out_path}")
            raise
        except Exception as e:
            print(f"[ERROR] Attempt {attempt}/{retries} failed: {e}")
            # DON'T delete partial files - keep them for potential resume or debugging
            # The file will be overwritten on successful retry anyway
            if attempt < retries:
                wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)

    raise RuntimeError(f"Failed after {retries} attempts: {filename}")


def check_existing_files(out_dir: str, session: requests.Session, base_url: str, 
                         required_files: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    Check which files already exist and are complete.
    
    Returns dict: {product: {filename: status}} where status is 'complete', 'incomplete', or 'missing'
    """
    result = {}
    
    if not os.path.exists(out_dir):
        # No directory = all files missing
        for product, files in required_files.items():
            result[product] = {f: 'missing' for f in files}
        return result
    
    existing_files = set(os.listdir(out_dir))
    
    for product, files in required_files.items():
        result[product] = {}
        
        for filename in files:
            if filename not in existing_files:
                result[product][filename] = 'missing'
                continue
            
            # File exists - check if complete by comparing size with server
            local_path = os.path.join(out_dir, filename)
            local_size = os.path.getsize(local_path)
            
            try:
                url = base_url + filename
                head_resp = session.head(url, timeout=30)
                server_size = int(head_resp.headers.get('Content-Length') or 0)
                
                if server_size > 0 and local_size >= server_size:
                    result[product][filename] = 'complete'
                elif local_size > 0:
                    result[product][filename] = 'incomplete'
                else:
                    result[product][filename] = 'missing'
            except Exception:
                # Can't verify size - assume complete if file exists and has content
                if local_size > 0:
                    result[product][filename] = 'complete'
                else:
                    result[product][filename] = 'missing'
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Download GEOS-FP data for Prithvi-WxC (3-hourly)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python geosfp_prithvi_downloader.py --date 2025-12-03 --dry-run
  python geosfp_prithvi_downloader.py --date 2025-12-03 --out ./data
        """
    )
    
    parser.add_argument('--date', '-d', type=str, required=True,
                       help='Date in YYYY-MM-DD format')
    parser.add_argument('--out', '-o', default='geosfp_data', 
                       help='Output directory (default: geosfp_data)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='List files only, do not download')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    # Parse date
    try:
        dt = datetime.strptime(args.date, '%Y-%m-%d')
        year, month, day = dt.year, dt.month, dt.day
    except ValueError:
        print("[ERROR] Invalid date. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           GEOS-FP DOWNLOADER FOR PRITHVI-WxC                         ║
║                                                                      ║
║  Date: {year}-{month:02d}-{day:02d}                                                        ║
║  Resolution: 3-hourly (00, 03, 06, 09, 12, 15, 18, 21 UTC)           ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Fetch directory listing from NASA with automatic DAS → Forecast fallback
    try:
        files, base_url = fetch_directory_listing_with_fallback(year, month, day)
        print(f"[INFO] Using URL: {base_url}")
    except requests.HTTPError as e:
        print(f"[ERROR] Failed to fetch: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    if not files:
        print("[ERROR] No files found.")
        sys.exit(1)
    
    # Filter for Prithvi products
    filtered = filter_prithvi_files(files)
    
    if not filtered:
        print("[ERROR] No matching Prithvi files found.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Check existing files BEFORE downloading
    print("\n[CHECK] Scanning existing files...")
    session = requests.Session()
    file_status = check_existing_files(args.out, session, base_url, filtered)
    
    # Print detailed summary
    print("\n[INFO] FILE STATUS BY COLLECTION:")
    print("=" * 70)
    
    total_files = 0
    files_complete = 0
    files_incomplete = 0
    files_missing = 0
    
    for product, product_files in filtered.items():
        info = PRITHVI_GEOSFP_PRODUCTS[product]
        statuses = file_status.get(product, {})
        
        complete = sum(1 for s in statuses.values() if s == 'complete')
        incomplete = sum(1 for s in statuses.values() if s == 'incomplete')
        missing = sum(1 for s in statuses.values() if s == 'missing')
        
        print(f"\n[PRODUCT] {product}")
        print(f"   {info['description']}")
        print(f"   Variables: {', '.join(info['variables'])}")
        print(f"   Status: {complete} complete, {incomplete} incomplete, {missing} missing")
        
        for f in product_files:
            status = statuses.get(f, 'missing')
            if status == 'complete':
                print(f"      [COMPLETE] {f}")
            elif status == 'incomplete':
                print(f"      [INCOMPLETE] {f} - will re-download")
            else:
                print(f"      [MISSING] {f}")
        
        total_files += len(product_files)
        files_complete += complete
        files_incomplete += incomplete
        files_missing += missing
    
    files_to_download = files_incomplete + files_missing
    
    print(f"\n{'=' * 70}")
    print(f"[SUMMARY] Total: {total_files} files")
    print(f"   Complete (skip): {files_complete}")
    print(f"   Incomplete (re-download): {files_incomplete}")
    print(f"   Missing (download): {files_missing}")
    print(f"   To download: {files_to_download}")
    
    if args.dry_run:
        print("\n[INFO] Dry run complete.")
        return
    
    if files_to_download == 0 and not args.force:
        print("\n[INFO] All files already complete. Nothing to download.")
        print("[INFO] Use --force to re-download anyway.")
        return
    
    # Download only missing/incomplete files
    print(f"\n[INFO] Downloading to: {args.out}")
    
    downloaded, skipped, failed = 0, 0, 0
    
    for product, product_files in filtered.items():
        statuses = file_status.get(product, {})
        
        # Check if any files need downloading for this product
        needs_download = [f for f in product_files 
                        if statuses.get(f) != 'complete' or args.force]
        
        if not needs_download:
            print(f"\n[PRODUCT] {product}: All {len(product_files)} files complete - SKIPPING")
            skipped += len(product_files)
            continue
        
        print(f"\n[PRODUCT] {product}: {len(needs_download)}/{len(product_files)} files to download")
        
        for f in product_files:
            status = statuses.get(f, 'missing')
            
            if status == 'complete' and not args.force:
                print(f"[SKIP] Complete: {f}")
                skipped += 1
                continue
            
            try:
                download_file(session, base_url, f, args.out, skip_existing=False)
                downloaded += 1
            except Exception as e:
                print(f"[ERROR] {f}: {e}")
                failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"[DONE] Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"[INFO] Output: {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
