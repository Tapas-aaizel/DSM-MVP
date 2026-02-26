#!/usr/bin/env python3
"""
Multi-day downloader for GEOS-FP tavg3_2d_aer_Nx files (Aerosols).
Downloads files for a specified date range.

Usage:
    python download_aer_range.py --start-date 2026-02-01 --days 5 --out ./aer_data
"""
import argparse
import os
import sys
import time
import requests
from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import List, Tuple

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

COLLECTION_NAME = "tavg3_2d_aer_Nx"

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
    """Build NCCS datashare Forecast URL (D-1 H00)."""
    target_date = datetime(year, month, day)
    folder_date = target_date - timedelta(days=1)
    return f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y{folder_date.year:04d}/M{folder_date.month:02d}/D{folder_date.day:02d}/H00/"

def fetch_directory_listing(url: str, timeout: int = 60) -> List[str]:
    """Fetch and parse directory listing."""
    print(f"[DIR] Fetching: {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Failed to fetch directory: {e}")
        return []

    parser = LinkParser()
    parser.feed(resp.text)
    # Filter for .nc4 files
    files = [href for href in parser.links if href.endswith('.nc4')]
    return list(dict.fromkeys(files))

def get_files_with_fallback(dt: datetime) -> Tuple[List[str], str]:
    """Get list of files using Forecast-only strategy for a specific date."""
    year, month, day = dt.year, dt.month, dt.day

    fcst_url = build_forecast_url(year, month, day)
    print(f"[FORECAST-ONLY] Checking Forecast for {dt.date()}: {fcst_url}")
    files = fetch_directory_listing(fcst_url)
    if files:
        print(f"[INFO] Found {len(files)} files in Forecast for {dt.date()}")
        return files, fcst_url

    print(f"[WARN] No files found for {dt.date()} in Forecast location")
    return [], ""

def download_file(url: str, out_path: str):
    """Download a single file."""
    if os.path.exists(out_path):
        # We could add size verification here, but basic existence check is usually enough for bulk getters
        print(f"[SKIP] Exists: {os.path.basename(out_path)}")
        return

    print(f"[DOWNLOAD] {os.path.basename(out_path)}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(out_path, 'wb') as f:
                if tqdm:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(out_path)[:20]) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        if os.path.exists(out_path):
            os.remove(out_path) # Clean up partial

def main():
    parser = argparse.ArgumentParser(description=f"Download {COLLECTION_NAME} files for a date range")
    parser.add_argument("--start-date", required=True, help="Start Date in YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=1, help="Number of days to fetch")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    try:
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError:
        print("[FATAL] Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    
    total_downloaded = 0
    
    for i in range(args.days):
        current_dt = start_dt + timedelta(days=i)
        print(f"\n[{i+1}/{args.days}] Processing {current_dt.date()}...")
        
        try:
            all_files, base_url = get_files_with_fallback(current_dt)
        except Exception as e:
            print(f"[ERROR] Error checking {current_dt.date()}: {e}")
            continue

        if not all_files:
            continue

        # Filter for target collection
        target_files = [f for f in all_files if COLLECTION_NAME in f]
        
        if not target_files:
            print(f"[WARN] No {COLLECTION_NAME} files found for {current_dt.date()}")
            continue

        print(f"[INFO] Found {len(target_files)} relevant files to download")

        for filename in sorted(target_files):
            url = base_url + filename
            
            # Organize by date subfolders if downloading many days?
            # User didn't strictly request it, but it's often cleaner. 
            # Sticking to flat output dir as per request unless user asks otherwise.
            out_path = os.path.join(args.out, filename)
            
            download_file(url, out_path)
            total_downloaded += 1

    print(f"\n[DONE] Process finished. Downloaded/Checked files for {args.days} days.")

if __name__ == "__main__":
    main()
