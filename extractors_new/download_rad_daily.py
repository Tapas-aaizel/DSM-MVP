#!/usr/bin/env python3
"""
Single-file downloader for GEOS-FP tavg1_2d_rad_Nx files.
Downloads all available files for a specific date (typically 24 hourly files).

Usage:
    python download_rad_daily.py --date 2025-02-01 --out ./rad_data
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

COLLECTION_NAME = "tavg1_2d_rad_Nx"

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

def get_files_with_fallback(date_str: str) -> Tuple[List[str], str]:
    """Get list of files using Forecast-only strategy."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year, month, day = dt.year, dt.month, dt.day

    fcst_url = build_forecast_url(year, month, day)
    print(f"[FORECAST-ONLY] Checking Forecast: {fcst_url}")
    files = fetch_directory_listing(fcst_url)
    if files:
        print(f"[INFO] Found {len(files)} files in Forecast")
        return files, fcst_url

    raise RuntimeError(f"No files found in Forecast location for target date {date_str}")

def download_file(url: str, out_path: str):
    """Download a single file."""
    if os.path.exists(out_path):
        # Simple size check could be added here
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
    parser = argparse.ArgumentParser(description=f"Download {COLLECTION_NAME} files")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    # 1. Get file list
    try:
        all_files, base_url = get_files_with_fallback(args.date)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    # 2. Filter for target collection
    target_files = [f for f in all_files if COLLECTION_NAME in f]
    
    if not target_files:
        print(f"[ERROR] No {COLLECTION_NAME} files found for {args.date}")
        sys.exit(1)

    print(f"[INFO] Found {len(target_files)} relevant files to download")

    # 3. Download
    os.makedirs(args.out, exist_ok=True)
    
    for filename in sorted(target_files):
        url = base_url + filename
        out_path = os.path.join(args.out, filename)
        download_file(url, out_path)

    print("[DONE] Download process finished.")

if __name__ == "__main__":
    main()
