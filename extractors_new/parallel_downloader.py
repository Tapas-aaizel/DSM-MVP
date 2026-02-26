#!/usr/bin/env python3
"""
Parallel GEOS-FP Downloader with aria2c support

Features:
- aria2c integration for 4-6x faster downloads (16 connections per file, 8 parallel files)
- Resume capability using HTTP Range headers
- Progress tracking via callback functions
- XCom-compatible progress reporting for Airflow

Usage:
    # Download single collection
    python parallel_downloader.py --date 2025-12-03 --collection inst3_2d_asm_Nx --out ./data
    
    # Download all collections
    python parallel_downloader.py --date 2025-12-03 --out ./data
    
    # Dry run
    python parallel_downloader.py --date 2025-12-03 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Configuration
# =============================================================================

PRITHVI_GEOSFP_PRODUCTS = {
    'inst3_2d_asm_Nx': {
        'description': '2D Instantaneous 3-hr surface atmospheric variables',
        'variables': ['PS', 'QV2M', 'SLP', 'T2M', 'TQI', 'TQL', 'TQV', 'TS', 'U10M', 'V10M'],
        'expected_files': 8,
        'approx_size_mb': 50,
    },
    'tavg1_2d_flx_Nx': {
        'description': '2D Flux 1-hr -> select 3-hr (surface flux)',
        'variables': ['EFLUX', 'HFLUX', 'Z0M', 'FRSEAICE'],
        'expected_files': 8,
        'approx_size_mb': 20,
    },
    'tavg1_2d_lnd_Nx': {
        'description': '2D Land 1-hr -> select 3-hr (land surface)',
        'variables': ['GWETROOT', 'LAI'],
        'expected_files': 8,
        'approx_size_mb': 15,
    },
    'tavg1_2d_rad_Nx': {
        'description': '2D Radiation 1-hr -> select 3-hr (radiation)',
        'variables': ['LWGAB', 'LWGEM', 'LWTUP', 'SWGNT', 'SWTNT'],
        'expected_files': 8,
        'approx_size_mb': 30,
    },
    'tavg1_2d_slv_Nx': {
        'description': '2D Single-level 1-hr -> select 3-hr (sea ice fraction)',
        'variables': ['FRSEAICE'],
        'expected_files': 8,
        'approx_size_mb': 10,
    },
    'const_2d_asm_Nx': {
        'description': '2D Constants - static surface properties',
        'variables': ['FRLAND', 'FROCEAN', 'PHIS'],
        'expected_files': 1,
        'approx_size_mb': 5,
        'is_static': True,
    },
    'inst3_3d_asm_Nv': {
        'description': '3D Model levels 3-hr (all vertical variables for Prithvi)',
        'variables': ['CLOUD', 'H', 'OMEGA', 'PL', 'QI', 'QL', 'QV', 'T', 'U', 'V'],
        'expected_files': 8,
        'approx_size_mb': 1500,  # ~1.5GB each!
    },
}

HOURS_3HOURLY = [0, 3, 6, 9, 12, 15, 18, 21]
TAVG1_3HR_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# aria2c configuration — aggressive settings for speed
ARIA2C_CONNECTIONS = 16   # Increased for maximum speed (stable on NASA portal)
ARIA2C_PARALLEL = 10      # Increased from 4 to download more files concurrently
ARIA2C_TIMEOUT = 600      # 10 minutes per file
ARIA2C_RETRY = 5


def exponential_backoff_retry(
    func: Callable,
    max_retries: int = 5,
    base_delay: float = 30.0,
    max_delay: float = 300.0,
    backoff_multiplier: float = 2.0,
    retryable_status_codes: tuple = (502, 503, 504),
    *args,
    **kwargs
):
    """
    Execute a function with exponential backoff retry logic.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 30)
        max_delay: Maximum delay in seconds (default: 300)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        retryable_status_codes: HTTP status codes that should trigger retry
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result from func if successful
        
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except requests.HTTPError as e:
            last_exception = e
            
            # Check if this is a retryable error
            if e.response and e.response.status_code in retryable_status_codes:
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                    print(f"[RETRY] HTTP {e.response.status_code} error. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.0f}s...")
                    sys.stdout.flush()
                    time.sleep(delay)
                    continue
                else:
                    print(f"[RETRY] Max retries ({max_retries}) exhausted for HTTP {e.response.status_code}")
                    sys.stdout.flush()
            else:
                # Non-retryable error (e.g., 404) - fail immediately
                status_code = e.response.status_code if e.response else "unknown"
                print(f"[ERROR] Non-retryable HTTP error {status_code} - failing immediately")
                sys.stdout.flush()
                raise
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                print(f"[RETRY] Connection error: {type(e).__name__}. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.0f}s...")
                sys.stdout.flush()
                time.sleep(delay)
                continue
            else:
                print(f"[RETRY] Max retries ({max_retries}) exhausted for connection error")
                sys.stdout.flush()
        except Exception as e:
            # Unexpected error - don't retry
            print(f"[ERROR] Unexpected error (not retrying): {type(e).__name__}: {e}")
            sys.stdout.flush()
            raise
    
    # All retries exhausted
    raise last_exception


@dataclass
class DownloadProgress:
    """Track download progress for a collection."""
    collection: str
    total_files: int
    completed_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, downloading, completed, failed
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'collection': self.collection,
            'total_files': self.total_files,
            'completed_files': self.completed_files,
            'failed_files': self.failed_files,
            'total_bytes': self.total_bytes,
            'downloaded_bytes': self.downloaded_bytes,
            'current_file': self.current_file,
            'status': self.status,
            'percent': round(self.completed_files / self.total_files * 100, 1) if self.total_files > 0 else 0,
            'errors': self.errors[-5:],  # Last 5 errors
        }


class LinkParser(HTMLParser):
    """Parse HTML to extract file links."""
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            for attr, val in attrs:
                if attr.lower() == 'href' and val:
                    decoded_val = unquote(val)
                    # print(f"[DEBUG] Found link: {decoded_val}")
                    self.links.append(decoded_val)


def check_aria2c_available() -> bool:
    """Check if aria2c is installed."""
    return shutil.which('aria2c') is not None





def build_forecast_url(year: int, month: int, day: int) -> str:
    """
    Build NCCS datashare Forecast URL for a specific date.
    Forecast logic: /forecast/ path, Date - 1 Day, H00 subdirectory.
    This is used as fallback when DAS data is not yet available.
    """
    from datetime import date, timedelta
    
    # Target date from arguments
    target_date = date(year, month, day)
    
    # Logic: Folder is named 1 day prior to target date (D-1)
    folder_date = target_date - timedelta(days=1)
    
    # Use H00 initialization (00 UTC)
    return f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y{folder_date.year:04d}/M{folder_date.month:02d}/D{folder_date.day:02d}/H00/"


def build_directory_url(year: int, month: int, day: int) -> str:
    """
    Build NCCS datashare URL for a specific date.
    Defaults to Forecast URL (D-1 H00) as primary source.
    """
    return build_forecast_url(year, month, day)


def fetch_directory_listing(url: str, timeout: int = 60) -> List[str]:
    """Fetch and parse directory listing from NCCS portal with retry logic."""
    print(f"[DIR] Fetching directory listing: {url}")
    sys.stdout.flush()
    
    def _fetch():
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp
    
    # Use retry logic for transient failures
    resp = exponential_backoff_retry(
        _fetch,
        max_retries=5,
        base_delay=30.0,
        max_delay=300.0
    )

    parser = LinkParser()
    parser.feed(resp.text)

    files = [href for href in parser.links if href.endswith('.nc4')]
    return list(dict.fromkeys(files))


def fetch_directory_listing_with_fallback(year: int, month: int, day: int, timeout: int = 60) -> Tuple[List[str], str]:
    """
    Fetch directory listing using the stable D-1 Forecast folder.
    
    Logic:
    - For future dates, it uses the forecast run from "Yesterday" (relative to today).
    - This folder is guaranteed to exist and contains 10 days of forecast data.
    """
    from datetime import date, timedelta, datetime, timezone
    
    target_date = date(year, month, day)
    # Use UTC today to align with NASA servers
    today_utc = datetime.now(timezone.utc).date()
    
    # Stable Forecast Logic: 
    # If target is in the future or today, use Yesterday (Today - 1) as the base.
    # Otherwise, use the canonical D-1 (Target - 1) for historical data.
    if target_date >= today_utc:
        forecast_base_date = today_utc - timedelta(days=1)
    else:
        forecast_base_date = target_date - timedelta(days=1)
        
    forecast_url = f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/forecast/Y{forecast_base_date.year:04d}/M{forecast_base_date.month:02d}/D{forecast_base_date.day:02d}/H00/"
    
    print(f"[FORECAST-STABLE] Using Base Folder (D-1): {forecast_url}")
    sys.stdout.flush()
    
    try:
        files = fetch_directory_listing(forecast_url, timeout)
        if files:
            # Filter for files matching the target date
            target_str_compact = target_date.strftime('%Y%m%d')
            matching = [f for f in files if target_str_compact in f]
            
            if matching:
                print(f"[SUCCESS] Found {len(matching)} matching files in Forecast Run {forecast_base_date}")
                sys.stdout.flush()
                return matching, forecast_url
            else:
                raise RuntimeError(f"Forecast run {forecast_base_date} does not yet/no longer contain data for {target_date}")
        else:
            raise RuntimeError(f"Forecast folder {forecast_url} is empty.")
    except Exception as e:
        error_msg = f"Failed to fetch forecast from {forecast_url}: {e}"
        print(f"[ERROR] {error_msg}")
        sys.stdout.flush()
        raise RuntimeError(error_msg)



def filter_collection_files(files: List[str], collection: str, allowed_hours: Optional[List[int]] = None, target_date_str: Optional[str] = None) -> List[str]:
    """
    Filter files for a specific collection at 3-hourly intervals.
    
    Args:
        files: List of all files
        collection: Collection name
        allowed_hours: List of hours to keep (e.g. [0, 6, 12, 18])
        target_date_str: Target date in YYYY-MM-DD format. If provided, filters files 
                        where the valid time matches this date.
    """
    import re
    matching = []
    
    # Convert YYYY-MM-DD to YYYYMMDD for filename matching
    target_date_compact = target_date_str.replace('-', '') if target_date_str else None
    
    for f in files:
        # Check collection match (handling inst3_2d_asm_Nx -> inst3_2d_smp_Nx mapping for Forecasts)
        if collection in f:
            pass
        elif collection == 'inst3_2d_asm_Nx' and 'inst3_2d_smp_Nx' in f:
            pass # Allow forecast variant
        else:
            continue
        
        # Regex to capture the VALID time (after the + sign)
        # Format: ...+YYYYMMDD_HHMM.V01...
        match = re.search(r'\+(\d{8})_(\d{2})(\d{2})\.', f)
        
        # Fallback: Try standard format without + (e.g. legacy or analysis)
        if not match:
            match = re.search(r'\.(\d{8})_(\d{2})(\d{2})\.', f)
        
        if match:
            file_date = match.group(1)
            file_hour = int(match.group(2))
            file_min = int(match.group(3))
            
            # Debug match to verify
            # print(f"[DEBUG] Parsed {f} -> Date: {file_date}, Hour: {file_hour}, Min: {file_min}")
            
            # Check date match if target date is provided
            if target_date_compact and file_date != target_date_compact:
                continue

            # Check if hour is allowed (skip for const as it is static/invariant)
            if allowed_hours is not None and 'const' not in collection and file_hour not in allowed_hours:
                continue
            
            if 'inst3' in collection:
                if file_hour in HOURS_3HOURLY and file_min == 0:
                    matching.append(f)
            elif 'tavg1' in collection:
                # Allow both 00 and 30 minutes for tavg1 to be robust
                if file_hour in TAVG1_3HR_HOURS and (file_min == 30 or file_min == 0):
                    matching.append(f)
            elif 'const' in collection:
                matching.append(f)  # Static - take all
                
        # Use fallback regex for non-forecast files (if any legacy files exist without +)
        elif 'const' in collection:
             matching.append(f)
    
    return sorted(matching)


def rename_smp_to_asm(filepath: str) -> str:
    """
    Rename inst3_2d_smp_Nx files to inst3_2d_asm_Nx for pipeline compatibility.
    
    Args:
        filepath: Path to the downloaded file
        
    Returns:
        New filepath if renamed, else original filepath
    """
    dirname, filename = os.path.split(filepath)
    
    # Check if it's a forecast file needing normalization
    if 'inst3_2d_smp_Nx' in filename:
        # We need to replace smp with asm
        # Example: GEOS.fp.fcst.inst3_2d_smp_Nx.20260111_00+... .nc4
        new_name = filename.replace('inst3_2d_smp_Nx', 'inst3_2d_asm_Nx')
        new_path = os.path.join(dirname, new_name)
        
        try:
            os.rename(filepath, new_path)
            print(f"[RENAME-FALLBACK] {filename} -> {new_name}")
            return new_path
        except OSError as e:
            print(f"[ERROR] Failed to rename {filename}: {e}")
            return filepath
            
    return filepath





def get_remote_file_size(session: requests.Session, url: str) -> int:
    """Get file size from server using HEAD request."""
    try:
        resp = session.head(url, timeout=30)
        return int(resp.headers.get('Content-Length', 0))
    except Exception:
        return 0


def download_with_resume(
    session: requests.Session,
    url: str,
    out_path: str,
    chunk_size: int = 1024 * 256,  # 256KB chunks
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[bool, str]:
    """
    Download file with resume capability using HTTP Range headers.
    
    Args:
        session: requests Session
        url: URL to download
        out_path: Output file path
        chunk_size: Download chunk size
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Check existing file
        existing_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        
        # Get total size from server
        head_resp = session.head(url, timeout=30)
        total_size = int(head_resp.headers.get('Content-Length', 0))
        
        # Check if already complete
        if existing_size > 0 and total_size > 0 and existing_size >= total_size:
            print(f"[SKIP] Already complete: {os.path.basename(out_path)}")
            if progress_callback:
                progress_callback(total_size, total_size)
            return True, "Already complete"
        
        # Setup Range header for resume
        headers = {}
        mode = 'wb'
        if existing_size > 0 and total_size > 0:
            headers['Range'] = f'bytes={existing_size}-'
            mode = 'ab'
            print(f"[RESUME] Resuming from {existing_size / 1e6:.1f}MB / {total_size / 1e6:.1f}MB")
        
        # Download
        with session.get(url, stream=True, headers=headers, timeout=300) as resp:
            resp.raise_for_status()
            
            # Update total if we got Content-Range
            content_range = resp.headers.get('Content-Range', '')
            if content_range:
                # Format: bytes 1234-5678/9999
                total_size = int(content_range.split('/')[-1])
            
            downloaded = existing_size
            
            with open(out_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_size)
        
        return True, f"Downloaded {downloaded / 1e6:.1f}MB"
        
    except Exception as e:
        return False, str(e)


def download_with_aria2c(
    urls: List[str],
    out_dir: str,
    connections: int = ARIA2C_CONNECTIONS,
    parallel: int = ARIA2C_PARALLEL,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Tuple[List[str], List[str]]:
    """
    Download multiple files using aria2c with real-time progress output.
    
    Args:
        urls: List of URLs to download
        out_dir: Output directory
        connections: Connections per file (default 16)
        parallel: Parallel file downloads (default 4)
        progress_callback: Optional callback(filename, completed, total)
        
    Returns:
        Tuple of (successful_files, failed_files)
    """
    import sys
    
    if not check_aria2c_available():
        raise RuntimeError("aria2c is not installed")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Create URL file for aria2c
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        url_file = f.name
        for url in urls:
            f.write(f"{url}\n")
    
    try:
        # Build aria2c command
        cmd = [
            'aria2c',
            f'--input-file={url_file}',
            f'--dir={out_dir}',
            f'--max-connection-per-server={connections}',
            f'--split={connections}',
            f'--max-concurrent-downloads={parallel}',
            f'--timeout={ARIA2C_TIMEOUT}',
            f'--max-tries={ARIA2C_RETRY}',
            '--continue=true',           # Resume support
            '--file-allocation=none',    # Prevent pre-allocation (avoids full-size empty files on fail)
            '--auto-file-renaming=false',
            '--allow-overwrite=true',    # Overwrite stale partial files
            '--conditional-get=true',    # Skip completed files
            '--console-log-level=notice',
            '--summary-interval=5',      # Progress every 5 seconds
        ]
        
        print(f"[ARIA2C] Starting download of {len(urls)} files")
        print(f"[ARIA2C] Connections: {connections}/file, Parallel: {parallel} files")
        sys.stdout.flush()
        
        # Run aria2c with real-time output using Popen
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        # Stream output in real-time
        last_summary_time = 0
        import time

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                # Reduce verbosity: Only print overall summaries and completion/error
                
                # '***' usually denotes the periodic summary line in aria2c
                # e.g. [*** Downloaded 100.0MiB of 500.0MiB (20%) CN:4 ... ***]
                if '***' in line and 'Downloaded' in line:
                    # Optional: Rate limit summary prints if they come too fast
                    current_time = time.time()
                    if current_time - last_summary_time > 10: # Print every 10s max
                        print(f"[ARIA2C] PROGRESS: {line}")
                        sys.stdout.flush()
                        last_summary_time = current_time
                
                elif 'Download complete' in line:
                    print(f"[ARIA2C] FILE COMPLETE: {line}")
                    sys.stdout.flush()
                    
                elif 'ERR' in line or 'Error' in line or 'Failed' in line:
                    print(f"[ARIA2C] ERROR: {line}")
                    sys.stdout.flush()
                
                # Keep initial seeding/starting messages if needed
                elif 'NOTICE' in line and 'Downloading' in line:
                     # Only print if it's not a noisy chunk update
                     pass

        
        process.wait()
        
        print(f"[ARIA2C] Process completed with return code: {process.returncode}")
        sys.stdout.flush()
        
        # Check which files were downloaded
        successful = []
        failed = []
        
        for url in urls:
            filename = url.split('/')[-1]
            filepath = os.path.join(out_dir, filename)
            aria2_file = filepath + '.aria2'
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                successful.append(filepath)
                # Clean up .aria2 progress file if download succeeded
                if os.path.exists(aria2_file):
                    try:
                        os.unlink(aria2_file)
                        print(f"[CLEANUP] Removed progress file: {aria2_file}")
                    except Exception:
                        pass
                
                # Check if we need to rename SMP -> ASM (fallback case)
                if 'inst3_2d_smp_Nx' in filepath:
                    filepath = rename_smp_to_asm(filepath)
            else:
                failed.append(url)
        
        return successful, failed
        
    finally:
        # Cleanup temp file
        try:
            os.unlink(url_file)
        except Exception:
            pass


def cleanup_aria2_files(directory: str) -> int:
    """
    Clean up any leftover .aria2 progress files in a directory.
    
    Args:
        directory: Directory to clean
        
    Returns:
        Number of files cleaned up
    """
    cleaned = 0
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith('.aria2'):
                aria2_path = os.path.join(directory, f)
                # Check if the actual file exists and is complete
                actual_file = aria2_path[:-6]  # Remove .aria2 extension
                if os.path.exists(actual_file) and os.path.getsize(actual_file) > 0:
                    try:
                        os.unlink(aria2_path)
                        print(f"[CLEANUP] Removed stale: {f}")
                        cleaned += 1
                    except Exception:
                        pass
    return cleaned


def copy_static_const_file(out_dir: str, date_str: str) -> Dict:
    """
    Copy the static const_2d_asm_Nx file instead of downloading.
    
    The const file is static (never changes) and is stored locally in the
    transformation directory. This function copies it to the output directory
    with the correct naming convention.
    
    Args:
        out_dir: Output directory
        date_str: Date in YYYY-MM-DD format (used for filename)
        
    Returns:
        Dict with copy results
    """
    import shutil
    
    collection = 'const_2d_asm_Nx'
    
    progress = DownloadProgress(
        collection=collection,
        total_files=1,
        status='copying'
    )
    
    # Source file location (in transformation directory)
    # The file is named MERRA2_101.const_2d_asm_Nx.00000000.nc4
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(script_dir, 'transformation', 'MERRA2_101.const_2d_asm_Nx.00000000.nc4')
    
    # Also check alternative locations
    alt_locations = [
        '/opt/airflow/transformation/MERRA2_101.const_2d_asm_Nx.00000000.nc4',
        '/app/transformation/MERRA2_101.const_2d_asm_Nx.00000000.nc4',
        os.path.join(os.getcwd(), 'transformation', 'MERRA2_101.const_2d_asm_Nx.00000000.nc4'),
    ]
    
    # Find the source file
    actual_source = None
    if os.path.exists(source_file):
        actual_source = source_file
    else:
        for alt in alt_locations:
            if os.path.exists(alt):
                actual_source = alt
                break
    
    if not actual_source:
        progress.status = 'failed'
        progress.errors.append(f"Static const file not found. Checked: {source_file} and alternatives")
        print(f"[ERROR] Static const file not found!")
        print(f"[ERROR] Expected at: {source_file}")
        return progress.to_dict()
    
    # Target filename: GEOS.fp.asm.const_2d_asm_Nx.00000000.nc4
    # (matches the pattern expected by regridder)
    target_filename = 'GEOS.fp.asm.const_2d_asm_Nx.00000000.nc4'
    target_path = os.path.join(out_dir, target_filename)
    
    print(f"\n{'='*60}")
    print(f"[COLLECTION] {collection} (STATIC)")
    print(f"[INFO] Copying static constants file (not downloaded daily)")
    print(f"[SOURCE] {actual_source}")
    print(f"[TARGET] {target_path}")
    print(f"{'='*60}")
    
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        # Check if already copied
        if os.path.exists(target_path):
            source_size = os.path.getsize(actual_source)
            target_size = os.path.getsize(target_path)
            if source_size == target_size:
                print(f"[SKIP] Static const file already exists: {target_filename}")
                progress.status = 'completed'
                progress.completed_files = 1
                return progress.to_dict()
        
        # Copy the file
        shutil.copy2(actual_source, target_path)
        progress.status = 'completed'
        progress.completed_files = 1
        print(f"[OK] Copied static const file: {target_filename}")
        
    except Exception as e:
        progress.status = 'failed'
        progress.errors.append(f"Copy failed: {str(e)}")
        print(f"[ERROR] Failed to copy static const file: {e}")
    
    return progress.to_dict()


def download_collection(
    collection: str,
    date_str: str,
    out_dir: str,
    use_aria2c: bool = True,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    hours: Optional[List[int]] = None
) -> Dict:
    """
    Download a single GEOS-FP collection.
    
    Args:
        collection: Collection name (e.g., 'inst3_2d_asm_Nx')
        date_str: Date in YYYY-MM-DD format
        out_dir: Output directory
        use_aria2c: Use aria2c if available (default True)
        max_workers: Workers for fallback ThreadPool download
        progress_callback: Progress callback for Airflow XCom
        
    Returns:
        Dict with download results
    """
    # Handle static const file specially
    if collection == 'const_2d_asm_Nx':
        # Try copying first
        result = copy_static_const_file(out_dir, date_str)
        if result.get('status') == 'completed':
            return result
        
        print("[WARN] Local static const file not found (or copy failed). Downloading specific static file from NASA...")
        # Direct download of the specific invariant file
        target_path = os.path.join(out_dir, "GEOS.fp.asm.const_2d_asm_Nx.00000000.nc4")
        static_url = "https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4"
        
        try:
            os.makedirs(out_dir, exist_ok=True)
            # requests is already imported globally
            print(f"[DOWNLOAD] Fetching static file from: {static_url}")
            resp = requests.get(static_url, stream=True, timeout=300)
            if resp.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                # Validation: Check if it's a valid file (> 1KB)
                if os.path.exists(target_path) and os.path.getsize(target_path) < 1000:
                    print(f"[ERROR] Downloaded file is too small ({os.path.getsize(target_path)} bytes). Likely HTML error.")
                    os.remove(target_path)
                    return {'status': 'failed', 'errors': ["File too small (HTML)"]}
                
                print(f"[OK] Downloaded static const file to: {target_path}")
                return {'status': 'completed', 'completed_files': 1, 'total_files': 1}
            else:
                print(f"[ERROR] Failed to download static file: HTTP {resp.status_code}")
                return {'status': 'failed', 'errors': [f"HTTP {resp.status_code}"]}
        except Exception as e:
            print(f"[ERROR] Exception downloading static file: {e}")
            return {'status': 'failed', 'errors': [str(e)]}

    
    # Clean up any stale .aria2 files from previous interrupted runs
    if os.path.exists(out_dir):
        cleaned = cleanup_aria2_files(out_dir)
        if cleaned > 0:
            print(f"[CLEANUP] Removed {cleaned} stale .aria2 files")
    
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Initialize progress
    progress = DownloadProgress(
        collection=collection,
        total_files=PRITHVI_GEOSFP_PRODUCTS[collection]['expected_files'],
        status='downloading'
    )
    
    if progress_callback:
        progress_callback(progress)
    
    print(f"\n{'='*60}")
    print(f"[COLLECTION] {collection}")
    print(f"[INFO] {PRITHVI_GEOSFP_PRODUCTS[collection]['description']}")
    print(f"[INFO] Expected: {progress.total_files} files, ~{PRITHVI_GEOSFP_PRODUCTS[collection]['approx_size_mb']}MB each")
    print(f"{'='*60}")
    
    # Get directory listing with automatic DAS → Forecast fallback
    try:
        all_files, base_url = fetch_directory_listing_with_fallback(dt.year, dt.month, dt.day)
        print(f"[INFO] Using URL: {base_url}")
        sys.stdout.flush()
    except Exception as e:
        progress.status = 'failed'
        progress.errors.append(f"Failed to fetch directory: {e}")
        if progress_callback:
            progress_callback(progress)
        return progress.to_dict()
    
    # Filter for this collection
    collection_files = filter_collection_files(all_files, collection, hours, date_str)
    
    # [NEW] Check for DAS incompleteness fallback to Forecast
    # If we used DAS (analysis) but found FEWER files than expected (e.g. 09:00 missing),
    # try the Forecast URL which might have the prediction for this date (from yesterday).
    
    expected_count = len(hours) if hours else PRITHVI_GEOSFP_PRODUCTS[collection]['expected_files']
    
    # [GAP FILLING] Check if we have all requested hours
    # If using DAS (Analysis), we might be missing recent hours (e.g. 15:00 while 09:00 is there).
    # We should fill missing hours from Forecast.
    if hours and "/das/" in base_url:
        found_hours = set()
        import re
        for f in collection_files:
            m = re.search(r'[\._](\d{2})\d{2}\.', f)
            if m:
                found_hours.add(int(m.group(1)))
            else:
                 # Fallback regex for + format in case mixed
                 m = re.search(r'\+(\d{8})_(\d{2})(\d{2})\.', f)
                 if m:
                     found_hours.add(int(m.group(2)))
        
        missing_hours = [h for h in hours if h not in found_hours]
        
        # Only trigger gap fill if we actually have missing hours
        if missing_hours:
            print(f"[GapFill] DAS incomplete ({len(collection_files)}/{expected_count}). Missing: {missing_hours}. Checking Forecast...")
            try:
                fcst_url = build_forecast_url(dt.year, dt.month, dt.day)
                print(f"[GapFill] Fetching Forecast listing: {fcst_url}")
                fcst_files_raw = fetch_directory_listing(fcst_url)
                
                # Check different collection names for forecast (smp vs asm)
                check_collections = [collection]
                if collection == 'inst3_2d_asm_Nx':
                    check_collections.append('inst3_2d_smp_Nx')
                
                added_files = []
                for fcst_coll in check_collections:
                     found = filter_collection_files(fcst_files_raw, fcst_coll, missing_hours, date_str)
                     if found:
                         # Found files for missing hours!
                         # Convert to FULL URLs immediately
                         for f in found:
                             added_files.append(fcst_url + f)
                         
                         # If we found files for missing hours in one variant, we can stop checking variants
                         # But we should ensure we found ALL missing hours? 
                         # For simplicity, if we found anything, we add it. 
                         break 
                
                if added_files:
                    print(f"[GapFill] Found {len(added_files)} missing files in Forecast")
                    
                    # Convert existing DAS files to full URLs
                    full_url_list = [base_url + f for f in collection_files] + added_files
                    
                    # Replace collection_files with full URLs and clear base_url
                    collection_files = full_url_list 
                    base_url = "" 
                    
                else:
                    print(f"[GapFill] Still could not find files for hours: {missing_hours}")

            except Exception as e:
                print(f"[GapFill] Error checking forecast: {e}")
    
    # [Legacy Fallback Logic - Keep for cases where DAS is totally empty/missing]
    # If collection is totally empty in DAS (or we are not using DAS), try switching entirely to Forecast
    # But only if we haven't already done the GapFill (base_url != "")
    elif ("/das/" in base_url) and (len(collection_files) < expected_count) and base_url != "":
        print(f"[WARN] DAS incomplete ({len(collection_files)}/{expected_count} files) and GapFill skipped/failed. Checking Forecast...")
        
        try:
            fcst_url = build_forecast_url(dt.year, dt.month, dt.day)
            print(f"[FALLBACK] Checking Forecast URL: {fcst_url}")
            fcst_files = fetch_directory_listing(fcst_url)
            fcst_matches = filter_collection_files(fcst_files, collection, hours, date_str)
            
            # If Forecast has MORE files than DAS, switch entirely
            if len(fcst_matches) > len(collection_files):
                print(f"[FALLBACK] Switching to Forecast (Found {len(fcst_matches)} files, DAS had {len(collection_files)})")
                collection_files = fcst_matches
                base_url = fcst_url
            else:
                 print(f"[FALLBACK] Forecast does not have more files ({len(fcst_matches)}). Sticking with DAS.")
        except Exception as e:
            print(f"[FALLBACK] Failed to check Forecast: {e}")

    progress.total_files = len(collection_files)
    
    # Fallback Logic for inst3_2d_asm_Nx -> inst3_2d_smp_Nx
    if not collection_files and collection == 'inst3_2d_asm_Nx':
        print(f"[WARN] No files found for {collection}. Checking fallback: inst3_2d_smp_Nx")
        fallback_collection = 'inst3_2d_smp_Nx'
        fallback_files = filter_collection_files(all_files, fallback_collection, hours, date_str)
        
        if fallback_files:
            print(f"[FALLBACK] Found {len(fallback_files)} files in {fallback_collection}")
            print(f"[FALLBACK] Will download and rename to {collection}")
            collection_files = fallback_files
            progress.total_files = len(collection_files)
            # NOTE: We keep 'collection' variable as checks downstream might rely on it,
            # but the URLs will be built from 'collection_files' which contain 'smp'.
        else:
            print(f"[FALLBACK] No files found for fallback {fallback_collection} either.")

    if not collection_files:
        progress.status = 'failed'
        progress.errors.append(f"No files found for collection {collection}")
        if progress_callback:
            progress_callback(progress)
        return progress.to_dict()
    
    print(f"[FILES] Found {len(collection_files)} files")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Build full URLs
    urls = [base_url + f for f in collection_files]
    
    # Check which files need downloading
    files_to_download = []
    session = requests.Session()
    
    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(out_dir, filename)
        
        if os.path.exists(filepath):
            # Check if complete (Assume if exists > 0 bytes it's valid to save time)
            local_size = os.path.getsize(filepath)
            
            if local_size > 0:
                print(f"[SKIP] Found local file: {filename} ({local_size/1e6:.1f}MB)")
                progress.completed_files += 1
                continue
            else:
                print(f"[RE-DOWNLOAD] Found empty file: {filename}")
        
        files_to_download.append(url)
    
    if not files_to_download:
        print(f"[DONE] All files already downloaded for {collection}")
        progress.status = 'completed'
        if progress_callback:
            progress_callback(progress)
        return progress.to_dict()
    
    print(f"[DOWNLOAD] {len(files_to_download)} files to download")
    sys.stdout.flush()
    
    # Download using aria2c or fallback
    if use_aria2c and check_aria2c_available():
        print("[METHOD] Using aria2c (fast parallel download)")
        sys.stdout.flush()
        successful, failed = download_with_aria2c(
            files_to_download,
            out_dir,
            connections=ARIA2C_CONNECTIONS,
            parallel=ARIA2C_PARALLEL if collection != 'inst3_3d_asm_Nv' else 3  # 3 parallel for large 3D files
        )
        progress.completed_files += len(successful)
        progress.failed_files = len(failed)
        for f in failed:
            progress.errors.append(f"Failed: {f}")
    else:
        print("[METHOD] Using Python ThreadPoolExecutor (fallback)")
        
        def download_one(url):
            filename = url.split('/')[-1]
            filepath = os.path.join(out_dir, filename)
            progress.current_file = filename
            
            success, msg = download_with_resume(session, url, filepath)
            
            if success and 'inst3_2d_smp_Nx' in filepath:
                rename_smp_to_asm(filepath)
                
            return url, success, msg
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, url): url for url in files_to_download}
            
            for future in as_completed(futures):
                url, success, msg = future.result()
                filename = url.split('/')[-1]
                
                if success:
                    progress.completed_files += 1
                    print(f"[OK] {filename}: {msg}")
                else:
                    progress.failed_files += 1
                    progress.errors.append(f"{filename}: {msg}")
                    print(f"[FAIL] {filename}: {msg}")
                
                if progress_callback:
                    progress_callback(progress)
    
    # Final status
    if progress.failed_files == 0:
        progress.status = 'completed'
    elif progress.completed_files > 0:
        progress.status = 'partial'
    else:
        progress.status = 'failed'
    
    if progress_callback:
        progress_callback(progress)
    
    print(f"\n[RESULT] {collection}: {progress.completed_files}/{progress.total_files} completed, {progress.failed_files} failed")
    
    return progress.to_dict()


def download_all_collections(
    date_str: str,
    out_dir: str,
    collections: Optional[List[str]] = None,
    use_aria2c: bool = True,
    progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict:
    """
    Download all GEOS-FP collections for Prithvi.
    
    Args:
        date_str: Date in YYYY-MM-DD format
        out_dir: Output directory
        collections: List of collections to download (default: all)
        use_aria2c: Use aria2c if available
        progress_callback: Progress callback for each collection
        
    Returns:
        Dict with all results
    """
    collections = collections or list(PRITHVI_GEOSFP_PRODUCTS.keys())
    
    results = {}
    for collection in collections:
        result = download_collection(
            collection=collection,
            date_str=date_str,
            out_dir=out_dir,
            use_aria2c=use_aria2c,
            progress_callback=lambda p: progress_callback({collection: p.to_dict()}) if progress_callback else None
        )
        results[collection] = result
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Parallel GEOS-FP downloader with aria2c support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--date', '-d', type=str, required=True,
                       help='Date in YYYY-MM-DD format')
    parser.add_argument('--out', '-o', default='geosfp_data',
                       help='Output directory')
    parser.add_argument('--collection', '-c', type=str, default=None,
                       choices=list(PRITHVI_GEOSFP_PRODUCTS.keys()),
                       help='Download single collection')
    parser.add_argument('--no-aria2c', action='store_true',
                       help='Disable aria2c, use Python fallback')
    parser.add_argument('--dry-run', action='store_true',
                       help='List files only, do not download')
    parser.add_argument('--workers', type=int, default=4,
                       help='Workers for Python fallback (default: 4)')
    parser.add_argument('--hours', type=int, nargs='+', default=None,
                       help='Specific hours to download (e.g. 0 3 6)')
    
    args = parser.parse_args()
    
    # Parse date
    try:
        dt = datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print("[ERROR] Invalid date format. Use YYYY-MM-DD")
        sys.exit(1)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           PARALLEL GEOS-FP DOWNLOADER FOR PRITHVI-WxC                ║
║                                                                      ║
║  Date: {args.date}                                                   ║
║  aria2c: {'Available' if check_aria2c_available() else 'Not found (using fallback)'}                                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if args.dry_run:
        base_url = build_directory_url(dt.year, dt.month, dt.day)
        all_files = fetch_directory_listing(base_url)
        
        for collection in (args.collection and [args.collection]) or PRITHVI_GEOSFP_PRODUCTS.keys():
            files = filter_collection_files(all_files, collection, args.hours)
            print(f"\n[{collection}] {len(files)} files:")
            for f in files:
                print(f"  - {f}")
        return
    
    # Download
    use_aria2c = not args.no_aria2c
    
    if args.collection:
        result = download_collection(
            collection=args.collection,
            date_str=args.date,
            out_dir=args.out,
            use_aria2c=use_aria2c,
            max_workers=args.workers,
            hours=args.hours
        )
        print(f"\n[FINAL] {json.dumps(result, indent=2)}")
    else:
        results = download_all_collections(
            date_str=args.date,
            out_dir=args.out,
            use_aria2c=use_aria2c
        )
        print(f"\n[FINAL] Results:")
        for coll, res in results.items():
            status = res.get('status', 'unknown')
            completed = res.get('completed_files', 0)
            total = res.get('total_files', 0)
            print(f"  {coll}: {status} ({completed}/{total})")


if __name__ == '__main__':
    main()
