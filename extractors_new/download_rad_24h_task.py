
import os
import sys
import re
from datetime import datetime, timedelta, timezone

# Try importing from parallel_downloader. 
# We assume this script is in the same directory or parallel_downloader is in the python path.

# Try importing from parallel_downloader. 
# We assume this script is in the same directory or parallel_downloader is in the python path.
try:
    from parallel_downloader import (
        build_directory_url,
        fetch_directory_listing,
        fetch_directory_listing_with_fallback,
        check_aria2c_available,
        download_with_aria2c,
    )
except ImportError:
    # If run directly or path issues, try appending current dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from parallel_downloader import (
        build_directory_url,
        fetch_directory_listing,
        fetch_directory_listing_with_fallback,
        check_aria2c_available,
        download_with_aria2c,
    )

COLLECTION = 'tavg1_2d_rad_Nx'

def extract_datetime_from_filename(filename):
    """
    Extract date and time from filename.
    Format example: GEOS.fp.fcst.tavg1_2d_rad_Nx.20260201_00+20260201_0030.V01.nc4
    We care about the VALID time (second timestamp).
    """
    # Regex to capture the VALID time (second part after + or .)
    # Format: ...+YYYYMMDD_HHMM.V01... or .YYYYMMDD_HHMM.V01...
    
    # Try with '+' first (Forecast)
    match = re.search(r'\+(\d{8})_(\d{2})(\d{2})\.', filename)
    if not match:
        # Try with '.' (Analysis/Legacy)
        match = re.search(r'\.(\d{8})_(\d{2})(\d{2})\.', filename)
        
    if match:
        d_str = match.group(1)
        h_str = match.group(2)
        m_str = match.group(3)
        try:
            dt = datetime.strptime(f"{d_str}{h_str}{m_str}", "%Y%m%d%H%M")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None

def download_rad_24h_logic(execution_input, output_base_dir, time_folder=None, custom_date_folder=None):
    """
    Download 24h rad files starting from the specific execution time.
    
    Args:
        execution_input: Can be a datetime object OR a date string (YYYY-MM-DD).
                         If date string, assumes start time is 00:00 of that day.
        output_base_dir: Base data directory.
        time_folder: Optional subfolder for time isolation (e.g. HHMM).
        custom_date_folder: Optional string to force a specific date folder name (DD-MM-YYYY).
        
    Returns:
        dict: Status compatible with pipeline aggregation.
    """
    
    # normalize input to start_datetime
    if isinstance(execution_input, str):
        start_dt = datetime.strptime(execution_input, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    elif isinstance(execution_input, datetime):
        start_dt = execution_input
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    else:
        raise ValueError("execution_input must be date string or datetime object")
        
    # Define window: [Start, Start + 24 Hours]
    end_dt = start_dt + timedelta(hours=24)
    
    # Output directory - Store in date folder of the START date (or custom override)
    # Keeps it organized by the "Run Date"
    if custom_date_folder:
        date_folder = custom_date_folder
    else:
        date_folder = start_dt.strftime('%d-%m-%Y')
    
    if time_folder:
        out_dir = os.path.join(output_base_dir, 'rad_files', date_folder, time_folder)
    else:
        out_dir = os.path.join(output_base_dir, 'rad_files', date_folder)
    
    print(f"[RAD-24H] Time Window: {start_dt} to {end_dt}")
    print(f"[RAD-24H] Target Output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Identify unique days we need to fetch from NASA
    # (Could be 1 day if 00:00, or 2 days if crossing midnight)
    current = start_dt
    days_to_check = set()
    
    # Check start day (maps to D-1 Forecast, which is stable)
    days_to_check.add(current.date())
    
    # User requested to SKIP checking the end day/next day catalog 
    # to avoid latency issues with fresh forecasts. 
    # The D-1 Forecast covers the full 24h window.
    # HOWEVER: The underlying fetch_directory_listing_with_fallback strictly limits 
    # results to the requested day. So we MUST check both days to get the full 24h set.
    days_to_check.add(end_dt.date())
    
    sorted_days = sorted(list(days_to_check))
    
    # Use dict to deduplicate by valid time (keeps latest fresher forecast if overlap)
    files_to_download_map = {}
    
    # Cache for directory listings to avoid redundant fetches
    catalog_cache = {}
    
    for d in sorted_days:
        print(f"[RAD-24H] Checking catalog for: {d}")
        try:
             # Calculate the expected base URL for this day to check cache
             # (Simplified check: if we already fetched a forecast for 'today', skip)
             # But it's safer to use the return of the fallback function.
             # We'll just cache the results of fetch_directory_listing_with_fallback.
             cache_key = d.strftime('%Y%m%d')
             
             if cache_key in catalog_cache:
                 all_files, base_url = catalog_cache[cache_key]
             else:
                 all_files, base_url = fetch_directory_listing_with_fallback(d.year, d.month, d.day)
                 catalog_cache[cache_key] = (all_files, base_url)
             
             # Filter matches
             for f in all_files:
                 if COLLECTION not in f:
                     continue
                 
                 file_dt = extract_datetime_from_filename(f)
                 if file_dt:
                     # Logic: Is this file within [start, end]? 
                     if start_dt <= file_dt <= end_dt:
                         # Store in map (overwrites earlier entries with later ones)
                         files_to_download_map[file_dt] = base_url + f
                         
        except Exception as e:
             print(f"[RAD-24H] Error fetching directory for {d}: {e}")
             pass


    # Sort by time to be deterministic
    files_to_download_urls = [url for dt, url in sorted(files_to_download_map.items())]

    total_files = len(files_to_download_urls)
    print(f"[RAD-24H] Found {total_files} files in range.")
    
    if total_files == 0:
        print("[ERROR] No radiation files found in time window. Failing task.")
        raise RuntimeError("No radiation files found in time window. Check logs for directory fetch errors.")

    # Filter out existing files
    final_urls = []
    completed = 0
    
    for url in files_to_download_urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(out_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            completed += 1
        else:
            final_urls.append(url)
            
    if not final_urls:
        print("[RAD-24H] All files already exist.")
        return {
            "collection": "rad_24h",
            "status": "completed",
            "completed": total_files,
            "total": total_files,
            "output_dir": out_dir
        }

    print(f"[RAD-24H] Downloading {len(final_urls)} files...")
    
    failed_files = []
    if check_aria2c_available():
         print("[RAD-24H] Using aria2c")
         succ, fail = download_with_aria2c(final_urls, out_dir)
         completed += len(succ)
         failed_files = fail
    else:
        print("[RAD-24H] Using requests fallback")
        import requests
        for url in final_urls:
            filename = url.split('/')[-1]
            filepath = os.path.join(out_dir, filename)
            try:
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                    completed += 1
                    print(f"[OK] {filename}")
                else:
                    print(f"[FAIL] {filename} HTTP {r.status_code}")
                    failed_files.append(url)
            except Exception as e:
                print(f"[FAIL] {filename} {e}")
                failed_files.append(url)
    
    status = "completed"
    if failed_files:
        status = "partial" if completed > 0 else "failed"
        
    return {
        "collection": "rad_24h",
        "status": status,
        "completed": completed,
        "total": total_files,
        "output_dir": out_dir
    }
