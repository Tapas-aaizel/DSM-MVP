#!/usr/bin/env python3
import os
import re
import json
import unicodedata
import requests
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ============================================================================
# CONFIGURATION: GUJARAT & RAJASTHAN DSM (1km)
# ============================================================================
# Set environment variables to prevent multi-threading within individual workers
# This is crucial for stability when using ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

GRID_RES = 0.009  # ~1km Resolution
LAT_MIN, LAT_MAX = 27.0, 28.0
LON_MIN, LON_MAX = 71.4, 72.4

BASE_URL = "https://www.weather-india.in"
INDIA_INDEX_URL = "https://www.weather-india.in/en/india/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
MAX_WORKERS = 20 # Reduced from 50 to lower network/threading overhead

# ============================================================================
# METEOROLOGICAL & CLEANING UTILS
# ============================================================================
def dir_to_degree(dir_str):
    """Converts compass directions to mathematical degrees."""
    mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    return mapping.get(str(dir_str).upper().strip(), 0)

def percent_to_okta(p):
    try: return (float(p) / 100.0) * 8.0
    except: return 0.0

def clean_station_name(text):
    if not text: return "Unknown"
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-zA-Z0-9\s-]', '', text).strip().title()

def clean_value(text):
    try:
        val = re.sub(r'[^\d.]', '', str(text))
        return float(val) if val else 0.0
    except: return 0.0

# ============================================================================
# PHASE 1: FILTERED CRAWLER (GJ & RJ ONLY)
# ============================================================================
def get_state_station_links():
    print("🌍 Step 1: Mapping Stations for Gujarat & Rajasthan...")
    target_states = ['rajasthan']
    try:
        res = requests.get(INDIA_INDEX_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        regions = []
        for a in soup.find_all('a', href=True):
            if '(' in a.text and '/india/' in a['href']:
                slug = a['href'].replace('/en/', '/').replace('/india/', '').strip('/')
                if any(state in slug.lower() for state in target_states):
                    regions.append({'url': BASE_URL + a['href'], 'slug': slug})
        
        all_links = []
        for reg in regions:
            page = 1
            while True:
                url = reg['url'] if page == 1 else f"{reg['url'].rstrip('/')}/{page}"
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code != 200: break
                s = BeautifulSoup(r.content, 'html.parser')
                found = 0
                for a in s.find_all('a', href=True):
                    href = a['href'].split('?')[0]
                    if reg['slug'] in href and 'weather-' in href:
                        if not any(x in href for x in ['tomorrow', '10-days', 'map', 'hourly']):
                            full_url = f"{BASE_URL}{href.rstrip('/')}/weather-hourly"
                            if not any(l['url'] == full_url for l in all_links):
                                all_links.append({'name': a.text.strip(), 'url': full_url})
                                found += 1
                if found == 0: break
                page += 1
        print(f"✅ Found {len(all_links)} stations")
        return all_links
    except Exception as e:
        print(f"❌ Error getting station links: {e}")
        return []

def scrape_worker(station):
    try:
        r = requests.get(station['url'], headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        lat, lon = None, None
        for s in soup.find_all('script', type='application/ld+json'):
            d = json.loads(s.string)
            if isinstance(d, list): d = d[0]
            if d.get('@type') == 'Place' and 'geo' in d:
                lat, lon = float(d['geo']['latitude']), float(d['geo']['longitude'])
                break
        if lat is None: return []

        rows = []
        h_cont = soup.find_all('div', class_='values-parent')
        for idx, cont in enumerate(h_cont):
            date_obj = datetime.now().date() + timedelta(days=idx)
            hrs = [h.text.strip() for h in cont.find_all('div', class_=re.compile(r'\bhour\b'))]
            temps = [d.text.strip() for d in cont.find_all('div', class_=lambda x: x and 'temperature' in x and 'feels_like' not in x)]
            winds = [d.text.strip() for d in cont.find_all('div', class_=re.compile(r'\bwind_speed\b'))]
            dirs = [d.find('span').text.strip() if d.find('span') else "N" for d in cont.find_all('div', class_=re.compile(r'\bwind_icon\b'))]
            clouds = [d.text.strip() for d in cont.find_all('div', class_=re.compile(r'\bcloudiness_percent\b'))]

            for i in range(len(hrs)):
                if ":" not in hrs[i]: continue
                dt_ist = datetime.strptime(f"{date_obj} {hrs[i]}", '%Y-%m-%d %H:%M')
                
                # --- TIME CONVERSION: IST to UTC ---
                dt_utc = dt_ist - timedelta(hours=5, minutes=30)
                
                speed = clean_value(winds[i])
                deg = dir_to_degree(dirs[i])
                rad = np.deg2rad(deg)
                
                rows.append({
                    'station_id': clean_station_name(station['name']),
                    'Lat': lat, 'Lon': lon, 
                    'Timestamp_UTC': dt_utc,
                    'T2M': clean_value(temps[i]) + 273.15,
                    'WS10': speed,
                    'U10': -speed * np.sin(rad),
                    'V10': -speed * np.cos(rad),
                    'TCC': percent_to_okta(clean_value(clouds[i]))
                })
        return rows
    except: return []

# ============================================================================
# PHASE 2: TEMPORAL EXPANSION
# ============================================================================
def expand_temporal(df, start_date, end_date):
    print("🕐 Step 3: Expanding to 15-minute intervals...")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    expanded_list = []
    for sid, group in df.groupby('station_id'):
        group = group.sort_values('Timestamp_UTC').drop_duplicates('Timestamp_UTC')
        lat, lon = group['Lat'].iloc[0], group['Lon'].iloc[0]
        group = group.set_index('Timestamp_UTC')
        resampled = group.resample('15min').asfreq().infer_objects(copy=False)
        nums = resampled.select_dtypes(include=[np.number]).columns
        resampled[nums] = resampled[nums].interpolate(method='linear')
        resampled['station_id'], resampled['Lat'], resampled['Lon'] = sid, lat, lon
        
        # Filtering range based on UTC date
        range_data = resampled[(resampled.index.date >= start_dt) & (resampled.index.date <= end_dt)]
        expanded_list.append(range_data)
    
    print(f"✅ Temporal expansion complete")
    return pd.concat(expanded_list).reset_index()

# ============================================================================
# PHASE 3: OPTIMIZED GAUSSIAN KRIGING
# ============================================================================
# Global variables to be shared with worker processes via COW (on Linux)
_GX = None
_GY = None
_X_GRID = None

def init_kriging_worker(gx, gy):
    """Initializes worker with the grid coordinates to avoid re-passing them."""
    global _GX, _GY, _X_GRID
    _GX = gx
    _GY = gy
    _X_GRID = np.column_stack((_GX.ravel(), _GY.ravel()))

def interpolate_gaussian_multi(x, y, data_matrix):
    """
    Interpolates multiple variables at once using a shared Gaussian Process kernel.
    """
    global _X_GRID, _GX
    
    # Filter points where any variable is NaN
    mask = ~np.any(np.isnan(data_matrix), axis=1)
    if np.sum(mask) < 5: 
        return {i: np.full(_GX.shape, np.nan, dtype=np.float32) for i in range(data_matrix.shape[1])}
        
    X_train = np.column_stack((x[mask], y[mask]))
    y_train = data_matrix[mask]
    
    # Use a shared kernel for all variables
    kernel = C(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
    
    # optimize_restarts=0 to save time/CPU
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    gp.fit(X_train, y_train)
    
    # Predict on the pre-shared high-res grid in chunks to cap memory usage per worker
    chunk_size = 250000
    num_points = _X_GRID.shape[0]
    num_vars = data_matrix.shape[1]
    y_pred = np.zeros((num_points, num_vars), dtype=np.float32)
    
    for i in range(0, num_points, chunk_size):
        end_idx = min(i + chunk_size, num_points)
        y_pred[i:end_idx] = gp.predict(_X_GRID[i:end_idx]).astype(np.float32)
    
    results = {}
    for i in range(num_vars):
        results[i] = y_pred[:, i].reshape(_GX.shape)
        
    return results

# ============================================================================
# ARGUMENT PARSING & PIPELINE ADAPTER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate DSM for Gujarat and Rajasthan")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--slots", type=str, default="NA", help="Ignored but kept for compatibility")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    return parser.parse_args()


# ============================================================================
# PARALLEL PROCESSING HELPER
# ============================================================================
def process_timestep(ts, lat, lon, data_matrix, variables):
    """
    Helper function to process a single timestamp in parallel.
    Now uses shared global grid to save memory.
    """
    if len(lat) < 5:
        return ts, None
        
    # data_matrix is (num_stations, num_variables)
    grids_dict = interpolate_gaussian_multi(lon, lat, data_matrix)
    
    # Map indices back to variable names
    results = {variables[i]: grids_dict[i] for i in range(len(variables))}
        
    return ts, results

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def generate_dsm(start_date, end_date, output_dir):
    """
    Generate DSM for Rajasthan (Bhadla Focus) and save as upscaled_merra2_{start_date}.nc
    to match pipeline expectations.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Starting DSM Generation Pipeline")
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"📂 Output Dir: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. Scrape RJ Only
    links = get_state_station_links()
    if not links:
        print("❌ No stations found!")
        return {"status": "error", "message": "No stations found"}
    
    print(f"\n🌐 Step 2: Scraping {len(links)} stations...")
    raw_rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = [exe.submit(scrape_worker, l) for l in links]
        completed = 0
        for f in as_completed(futures):
            result = f.result()
            raw_rows.extend(result)
            completed += 1
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{len(links)} stations scraped")
    
    if not raw_rows:
        print("❌ No data found!")
        sys.exit(1)
    
    print(f"✅ Collected {len(raw_rows)} data points from {completed} stations")
    
    # 2. 15-Min Expansion (UTC based)
    df_96 = expand_temporal(pd.DataFrame(raw_rows), start_date, end_date)
    
    # 3. Grid Setup (1km)
    print("\n🗺️  Step 4: Setting up 1km grid...")
    glat = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
    glon = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)
    gx, gy = np.meshgrid(glon, glat)
    print(f"   Grid size: {len(glat)} x {len(glon)} points")
    
    # 4. Interpolate Blocks
    valid_ts = sorted(df_96['Timestamp_UTC'].unique())
    variables = ['T2M', 'WS10', 'U10', 'V10', 'TCC']
    
    print(f"\n🌀 Step 5: Generating 1km Kriging grids ({len(valid_ts)} time steps)...")
    
    # Prepare tasks for parallel execution
    tasks = []
    
    # Use ProcessPoolExecutor for CPU-bound Kriging
    # RAM OPTIMIZATION: Reduced workers to 4 to prevent memory spikes on prediction.
    max_kriging_workers = min(4, os.cpu_count() or 2)
    from concurrent.futures import ProcessPoolExecutor
    
    # Pre-group data to avoid passing whole DF
    grouped = df_96.groupby('Timestamp_UTC')
    
    for ts in valid_ts:
        if ts not in grouped.groups: continue
        group = grouped.get_group(ts).dropna(subset=['Lat', 'Lon', 'T2M'])
        if len(group) < 5: continue
        
        # Extract numpy arrays to pass to worker
        lat_arr = group['Lat'].values
        lon_arr = group['Lon'].values
        data_matrix = group[variables].values
        
        # No longer passing gx, gy to tasks (passed via initializer)
        tasks.append((ts, lat_arr, lon_arr, data_matrix, variables))

    print(f"   Launching {len(tasks)} tasks using {max_kriging_workers} workers...")
    
    processed = 0
    results_map = {}
    
    # Init pool with initializer to share grid coordinates
    with ProcessPoolExecutor(
        max_workers=max_kriging_workers,
        initializer=init_kriging_worker,
        initargs=(gx, gy)
    ) as executor:
        futures = {executor.submit(process_timestep, *t): t[0] for t in tasks}
        
        for future in as_completed(futures):
            try:
                ts, res = future.result()
                if res:
                    results_map[ts] = res
            except Exception as e:
                print(f"[ERROR] Timestep failed: {e}")
            
            processed += 1
            if processed % 10 == 0:
                print(f"   Progress: {processed}/{len(tasks)} time steps interpolated", flush=True)

    # Reassemble results using pre-allocated numpy arrays to save memory spike
    actual_ts_list = [ts for ts in valid_ts if ts in results_map]
    num_steps = len(actual_ts_list)
    
    print(f"✅ Interpolation complete: {num_steps} time steps. Finalizing Dataset...")
    
    # Pre-allocate variable storage
    storage = {
        v: np.zeros((num_steps, len(glat), len(glon)), dtype=np.float32) 
        for v in variables
    }
    
    for idx, ts in enumerate(actual_ts_list):
        step_results = results_map.pop(ts) # pop to clear memory as we go
        for v in variables:
            storage[v][idx] = step_results[v]
        del step_results # explicitly trigger cleanup

    # 5. Export
    print("\n💾 Step 6: Saving NetCDF...")
    
    # Match pipeline filename expectation: upscaled_merra2_{date}.nc
    nc_filename = f"upscaled_merra2_{end_date}.nc"
    csv_filename = f"upscaled_merra2_{end_date}.csv"
    
    nc_path = os.path.join(output_dir, nc_filename)
    csv_path = os.path.join(output_dir, csv_filename)
    
    ds = xr.Dataset(
        {v: (("time", "lat", "lon"), storage[v]) for v in variables},
        coords={"time": actual_ts_list, "lat": glat, "lon": glon}
    )
    
    ds.attrs['description'] = "Rajasthan DSM 1km 15-min Grid (UTC)"
    ds.attrs['variables'] = "T2M (K), WS10 (km/h), U10 (km/h), V10 (km/h), TCC (Oktas)"
    ds.attrs['timezone'] = "UTC"
    ds.attrs['created'] = datetime.now().isoformat()
    
    ds.to_netcdf(nc_path)
    df_96.to_csv(csv_path, index=False)
    
    print(f"✅ NetCDF saved: {nc_path}")
    print(f"✅ CSV saved: {csv_path}")
    
    # Explicitly clear large objects
    del storage
    del ds
    
    print(f"\n{'='*60}")
    print(f"✨ DSM Generation Complete!")
    print(f"{'='*60}\n")
    
    return {
        "status": "success", 
        "netcdf_file": nc_path,
        "csv_file": csv_path
    }

# ============================================================================
# RUN THE PIPELINE
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    generate_dsm(args.start_date, args.end_date, args.output_dir)