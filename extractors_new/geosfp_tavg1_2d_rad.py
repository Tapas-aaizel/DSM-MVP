"""GEOS-FP 2D Radiation Variables Extractor (tavg1_2d_rad_Nx)

Product: tavg1_2d_rad_Nx
Variables: LWGAB, LWGEM, LWTUP, SWGNT, SWTNT
Frequency: 1-hourly → select 3-hourly (0030, 0330, 0630, ...)
Purpose: Radiation variables for Prithvi
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def download_geosfp_tavg1_2d_rad(date: str, output_dir: str) -> list:
    """Download GEOS-FP tavg1_2d_rad_Nx data (3-hourly subset).
    
    Args:
        date: Date in YYYY-MM-DD format
        output_dir: Output directory path
        
    Returns:
        List of downloaded file paths
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    downloader_path = Path(__file__).parent / "geosfp_prithvi_downloader.py"
    
    print(f"[INFO] Downloading tavg1_2d_rad_Nx for {date}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(downloader_path), "--date", date, "--out", str(out_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        
        files = list(out_dir.glob("*tavg1_2d_rad_Nx*.nc4"))
        print(f"[OK] Downloaded {len(files)} tavg1_2d_rad_Nx files")
        return [str(f) for f in files]
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed: {e.stderr}")
        return []


if __name__ == "__main__":
    from datetime import timedelta
    test_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    download_geosfp_tavg1_2d_rad(test_date, "./data/test_tavg1_2d_rad")
