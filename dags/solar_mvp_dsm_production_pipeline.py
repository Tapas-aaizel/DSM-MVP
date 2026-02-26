"""
Solar MVP-DSM: Production Pipeline (v2 — Unified)
Merges the robust S1 data ingestion from the Future Pipeline
with the full S2-S4 inference + DSM stages.

Stages:
- S1: DATA INGESTION (NASA/AWS) — with validation, renaming, error handling
- S2: PHYSICS FOUNDATION (PVLib/Downscaling)
- S3: RESIDUAL LEARNING (XGBoost)
- S4: DSM SETTLEMENT (Financials + Reports)
"""
from __future__ import annotations
import os
import sys
import glob
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from airflow.decorators import dag, task, task_group
from airflow.utils.dates import days_ago

# --- PROJECT IMPORTS ---
sys.path.insert(0, str(Path("/opt/airflow")))
from src.physics_baseline import run_pvlib_baseline
from src.ml_correction import predict_with_xgboost
from src.dsm_settlement import calculate_dsm_penalties
from src.dsm_settlement import generate_reports

# --- CONFIGURATION ---
DATA_BASE = "/opt/airflow/data"


def run_command_with_logging(cmd, logger):
    """Runs a command and streams output to the logger."""
    logger.info(f"Executing: {' '.join(cmd)}")
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    ) as proc:
        for line in proc.stdout:
            logger.info(f"[SUBPROCESS] {line.strip()}")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


@dag(
    dag_id="solar_mvp_dsm_production_pipeline",
    start_date=days_ago(1),
    schedule=None,
    catchup=False,
    render_template_as_native_obj=True,
    tags=["production", "solar", "mvp", "dsm"],
    max_active_runs=1,
    params={
        "target_date": "2026-02-10",
        "num_days": 1,
        "check_only": False,
        "run_inference": True,
        "model_uri": "models:/Bhadla_Solar_Residual_v2/Production"
    }
)
def solar_mvp_dsm_production_pipeline():

    # =========================================================================
    #  STAGE 1: DATA INGESTION
    # (Robust: validation, renaming, per-day error handling from Future DAG)
    # =========================================================================
    @task_group(group_id="S1_Data_Ingestion")
    def stage_s1():

        @task(task_id="check_nasa_availability")
        def check_avail(**context):
            logger = logging.getLogger("airflow.task")
            target_date_param = str(context["params"].get("target_date")).strip()[:10]
            target_root_date = datetime.strptime(target_date_param, "%Y-%m-%d")
            num_days = int(context["params"].get("num_days", 1))

            sys.path.insert(0, '/opt/airflow/extractors_new')
            from parallel_downloader import fetch_directory_listing_with_fallback

            active_days, missing_days = [], []

            for i in range(num_days):
                check_date = target_root_date + timedelta(days=i)
                try:
                    files, url = fetch_directory_listing_with_fallback(
                        check_date.year, check_date.month, check_date.day
                    )
                    if files:
                        logger.info(f"✅ {check_date.strftime('%Y-%m-%d')}: {len(files)} files from {url}")
                        active_days.append(check_date.strftime('%Y-%m-%d'))
                    else:
                        logger.warning(f"❌ No data for {check_date.strftime('%Y-%m-%d')}")
                        missing_days.append(check_date.strftime('%Y-%m-%d'))
                except Exception as e:
                    logger.error(f"❌ Check failed for {check_date.strftime('%Y-%m-%d')}: {e}")
                    missing_days.append(check_date.strftime('%Y-%m-%d'))

            logger.info(f"Summary: {len(active_days)} available, {len(missing_days)} missing")
            return {"active_days": active_days, "missing": len(missing_days)}

        @task(task_id="download_nwp_forecast")
        def download_nwp(**context):
            logger = logging.getLogger("airflow.task")
            if context["params"].get("check_only"):
                logger.info("check_only=True, skipping download")
                return []

            target_date_param = str(context["params"].get("target_date")).strip()[:10]
            target_root_date = datetime.strptime(target_date_param, "%Y-%m-%d")
            num_days = int(context["params"].get("num_days", 1))

            sys.path.insert(0, '/opt/airflow/extractors_new')
            from download_rad_24h_task import download_rad_24h_logic

            EXPECTED_RAD_FILES = 25  # 25 hourly .nc files per day

            downloaded_dirs = []

            for i in range(num_days):
                target_date = target_root_date + timedelta(days=i)
                target_folder = target_date.strftime('%d-%m-%Y')
                start_utc = target_date - timedelta(hours=5, minutes=30)

                out_dir = f"{DATA_BASE}/S1_Ingestion/rad_files/{target_folder}"

                # Skip if already downloaded
                existing = glob.glob(os.path.join(out_dir, "*.nc4")) if os.path.isdir(out_dir) else []
                if len(existing) >= EXPECTED_RAD_FILES:
                    logger.info(f"⏭️ SKIP Day {i+1}/{num_days}: {target_folder} already has {len(existing)} files")
                    downloaded_dirs.append(out_dir)
                    continue

                logger.info(f"⬇️ Downloading Day {i+1}/{num_days}: {target_date.strftime('%Y-%m-%d')} ({len(existing)}/{EXPECTED_RAD_FILES} files exist)")
                try:
                    result = download_rad_24h_logic(
                        execution_input=start_utc,
                        output_base_dir=f"{DATA_BASE}/S1_Ingestion",
                        custom_date_folder=target_folder
                    )
                    out_dir = result['output_dir']

                    # Validate: MUST have exactly 25 .nc files to proceed
                    nc_files = glob.glob(os.path.join(out_dir, "*.nc*"))
                    if len(nc_files) >= EXPECTED_RAD_FILES:
                        logger.info(f"✅ Downloaded all {len(nc_files)} files → {out_dir}")
                        downloaded_dirs.append(out_dir)
                    else:
                        logger.error(f"❌ Incomplete download: {len(nc_files)}/{EXPECTED_RAD_FILES} files in {out_dir}")
                        raise RuntimeError(f"Download incomplete for {target_folder}")
                except Exception as e:
                    logger.error(f"❌ Download failed for {target_date.strftime('%Y-%m-%d')}: {e}")
                    raise

            return downloaded_dirs

        @task(task_id="fetch_aws_ground_truth")
        def fetch_aws(**context):
            logger = logging.getLogger("airflow.task")
            if context["params"].get("check_only"):
                logger.info("check_only=True, skipping AWS fetch")
                return []

            target_date_param = str(context["params"].get("target_date")).strip()[:10]
            target_root_date = datetime.strptime(target_date_param, "%Y-%m-%d")
            num_days = int(context["params"].get("num_days", 1))

            aws_files = []

            for i in range(num_days):
                target_date = target_root_date + timedelta(days=i)
                target_str = target_date.strftime('%Y-%m-%d')
                target_folder = target_date.strftime('%d-%m-%Y')
                prev_str = (target_date - timedelta(days=1)).strftime('%Y-%m-%d')

                output_dir = f"{DATA_BASE}/S1_Ingestion/AWS_data/{target_folder}"
                expected_file = f"{output_dir}/upscaled_merra2_{target_str}.nc"

                # Skip if already fetched
                if os.path.exists(expected_file):
                    logger.info(f"⏭️ SKIP AWS for {target_str}: {os.path.basename(expected_file)} already exists")
                    aws_files.append(expected_file)
                    continue

                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    "python", "-u", "/opt/airflow/AWS-Data/weather.py",
                    "--start_date", prev_str,
                    "--end_date", target_str,
                    "--slots", "1,2,3,4,5,6,7,8",
                    "--output_dir", output_dir
                ]

                logger.info(f"⬇️ Fetching AWS for {target_str} (window: {prev_str} to {target_str})")
                try:
                    run_command_with_logging(cmd, logger)

                    # Handle file renaming (weather.py names file after start_date)
                    generated_file = f"{output_dir}/upscaled_merra2_{prev_str}.nc"

                    if os.path.exists(generated_file) and not os.path.exists(expected_file):
                        logger.info(f"Renaming {os.path.basename(generated_file)} → {os.path.basename(expected_file)}")
                        os.rename(generated_file, expected_file)
                        aws_files.append(expected_file)
                    elif os.path.exists(expected_file):
                        aws_files.append(expected_file)
                    else:
                        logger.warning(f"❌ AWS file not found: {generated_file}")
                except Exception as e:
                    logger.error(f"❌ AWS fetch failed for {target_str}: {e}")

            return aws_files

        avail = check_avail()
        nwp = download_nwp()
        aws = fetch_aws()
        avail >> [nwp, aws]
        return nwp, aws

    # =========================================================================
    #  STAGE 2: PHYSICS FOUNDATION
    # (Downscale → Interpolate → Merge → PVLib)
    # =========================================================================
    @task_group(group_id="S2_Physics_Foundation")
    def stage_s2(rad_dirs, aws_files):

        @task(task_id="spatial_interpolation")
        def downscale(input_dirs):
            logger = logging.getLogger("airflow.task")
            if not input_dirs:
                return []

            downscaled = []
            for input_dir in input_dirs:
                parent_name = os.path.basename(input_dir)
                output_dir = f"{DATA_BASE}/S2_Physics/rad_downscaled/{parent_name}"
                os.makedirs(output_dir, exist_ok=True)

                cmd = [
                    "python", "-u", "/opt/airflow/downscaling/downscale_radiation.py",
                    "--input_dir", input_dir, "--output_dir", output_dir
                ]
                try:
                    run_command_with_logging(cmd, logger)
                    out_files = glob.glob(f"{output_dir}/*.nc")
                    if out_files:
                        logger.info(f"✅ Downscaled {parent_name}: {len(out_files)} files")
                        downscaled.append(output_dir)
                    else:
                        logger.error(f"❌ No output from downscaling {parent_name}")
                except Exception as e:
                    logger.error(f"❌ Downscale error for {parent_name}: {e}")

            return downscaled

        @task(task_id="temporal_interpolation")
        def interpolate(input_dirs, **context):
            logger = logging.getLogger("airflow.task")
            if not input_dirs:
                return []

            interpolated = []
            for input_dir in input_dirs:
                parent_name = os.path.basename(input_dir)
                output_dir = f"{DATA_BASE}/S2_Physics/rad_15min/{parent_name}"
                os.makedirs(output_dir, exist_ok=True)

                # Parse date from folder name (DD-MM-YYYY)
                try:
                    dt = datetime.strptime(parent_name, "%d-%m-%Y")
                    date_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    date_str = str(context["params"].get("target_date")).strip()[:10]

                cmd = [
                    "python", "-u", "/opt/airflow/temporal_interpolation/interpolate_rad_1h_to_15min.py",
                    "--input_dir", input_dir, "--output_dir", output_dir, "--date", date_str
                ]
                try:
                    run_command_with_logging(cmd, logger)
                    out_files = glob.glob(f"{output_dir}/*.nc")
                    if out_files:
                        logger.info(f"✅ Interpolated {parent_name}: {len(out_files)} files")
                        interpolated.append(output_dir)
                    else:
                        logger.error(f"❌ No output from interpolation {parent_name}")
                except Exception as e:
                    logger.error(f"❌ Interpolation error for {parent_name}: {e}")

            return interpolated

        @task(task_id="merge_Geos-fp_and_AWS_data")
        def merge(rad_dirs, aws_files):
            logger = logging.getLogger("airflow.task")
            if not rad_dirs or not aws_files:
                return []

            merged_files = []
            for rad_dir in rad_dirs:
                folder_date = os.path.basename(rad_dir)

                # Find matching AWS file
                matching_aws = None
                try:
                    dt = datetime.strptime(folder_date, "%d-%m-%Y")
                    ymd = dt.strftime("%Y-%m-%d")
                    for af in aws_files:
                        if folder_date in af or ymd in os.path.basename(af):
                            matching_aws = af
                            break
                except ValueError:
                    pass

                if not matching_aws:
                    logger.warning(f"⚠️ No matching AWS file for {folder_date}, skipping")
                    continue

                output_nc = f"{DATA_BASE}/S2_Physics/merged/{folder_date}/final_merged_solar_mvp.nc"
                output_csv = f"{DATA_BASE}/S2_Physics/merged/{folder_date}/final_merged_solar_mvp.csv"
                os.makedirs(os.path.dirname(output_nc), exist_ok=True)

                # merge script expects --aws_dir (directory), not a file path
                aws_dir = os.path.dirname(matching_aws)

                cmd = [
                    "python", "-u", "/opt/airflow/utils/merge_final_outputs.py",
                    "--rad_dir", rad_dir,
                    "--aws_dir", aws_dir,
                    "--output_file", output_nc
                ]

                try:
                    run_command_with_logging(cmd, logger)
                    # Use CSV for downstream tasks to avoid HDF5/NetCDF errors
                    if os.path.exists(output_csv):
                        logger.info(f"✅ Merged {folder_date} (CSV ready)")
                        merged_files.append(output_csv)
                    elif os.path.exists(output_nc):
                        logger.warning(f"⚠️ CSV missing, falling back to NC for {folder_date}")
                        merged_files.append(output_nc)
                    else:
                        logger.error(f"❌ Merge output missing for {folder_date}")
                except Exception as e:
                    logger.error(f"❌ Merge error for {folder_date}: {e}")

            return merged_files

        @task(task_id="run_pvlib_faiman")
        def run_pvlib(merged_files):
            logger = logging.getLogger("airflow.task")
            if not merged_files:
                return []

            baseline_files = []
            for merged_f in merged_files:
                folder_date = os.path.basename(os.path.dirname(merged_f))
                output_dir = f"{DATA_BASE}/S2_Physics/baseline/{folder_date}"
                os.makedirs(output_dir, exist_ok=True)
                
                logger.info(f"Running PVLib for: {merged_f} -> {output_dir}")
                try:
                    result = run_pvlib_baseline(merged_f, output_dir=output_dir)
                    logger.info(f"✅ PVLib complete: {result}")
                    baseline_files.append(result)
                except Exception as e:
                    logger.error(f"❌ PVLib error for {merged_f}: {e}")

            return baseline_files

        @task(task_id="validate_pvlib_output")
        def validate_pvlib(baseline_files):
            """Verify physics_baseline.csv has correct schema and sane values."""
            import pandas as pd
            logger = logging.getLogger("airflow.task")

            if not baseline_files:
                raise RuntimeError("❌ No PVLib output files to validate")

            REQUIRED_COLS = [
                'timestamp_ist', 'block_number',
                'ghi_forecast', 't2m_forecast', 'wind_speed_forecast', 'tcc_forecast',
                'solar_zenith', 'solar_azimuth',
                'hour_of_day', 'day_of_year',
                'pvlib_predicted_mw'
            ]

            validated = []
            for path in baseline_files:
                logger.info(f"🔍 Validating: {path}")

                # 1. File exists
                if not os.path.exists(path):
                    raise RuntimeError(f"❌ File missing: {path}")

                df = pd.read_csv(path)

                # 2. Required columns
                missing = [c for c in REQUIRED_COLS if c not in df.columns]
                if missing:
                    raise RuntimeError(f"❌ Missing columns in {path}: {missing}")

                # 3. Row count (expect ~96 blocks)
                if len(df) < 90:
                    raise RuntimeError(f"❌ Only {len(df)} rows (expected ~96)")

                # 4. PVLib MW range [0, capacity]
                if df['pvlib_predicted_mw'].min() < 0:
                    raise RuntimeError(f"❌ Negative PVLib MW found")
                if df['pvlib_predicted_mw'].max() > 55:  # allow small overshoot
                    raise RuntimeError(f"❌ PVLib MW exceeds capacity: {df['pvlib_predicted_mw'].max():.1f}")

                # 5. Night blocks should be 0
                night = df[df['solar_zenith'] > 90]
                if len(night) > 0 and night['pvlib_predicted_mw'].max() > 0:
                    raise RuntimeError(f"❌ Night blocks have non-zero PVLib MW")

                # 6. Peak should be reasonable
                peak = df['pvlib_predicted_mw'].max()
                if peak < 5:
                    raise RuntimeError(f"❌ Peak too low ({peak:.1f} MW) — possible model error")

                daytime = (df['pvlib_predicted_mw'] > 0).sum()
                energy = (df['pvlib_predicted_mw'] * 0.25).sum()
                logger.info(f"✅ PASSED: {len(df)} rows, {daytime} daytime blocks, "
                           f"peak={peak:.1f} MW, energy={energy:.1f} MWh")
                validated.append(path)

            logger.info(f"✅ All {len(validated)} files validated successfully")
            return validated

        downscaled = downscale(rad_dirs)
        interpolated = interpolate(downscaled)
        merged = merge(interpolated, aws_files)
        baselines = run_pvlib(merged)
        validated = validate_pvlib(baselines)
        return validated

    # =========================================================================
    # 🧠 STAGE 3: RESIDUAL LEARNING (XGBoost)
    # =========================================================================
    @task_group(group_id="S3_Residual_Learning")
    def stage_s3(baseline_paths):

        @task(task_id="xgboost_residual_correction")
        def run_ml(paths, **context):
            logger = logging.getLogger("airflow.task")
            
            # Use model_uri from params (priority) or fall back to default logic
            model_uri = context["params"].get("model_uri")
            
            if not paths:
                return []

            corrected_files = []
            for path in paths:
                folder_date = os.path.basename(os.path.dirname(path))
                output_dir = f"{DATA_BASE}/S3_ML/forecast/{folder_date}"
                os.makedirs(output_dir, exist_ok=True)

                logger.info(f"Running XGBoost correction for: {path} -> {output_dir}")
                try:
                    # Pass the model_uri to the prediction function
                    result = predict_with_xgboost(path, model_path=model_uri, output_dir=output_dir)
                    logger.info(f"✅ ML correction complete: {result}")
                    corrected_files.append(result)
                except Exception as e:
                    logger.error(f"❌ ML error for {path}: {e}")

            return corrected_files

        return run_ml(baseline_paths)

    # =========================================================================
    # STAGE 4: DSM SETTLEMENT + REPORTING
    # =========================================================================
    @task_group(group_id="S4_DSM_Settlement")
    def stage_s4(prediction_paths):

        @task(task_id="calculate_deviations_and_penalties")
        def calc_dsm(paths, **context):
            logger = logging.getLogger("airflow.task")
            if not paths:
                return []

            if not paths:
                return []

            dsm_files = []
            for path in paths:
                folder_date = os.path.basename(os.path.dirname(path))
                output_dir = f"/opt/airflow/results/DSM/{folder_date}"
                os.makedirs(output_dir, exist_ok=True)

                logger.info(f"Calculating DSM penalties for: {path} -> {output_dir}")
                try:
                    result = calculate_dsm_penalties(path, output_dir=output_dir)
                    logger.info(f"✅ DSM calculation complete: {result}")
                    dsm_files.append(result)
                except Exception as e:
                    logger.error(f"❌ DSM error for {path}: {e}")

            return dsm_files

        @task(task_id="generate_reports_and_charts")
        def run_reports(dsm_paths):
            logger = logging.getLogger("airflow.task")
            if not dsm_paths:
                return {}

            all_reports = {}
            for dsm_path in dsm_paths:
                logger.info(f"Generating reports for: {dsm_path}")
                try:
                    result = generate_reports(dsm_path)
                    logger.info(f"✅ Reports generated")
                    all_reports[dsm_path] = result
                except Exception as e:
                    logger.error(f"❌ Report error for {dsm_path}: {e}")

            return all_reports

        dsm_results = calc_dsm(prediction_paths)
        reports = run_reports(dsm_results)
        return reports

    # =========================================================================
    # --- EXECUTION FLOW ---
    # =========================================================================
    nwp_data, aws_data = stage_s1()
    baselines = stage_s2(nwp_data, aws_data)
    predictions = stage_s3(baselines)
    dsm_results = stage_s4(predictions)


solar_mvp_dsm_production_pipeline()
