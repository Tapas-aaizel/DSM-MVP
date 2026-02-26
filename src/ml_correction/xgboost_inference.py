"""
Stage S3: XGBoost Residual Correction (Inference Mode)
Loads a pre-trained XGBoost model and corrects PVLib physics baseline predictions.
Following ClimateForte Solar DSM MVP Execution Directive v1.0
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import os
from src import config

# ============================================================================
# CONFIGURATION
# ============================================================================
CAPACITY_MW = config.CAPACITY_MW
# Canonical path for the production environment
# MODEL_FILENAME = "xgboost_residual_v1.json"
MODEL_FILENAME = "xgboost_residual_v1_1yr.json"

FEATURE_COLUMNS = [
    'ghi_forecast', 't2m_forecast', 'wind_speed_forecast', 'tcc_forecast',
    'solar_zenith', 'solar_azimuth', 'hour_of_day', 'day_of_year', 'pvlib_predicted_mw'
]

import mlflow
import mlflow.xgboost

def predict_with_xgboost(baseline_path, model_path=None, output_dir=None):
    """
    S3: Apply pre-trained XGBoost model to correct PVLib predictions.
    Includes Task 1.2 mandated synthesis of 'Actuals'.
    Supports loading from MLflow Registry or local file.
    """
    print(f"🚀 Module S3: Applying XGBoost Residual Correction & Synthesizing Actuals...")
    
    # 1. LOAD PHYSICS BASELINE
    if not os.path.exists(baseline_path):
        print(f"❌ ERROR: {baseline_path} not found.")
        return

    df = pd.read_csv(baseline_path)
    df['timestamp_ist'] = pd.to_datetime(df['timestamp_ist'])

    # 2. FEATURE ENGINEERING
    df['hour_of_day'] = df['timestamp_ist'].dt.hour
    df['day_of_year'] = df['timestamp_ist'].dt.dayofyear
    df['pvlib_predicted_mw'] = df['pvlib_predicted_mw'].fillna(0)

    # 3. LOAD TRAINED XGBOOST MODEL (MLflow Registry Priority)
    # Default to 'Production' stage if using MLflow
    # Note: MLFLOW_TRACKING_URI should be set in environment
    
    model = None
    has_model = False
    
    # Priority 1: MLflow URI (e.g. models:/ModelName/Production)
    if model_path and model_path.startswith("models:/"):
        try:
            print(f"🔗 Attempting to load from MLflow Registry: {model_path}")
            model = mlflow.xgboost.load_model(model_path)
            has_model = True
        except Exception as e:
            print(f"⚠️ MLflow Load Failed: {e}")

    # Priority 2: Standard local model path
    if not has_model:
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, MODEL_FILENAME)
            
        model = xgb.XGBRegressor()
        try:
            model.load_model(model_path)
            print(f"🧠 Loaded trained model from Local Path: {model_path}")
            has_model = True
        except Exception as e:
            print(f"⚠️ WARNING: Could not load model at {model_path}. Using zero correction.")
            has_model = False

    # 4. INFERENCE (DAYTIME ONLY)
    # Mandated definition: daytime is zenith <= 90
    daytime_mask = df['solar_zenith'] <= 90
    df['residual_predicted'] = 0.0
    
    if has_model and daytime_mask.any():
        # Verify all features exist
        missing = [f for f in features if f not in df.columns]
        if not missing:
            df.loc[daytime_mask, 'residual_predicted'] = model.predict(df.loc[daytime_mask, features])
        else:
            print(f"❌ ERROR: Missing features {missing}")

    # 5. FINAL CORRECTION
    df['corrected_mw'] = (df['pvlib_predicted_mw'] + df['residual_predicted']).clip(0, CAPACITY_MW)
    # Night Gate: P0 blocker requirement
    df.loc[df['solar_zenith'] > 90, 'corrected_mw'] = 0 

    # 6. TASK 1.2: SYNTHESIZE ACTUAL MW (±5% REALISTIC NOISE)
    # We generate 'Actuals' based on Physics + Random Noise to simulate ground truth
    print("🎲 Synthesizing actual_mw with ±5% realistic noise...")
    np.random.seed(42) # Mandated for reproducible audit results
    
    # Generate noise between 0.95 and 1.05
    noise = np.random.uniform(0.95, 1.05, len(df))
    #noise = np.random.uniform(0.85, 1.15, len(df))
    
    # Apply noise to the physics values to create the 'True' generation
    df['actual_mw'] = (df['pvlib_predicted_mw'] * noise).round(4)
    
    # Ensure night blocks remain 0
    df.loc[df['solar_zenith'] > 90, 'actual_mw'] = 0.0

    # 7. OUTPUT GENERATION
    final_output = df[[
        'timestamp_ist', 
        'block_number', 
        'pvlib_predicted_mw', 
        'residual_predicted', 
        'corrected_mw',
        'actual_mw'
    ]].copy()

    if output_dir is None:
        output_dir = os.path.dirname(baseline_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "forecast_corrected.csv")
    final_output.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: {output_path} generated with Actuals.")
    print("\nPreview of midday correction and synthesized actuals:")
    print(final_output.iloc[45:50])
    
    return output_path

if __name__ == "__main__":
    import sys
    # Usage: script.py <baseline_path> [<model_path>] [<output_dir>]
    if len(sys.argv) > 3:
        predict_with_xgboost(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2:
        predict_with_xgboost(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        predict_with_xgboost(sys.argv[1])
    else:
        print("Usage: python -m src.ml_correction.xgboost_inference <baseline_path> [<model_path>] [<output_dir>]")
