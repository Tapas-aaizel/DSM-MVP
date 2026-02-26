import os
import json
import yaml

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import RegressionPreset, DataDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
)
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
    TestMeanInNSigmas,
    TestNumberOfOutRangeValues,
)

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(os.path.dirname(BASE_DIR), "training_data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MLFLOW_TRACKING_URI   = "http://localhost:5000"
MLFLOW_EXPERIMENT     = "Solar_MVP_Training"
REGISTERED_MODEL_NAME = "Bhadla_Solar_Residual"
CAPACITY_MW           = 50.0

FEATURE_COLS = [
    "ghi_forecast", "t2m_forecast", "wind_speed_forecast", "tcc_forecast",
    "solar_zenith", "solar_azimuth", "hour_of_day", "day_of_year",
    "pvlib_predicted_mw",
]

DATA_PATH = DATA_DIR


# ----------------------------------------------------------
# 1. DATASET VERSION TRACKING
# ----------------------------------------------------------
def log_dataset_version(data_dir: str) -> None:
    """
    Read the DVC sidecar file for the raw NASA CSV and tag the active
    MLflow run with its MD5 hash.  This makes every run traceable back
    to an exact, content-addressed snapshot of the input data.
    """
    dvc_file = os.path.join(
        data_dir,
        "POWER_Point_Hourly_20150101_20260214_027d53N_071d92E_LST.csv.dvc",
    )
    if not os.path.exists(dvc_file):
        print("WARNING: DVC sidecar not found — dataset version not recorded.")
        mlflow.set_tag("data_version_dvc", "unknown")
        return

    with open(dvc_file, "r") as fh:
        dvc_meta = yaml.safe_load(fh)

    data_hash = dvc_meta["outs"][0]["md5"]
    mlflow.set_tag("data_version_dvc", data_hash)
    print(f"Dataset version recorded: {data_hash}")


# ----------------------------------------------------------
# 2. REPRODUCIBILITY LOGS
# ----------------------------------------------------------
def log_reproducibility_metadata(params: dict, features: list,
                                  train_rows: int, test_rows: int) -> None:
    """
    Persist all information needed to exactly reproduce a training run:
    hyperparameters, feature list, split sizes, library versions, and
    the random seed.
    """
    mlflow.log_params(params)
    mlflow.log_param("random_seed",    42)
    mlflow.log_param("train_rows",     train_rows)
    mlflow.log_param("test_rows",      test_rows)
    mlflow.log_param("feature_count",  len(features))
    mlflow.log_param("capacity_mw",    CAPACITY_MW)

    env = {
        "xgboost":  xgb.__version__,
        "pandas":   pd.__version__,
        "numpy":    np.__version__,
        "mlflow":   mlflow.__version__,
        "features": features,
    }
    env_path = os.path.join(REPORTS_DIR, "reproducibility_env.json")
    with open(env_path, "w") as fh:
        json.dump(env, fh, indent=2)
    mlflow.log_artifact(env_path, "reproducibility")
    print("Reproducibility metadata logged.")


# ----------------------------------------------------------
# 3. BIAS EVALUATION REPORT
# ----------------------------------------------------------
def run_bias_evaluation(train_df: pd.DataFrame,
                         test_df: pd.DataFrame) -> str:
    """
    Generate a regression bias report using Evidently.
    The report compares the XGBoost-corrected forecast against actuals
    and is saved as HTML before any model is registered.
    """
    ref = pd.DataFrame({
        "target":     train_df["actual_mw"].values,
        "prediction": train_df["pvlib_predicted_mw"].values,
    })
    cur = pd.DataFrame({
        "target":     test_df["actual_mw"].values,
        "prediction": test_df["corrected_mw"].values,
    })

    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=ref, current_data=cur)

    path = os.path.join(REPORTS_DIR, "bias_evaluation.html")
    report.save_html(path)
    mlflow.log_artifact(path, "reports")
    print(f"Bias evaluation report saved: {path}")
    return path


# ----------------------------------------------------------
# 4. MODEL DRIFT MONITORING
# ----------------------------------------------------------
def run_drift_monitoring(train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          features: list) -> str:
    """
    Compare feature distributions between training and test periods.
    Saves a drift HTML report and runs a test suite that will flag if
    more than 40% of features have drifted or if key weather features
    fall outside expected ranges.
    """
    ref_feat = train_df[features].copy()
    cur_feat = test_df[features].copy()

    # Drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        *[ColumnDriftMetric(column_name=f) for f in features],
    ])
    drift_report.run(reference_data=ref_feat, current_data=cur_feat)
    report_path = os.path.join(REPORTS_DIR, "drift_monitoring.html")
    drift_report.save_html(report_path)
    mlflow.log_artifact(report_path, "reports")

    # Automated drift test suite
    test_suite = TestSuite(tests=[
        TestNumberOfDriftedColumns(lt=4),
        TestShareOfDriftedColumns(lt=0.4),
        *[TestColumnDrift(column_name=f) for f in features[:5]],
        TestMeanInNSigmas(column_name="ghi_forecast",  n_sigmas=2),
        TestMeanInNSigmas(column_name="t2m_forecast",  n_sigmas=2),
        TestNumberOfOutRangeValues(column_name="ghi_forecast", left=0, right=1200),
    ])
    test_suite.run(reference_data=ref_feat, current_data=cur_feat)
    suite_path = os.path.join(REPORTS_DIR, "drift_test_suite.html")
    test_suite.save_html(suite_path)
    mlflow.log_artifact(suite_path, "reports")

    print(f"Drift monitoring report saved: {report_path}")
    return report_path


# ----------------------------------------------------------
# 5. MODEL VERSION TAGGING
# ----------------------------------------------------------
def register_and_tag_model(model: xgb.XGBRegressor,
                            metrics: dict,
                            best_iteration: int) -> None:
    """
    Log the trained model to MLflow Model Registry with a semantic
    version tag and key performance metrics embedded in the tags so
    that every registered version is self-describing.
    """
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    mlflow.set_tag("model_version",        "2.0.0")
    mlflow.set_tag("model_type",           "xgb_residual_correction")
    mlflow.set_tag("site",                 "Bhadla")
    mlflow.set_tag("capacity_mw",          str(CAPACITY_MW))
    mlflow.set_tag("best_iteration",       str(best_iteration))
    mlflow.set_tag("release_nMAE_pct",     str(metrics.get("nMAE_pct")))
    mlflow.set_tag("release_R2",           str(metrics.get("R2")))

    # Persist a local copy for offline serving
    local_path = os.path.join(MODELS_DIR, "xgb_residual_model.json")
    model.save_model(local_path)
    mlflow.log_artifact(local_path, "model_artifacts")
    print(f"Model registered as '{REGISTERED_MODEL_NAME}' and tagged.")


# ----------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------
def run_governed_training():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Disable autolog so every logged artefact is intentional
    mlflow.xgboost.autolog(disable=True)

    with mlflow.start_run(run_name="XGBoost_Bhadla_Solar"):

        # A. Load data
        phys = pd.read_csv(os.path.join(DATA_PATH, "physics_baseline.csv"))
        act  = pd.read_csv(os.path.join(DATA_PATH, "actual_mw.csv"))

        df = pd.merge(phys, act, on="timestamp_ist")
        df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])

        # B. Feature engineering
        df["hour_of_day"] = df["timestamp_ist"].dt.hour
        df["day_of_year"] = df["timestamp_ist"].dt.dayofyear
        df["residual"]    = df["actual_mw"] - df["pvlib_predicted_mw"]

        # C. Train / test split
        train_full    = df[df["timestamp_ist"].dt.year < 2025].copy()
        test_full     = df[df["timestamp_ist"].dt.year == 2025].copy()
        train_daytime = train_full[train_full["solar_zenith"] <= 90].copy()

        # D. Hyperparameters
        params = {
            "n_estimators"    : 1200,
            "learning_rate"   : 0.015,
            "max_depth"       : 8,
            "min_child_weight": 5,
            "subsample"       : 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha"       : 0.1,
            "reg_lambda"      : 1.0,
            "objective"       : "reg:squarederror",
            "tree_method"     : "hist",
        }

        # ----- GOVERNANCE STEP 1: Dataset version tracking -----
        log_dataset_version(DATA_DIR)

        # ----- GOVERNANCE STEP 2 (part 1): Reproducibility logs -----
        log_reproducibility_metadata(
            params,
            FEATURE_COLS,
            train_rows=len(train_daytime),
            test_rows=len(test_full),
        )

        # E. Training
        print(f"Training on {len(train_daytime)} daytime rows...")
        val_split = int(len(train_daytime) * 0.9)
        X_tr  = train_daytime[FEATURE_COLS].iloc[:val_split]
        y_tr  = train_daytime["residual"].iloc[:val_split]
        X_val = train_daytime[FEATURE_COLS].iloc[val_split:]
        y_val = train_daytime["residual"].iloc[val_split:]

        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric="mae",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
        best_iter = model.best_iteration

        # F. Inference
        test_full["residual_predicted"] = model.predict(test_full[FEATURE_COLS])
        test_full["corrected_mw"]       = (
            test_full["pvlib_predicted_mw"] + test_full["residual_predicted"]
        )
        test_full.loc[test_full["solar_zenith"] > 90, "corrected_mw"] = 0.0
        test_full["corrected_mw"] = test_full["corrected_mw"].clip(0, CAPACITY_MW)

        # G. Core metrics
        mae  = mean_absolute_error(test_full["actual_mw"], test_full["corrected_mw"])
        rmse = np.sqrt(mean_squared_error(test_full["actual_mw"], test_full["corrected_mw"]))
        r2   = r2_score(test_full["actual_mw"], test_full["corrected_mw"])
        metrics = {
            "nMAE_pct": round((mae / CAPACITY_MW) * 100, 4),
            "nRMSE_pct": round((rmse / CAPACITY_MW) * 100, 4),
            "R2": round(r2, 4),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # ----- GOVERNANCE STEP 2 (part 2): log best iteration -----
        mlflow.log_param("best_iteration", best_iter)

        # ----- GOVERNANCE STEP 3: Bias evaluation (required before release) -----
        run_bias_evaluation(train_full, test_full)

        # ----- GOVERNANCE STEP 4: Drift monitoring -----
        run_drift_monitoring(train_full, test_full, FEATURE_COLS)

        # ----- GOVERNANCE STEP 5: Model version tagging & registration -----
        register_and_tag_model(model, metrics, best_iter)

        print("\nGOVERNED TRAINING COMPLETE")
        print(f"  Run ID   : {mlflow.active_run().info.run_id}")
        print(f"  nMAE     : {metrics['nMAE_pct']:.2f}%")
        print(f"  R2       : {metrics['R2']:.4f}")
        print(f"  Reports  : {REPORTS_DIR}")

    return model, metrics


if __name__ == "__main__":
    run_governed_training()