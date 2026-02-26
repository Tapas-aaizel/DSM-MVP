"""
Stage S5: Reporting & Metrics
Generates metrics_comparison.json, hindcast_audit.csv, and charts (PNG).
Following ClimateForte Solar DSM MVP Execution Directive v1.0

Deliverables produced:
  - results/metrics_comparison.json  (nMAE, RMSE, Skill Score, Tail Risk)
  - results/hindcast_audit.csv       (forecast vs actual for every block)
  - results/predicted_vs_actual.png  (scatter plot)
  - results/daily_penalty_chart.png  (PVLib vs Model INR per day)
  - results/feature_importance.png   (XGBoost feature weights)
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib.dates as mdates
from src import config
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(df, forecast_col, actual_col='actual_mw', capacity=None):
    """
    Compute nMAE, RMSE, and Tail Risk for a forecast column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain forecast_col and actual_col.
    forecast_col : str
        Column with forecast values.
    actual_col : str
        Column with actual values.
    capacity : float
        Plant capacity for nMAE normalisation.

    Returns
    -------
    dict with nMAE, RMSE, tail_risk_nmae
    """
    if capacity is None:
        capacity = config.CAPACITY_MW

    errors = (df[forecast_col] - df[actual_col]).abs()

    nmae = errors.mean() / capacity * 100  # percentage
    rmse = np.sqrt(((df[forecast_col] - df[actual_col]) ** 2).mean())

    # Tail risk: worst 5% of blocks
    n_tail = max(1, int(0.05 * len(errors)))
    tail_errors = errors.nlargest(n_tail)
    tail_risk_nmae = tail_errors.mean() / capacity * 100

    return {
        'nMAE_pct': float(round(nmae, 4)),
        'RMSE_MW': float(round(rmse, 4)),
        'tail_risk_nmae_pct': float(round(tail_risk_nmae, 4)),
    }


def generate_performance_dashboard(df, output_dir):
    """
    Generates a high-quality performance dashboard with power curves and deviation rates.
    """
    import matplotlib.pyplot as plt

    # Filter for daytime hours only for better visual clarity (Sunrise to Sunset)
    daytime_df = df[df['forecast_mw'] > 0.1].copy()

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.3)

    # --- TOP PLOT: POWER CURVES ---
    ax1.plot(df['timestamp'], df['actual_mw'], label='Actual Generation', color='#2ecc71', linewidth=2, alpha=0.8)
    ax1.plot(df['timestamp'], df['forecast_mw'], label='XGBoost Forecast', color='#3498db', linewidth=2, linestyle='--')
    ax1.fill_between(df['timestamp'], df['actual_mw'], df['forecast_mw'], color='gray', alpha=0.2, label='Forecast Error Delta')

    ax1.set_title('ClimateForte: Solar Power Forecasting Performance (Bhadla, RJ)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power Output (MW)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 55) # Cap at 55MW for a 50MW plant

    # --- BOTTOM PLOT: ERROR PERCENTAGE & DSM THRESHOLD ---
    ax2.plot(daytime_df['timestamp'], daytime_df['deviation_pct'], color='#34495e', label='Actual Deviation %', linewidth=2)

    # Add Multi-Tier Penalty Bands (Directive v1.0 updated)
    ax2.fill_between(daytime_df['timestamp'], 0, 5, color='#2ecc71', alpha=0.15, label='Safe (<5%)')
    ax2.fill_between(daytime_df['timestamp'], 5, 7, color='#f1c40f', alpha=0.2, label='Partial (0.25x)')
    ax2.fill_between(daytime_df['timestamp'], 7, 15, color='#e67e22', alpha=0.25, label='Partial (0.50x)')
    ax2.fill_between(daytime_df['timestamp'], 15, 100, color='#e74c3c', alpha=0.1, label='Full Penalty (>15%)')
    
    # Add the Mandatory CERC 15% Threshold line
    ax2.axhline(y=15, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.set_title('Deviation % vs. Multi-Tier Regulatory Bands', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Deviation Rate (%)', fontsize=12)
    ax2.set_xlabel('Time of Day (IST)', fontsize=12)
    ax2.set_ylim(0, 25) # Show up to 25% for clarity
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Format X-axis to show hours clearly
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Save the output
    perf_path = os.path.join(output_dir, 'performance_dashboard.png')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ performance_dashboard.png saved to {perf_path}")
    return perf_path


def generate_reports(dsm_report_path, forecast_corrected_path=None,
                     model_path=None, output_dir=None):
    """
    Generate all reporting deliverables.

    Parameters
    ----------
    dsm_report_path : str
        Path to dsm_report.csv from Stage S4.
    forecast_corrected_path : str, optional
        Path to forecast_corrected.csv from Stage S3.
        If None, derives from dsm_report_path directory.
    model_path : str, optional
        Path to saved XGBoost model for feature importance.
    output_dir : str, optional
        Directory to save reports. Defaults to config.RESULTS_DIR.

    Returns
    -------
    dict : Paths to all generated files
    """
    print("=" * 80)
    print("STAGE S5: REPORTING & METRICS")
    print("=" * 80)

    if output_dir is None:
        # Default to the same directory as the input report (ensures date isolation)
        output_dir = os.path.dirname(dsm_report_path)
    os.makedirs(output_dir, exist_ok=True)

    # Load DSM report
    df_dsm = pd.read_csv(dsm_report_path)
    df_dsm['timestamp'] = pd.to_datetime(df_dsm['timestamp'])

    generated_files = {}

    # ── 1. METRICS COMPARISON JSON ────────────────────────────────────────
    print("  Computing metrics...")

    # Filter to daytime only (where actual generation exists)
    df_day = df_dsm[df_dsm['actual_mw'] > 0].copy()

    pvlib_metrics = compute_metrics(df_day, 'pvlib_baseline_mw', 'actual_mw')
    model_metrics = compute_metrics(df_day, 'forecast_mw', 'actual_mw')

    # # Skill Score = 1 - (Model_RMSE / PVLib_RMSE)
    # skill_score = 1 - (model_metrics['RMSE_MW'] / pvlib_metrics['RMSE_MW']) \
    #     if pvlib_metrics['RMSE_MW'] > 0 else 0

    # Night block check
    df_night = df_dsm[df_dsm['pvlib_baseline_mw'] == 0]
    night_correct = (df_night['forecast_mw'] == 0).all() if len(df_night) > 0 else True

    metrics = {
        'pvlib_only': {
            'nMAE_pct': pvlib_metrics['nMAE_pct'],
            'RMSE_MW': pvlib_metrics['RMSE_MW'],
            'tail_risk_nmae_pct': pvlib_metrics['tail_risk_nmae_pct'],
        },
        'xgboost_corrected': {
            'nMAE_pct': model_metrics['nMAE_pct'],
            'RMSE_MW': model_metrics['RMSE_MW'],
            'tail_risk_nmae_pct': model_metrics['tail_risk_nmae_pct'],
        },
        #'skill_score': float(round(skill_score, 4)),
        'night_blocks_correct': bool(night_correct),
        'pass_criteria': {
            'nMAE_lt_5pct': bool(model_metrics['nMAE_pct'] < 5.0),
            'RMSE_lt_4MW': bool(model_metrics['RMSE_MW'] < 4.0),
            #'skill_score_gt_0.15': bool(skill_score > 0.15),
            'tail_risk_lt_12pct': bool(model_metrics['tail_risk_nmae_pct'] < 12.0),
            'night_blocks_all_zero': bool(night_correct),
        }
    }

    metrics_path = os.path.join(output_dir, "metrics_comparison.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    generated_files['metrics'] = metrics_path
    print(f"  ✓ metrics_comparison.json")

    # ── 2. HINDCAST AUDIT CSV ─────────────────────────────────────────────
    print("  Generating hindcast audit...")
    audit = pd.DataFrame()
    # Forecast issue time: T0 (one day before target)
    audit['forecast_issue_time'] = df_dsm['timestamp'] - pd.Timedelta(days=1)
    audit['forecast_target_time'] = df_dsm['timestamp']
    audit['forecast_mw'] = df_dsm['forecast_mw']
    audit['actual_mw'] = df_dsm['actual_mw']
    audit['error_mw'] = df_dsm['forecast_mw'] - df_dsm['actual_mw']
    audit['error_pct'] = np.where(
        df_dsm['actual_mw'] > 0,
        (audit['error_mw'].abs() / df_dsm['actual_mw'] * 100).round(2),
        0.0
    )

    audit_path = os.path.join(output_dir, "hindcast_audit.csv")
    audit.to_csv(audit_path, index=False)
    generated_files['hindcast_audit'] = audit_path
    print(f"  ✓ hindcast_audit.csv")

    # ── 3. CHARTS (PNG) ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("  WARNING: matplotlib not installed — skipping chart generation")

    if HAS_MATPLOTLIB:
        # Chart 1: Predicted vs Actual scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df_day['actual_mw'], df_day['forecast_mw'],
                   alpha=0.5, s=15, c='#4fc3f7', label='XGBoost Corrected')
        ax.scatter(df_day['actual_mw'], df_day['pvlib_baseline_mw'],
                   alpha=0.3, s=15, c='#ff8a65', label='PVLib Only')
        max_val = max(df_day['actual_mw'].max(), df_day['forecast_mw'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Forecast')
        ax.set_xlabel('Actual MW', fontsize=12)
        ax.set_ylabel('Predicted MW', fontsize=12)
        ax.set_title('Predicted vs Actual MW (Test Set)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        scatter_path = os.path.join(output_dir, "predicted_vs_actual.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated_files['scatter'] = scatter_path
        print(f"  ✓ predicted_vs_actual.png")

        # Chart 2: Daily Penalty Comparison
        df_dsm['date'] = df_dsm['timestamp'].dt.date
        daily = df_dsm.groupby('date').agg(
            pvlib_penalty=('penalty_inr', lambda x: x.sum() + df_dsm.loc[x.index, 'penalty_saved_inr'].sum()),
            model_penalty=('penalty_inr', 'sum'),
        ).reset_index()
        # pvlib_penalty = model_penalty + saved
        daily['pvlib_penalty'] = daily['model_penalty'] + df_dsm.groupby('date')['penalty_saved_inr'].sum().values

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(daily))
        width = 0.35
        ax.bar([i - width/2 for i in x], daily['pvlib_penalty'],
               width, label='PVLib Only', color='#ff8a65', alpha=0.8)
        ax.bar([i + width/2 for i in x], daily['model_penalty'],
               width, label='XGBoost Corrected', color='#4fc3f7', alpha=0.8)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Penalty (INR)', fontsize=12)
        ax.set_title('Daily Penalty: PVLib vs Our Model (INR)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in daily['date']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        penalty_chart_path = os.path.join(output_dir, "daily_penalty_chart.png")
        fig.savefig(penalty_chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        generated_files['penalty_chart'] = penalty_chart_path
        print(f"  ✓ daily_penalty_chart.png")

        # Chart 3: Feature Importance (if model available)
        if model_path is None:
            model_path = os.path.join(config.MODELS_DIR, "xgboost_residual_v1.json")

        if os.path.exists(model_path):
            try:
                import xgboost as xgb
                from src.ml_correction.xgboost_inference import FEATURE_COLUMNS
                
                reg = xgb.XGBRegressor()
                reg.load_model(model_path)

                importance = reg.feature_importances_
                fi_df = pd.DataFrame({
                    'feature': FEATURE_COLUMNS,
                    'importance': importance
                }).sort_values('importance', ascending=True)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(fi_df['feature'], fi_df['importance'], color='#81c784', alpha=0.8)
                ax.set_xlabel('Feature Importance', fontsize=12)
                ax.set_title('XGBoost Feature Importance', fontsize=14)
                ax.grid(True, alpha=0.3, axis='x')
                fi_path = os.path.join(output_dir, "feature_importance.png")
                fig.savefig(fi_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                generated_files['feature_importance'] = fi_path
                print(f"  ✓ feature_importance.png")
            except ImportError:
                print(f"  ⊘ feature_importance.png (xgboost not installed)")
            except Exception as e:
                print(f"  ⊘ feature_importance.png (ML error: {e})")
        else:
            print(f"  ⊘ feature_importance.png (no model found)")

        # Chart 4: Performance Dashboard (New)
        try:
            perf_path = generate_performance_dashboard(df_dsm, output_dir)
            generated_files['performance_dashboard'] = perf_path
        except Exception as e:
            print(f"  ⊘ performance_dashboard.png (Error: {e})")


    # ── SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n  Generated {len(generated_files)} deliverables in {output_dir}")
    print("=" * 80)

    return generated_files


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_reports(sys.argv[1])
    else:
        test_path = os.path.join(
            config.DATA_DIR,
            "final_output/11-02-2026/dsm_report.csv"
        )
        if os.path.exists(test_path):
            generate_reports(test_path)
        else:
            print(f"Usage: python -m src.dsm_settlement.reporting <dsm_report_path>")
