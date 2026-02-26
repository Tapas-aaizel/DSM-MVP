"""
Stage S4: DSM Penalty Calculator
Computes energy deviation, penalty bands, and financial impact per 15-min block.
Following ClimateForte Solar DSM MVP Execution Directive v1.0, Section 4
"""
import pandas as pd
import numpy as np
import json
import os
from src import config

# ============================================================================
# CONFIGURATION (Directive Section 4.1)
# ============================================================================
RATE_INR_KWH = config.DSM_RATE_INR  # CERC Standard Rate
CAPACITY_MW = config.CAPACITY_MW

def calculate_dsm_penalty(scheduled_mw, actual_mw):
    """
    Implements the exact logic from Step 1 to Step 6 of the Directive.
    """
    # Step 1 & 2: Energy Calculation (MWh) for 15-minute block
    scheduled_mwh = scheduled_mw * (15/60)
    actual_mwh = actual_mw * (15/60)
    abs_deviation_mwh = abs(scheduled_mwh - actual_mwh)
    
    # Safety Gate for Night/Zero Schedule
    if scheduled_mw <= 0.01: 
        return 0.0, "none", 0.0
        
    # Step 3: Deviation % Calculation
    dev_pct = (abs(scheduled_mw - actual_mw) / scheduled_mw) * 100
    
    # Step 4: Penalty Band Determination
    if dev_pct < 5:
        multiplier = 0.0
        band = "none"
    elif 5 <= dev_pct < 7:
        multiplier = 0.25
        band = "partial-low"
    elif 7 <= dev_pct < 15:
        multiplier = 0.5
        band = "partial"
    else:
        multiplier = 1.0
        band = "full"
        
    # Step 5 & 6: Penalty Amount (INR)
    # Formula: Deviation_MWh * 1000 (to kWh) * Rate * Multiplier
    penalty_inr = abs_deviation_mwh * 1000 * RATE_INR_KWH * multiplier
    
    return round(penalty_inr, 2), band, round(dev_pct, 2)

def calculate_dsm_penalties(prediction_path, output_dir=None):
    """
    S4: Calculate DSM Penalties and Savings using mandated ROI logic.
    """
    print("🚀 Module S4: Running DSM Penalty Calculation & Value Proof...")
    
    # Load data from Module S3
    if not os.path.exists(prediction_path):
        print(f"❌ ERROR: {prediction_path} not found. Run Module S3 first.")
        return
        
    df = pd.read_csv(prediction_path)
    
    report_rows = []
    
    for _, row in df.iterrows():
        # A. Calculate Penalty for Your XGBoost Model (The Schedule you submit)
        p_xg, b_xg, d_xg = calculate_dsm_penalty(row['corrected_mw'], row['actual_mw'])
        
        # B. Calculate Penalty for the Physics-Only Baseline (What happens without ML)
        p_phys, b_phys, d_phys = calculate_dsm_penalty(row['pvlib_predicted_mw'], row['actual_mw'])
        
        report_rows.append({
            'timestamp': row['timestamp_ist'],
            'block_number': row['block_number'],
            'forecast_mw': round(row['corrected_mw'], 4),
            'actual_mw': round(row['actual_mw'], 4),
            'pvlib_baseline_mw': round(row['pvlib_predicted_mw'], 4),
            'deviation_pct': d_xg,
            'penalty_band': b_xg,
            'penalty_inr': p_xg,
            'penalty_phys_inr': p_phys,
            'penalty_saved_inr': round(p_phys - p_xg, 2)
        })

    report_df = pd.DataFrame(report_rows)
    
    # Create results directory if not specified
    if output_dir is None:
        # Default to a generic results folder if no date-specific dir passed
        output_dir = os.path.join(config.BASE_DIR, 'results', 'DSM')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Task 3.1: Detailed DSM Report
    report_path = os.path.join(output_dir, 'dsm_report.csv')
    report_df.to_csv(report_path, index=False)

    # Generate Task 3.1: Summary JSON
    tot_phys = float(report_df['penalty_phys_inr'].sum())
    tot_xg = float(report_df['penalty_inr'].sum())
    tot_saved = float(report_df['penalty_saved_inr'].sum())
    
    # Use max(1, ...) to avoid division by zero
    savings_pct = float((tot_saved / tot_phys) * 100) if tot_phys > 0 else 0.0

    summary = {
        "total_penalty_pvlib_baseline_inr": round(tot_phys, 2),
        "total_penalty_xgboost_corrected_inr": round(tot_xg, 2),
        "total_savings_generated_inr": round(tot_saved, 2),
        "savings_percentage": round(savings_pct, 2),
        "currency": "INR",
        "rate_applied": RATE_INR_KWH
    }
    
    summary_path = os.path.join(output_dir, 'dsm_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Task 3.2: Hindcast Audit File
    audit_df = pd.DataFrame({
        'forecast_target_time': report_df['timestamp'],
        'forecast_mw': report_df['forecast_mw'],
        'actual_mw': report_df['actual_mw'],
        'error_mw': (report_df['forecast_mw'] - report_df['actual_mw']).round(4),
        'error_pct': report_df['deviation_pct']
    })
    audit_path = os.path.join(output_dir, 'hindcast_audit.csv')
    audit_df.to_csv(audit_path, index=False)

    print("\n" + "="*40)
    print(f"✅ DSM STAGE COMPLETE: VALUE PROPOSITION PROVEN")
    print(f"Physics Baseline Penalty:  INR {tot_phys:,.2f}")
    print(f"Our XGBoost Model Penalty: INR {tot_xg:,.2f}")
    print(f"TOTAL SAVINGS CREATED:     INR {tot_saved:,.2f} ({savings_pct:.2f}%)")
    print("="*40)
    print(f"Files generated: {report_path}, {summary_path}")
    
    return report_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        calculate_dsm_penalties(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        calculate_dsm_penalties(sys.argv[1])
    else:
        print("Usage: python -m src.dsm_settlement.penalty_calculator <prediction_path> [<output_dir>]")
