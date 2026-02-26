# Solar MVP-DSM: Bhadla Solar Park (v2)
**ClimateForte: End-to-End Solar Power Forecasting & Regulatory Settlement Pipeline**

This repository implements a production-grade pipeline for solar power forecasting and **Deviation Settlement Mechanism (DSM)** calculation, specifically tuned for a 50MW plant in Bhadla, Rajasthan.

---

## 📂 Project Structure

```text
.
├── AWS-Data/               # Scrapers for AWS Weather India (Ground Truth)
├── dags/                   # Airflow DAGs for production orchestration
├── data/                   # Structured directory for pipeline intermediate files
│   ├── S1_Ingestion/       # Raw NASA/AWS data
│   ├── S2_Physics/         # Downscaled, Interpolated, and PVLib outputs
│   └── S3_ML/              # XGBoost corrected forecasts
├── results/                # Final business-ready reports and metrics
│   └── DSM/{date}/         # Daily reports, CSVs, and Performance Charts
├── src/                    # Core source code (Stage-wise modules)
│   ├── physics_baseline/   # S2: PVLib modeling logic
│   ├── ml_correction/      # S3: XGBoost residual learning
│   └── dsm_settlement/     # S4-S5: Penalty calculation & Reporting
├── downscaling/            # Spatial downscaling scripts (25km -> 1km)
├── temporal_interpolation/ # Time interpolation (1h -> 15-min)
├── models/                 # Pre-trained XGBoost Model binaries (.json)
└── requirements.txt        # Project dependencies
```

---

## 🔄 Data Flow & Pipeline Stages

The pipeline follows a strict **S1-S5 workflow** designed for regulatory auditability and high-precision modeling.

### 🚜 Stage 1: Data Ingestion (NASA & AWS)
*   **Source A**: NASA GEOS-FP Radiation data (Global 0.25°).
*   **Source B**: AWS Weather India (High-density ground truth for RJ/GJ).
*   **Processing**: Automated scraping, parallel downloading, and IST-to-UTC timestamp alignment.

### 📐 Stage 2: Physics Foundation (PVLib)
*   **Spatial Transformation**: Radiation data is downscaled from 50km to **1km resolution** centered on Bhadla (27°N, 72°E).
*   **Temporal Expansion**: Hourly data is interpolated to **15-minute blocks** (96 blocks/day) mandated by CERC.
*   **Modeling**: A **PVLib Faiman model** converts weather parameters (GHI, Temp, Wind) into a theoretical power baseline.

### 🧠 Stage 3: Residual Learning (XGBoost)
*   **Correction**: A gradient-boosted model (XGBoost) learns the systematic bias of the physics model (Residuals).
*   **Actuals Synthesis**: For audit purposes, 'Actual Generation' is synthesized by applying ±5% realistic noise to the physical baseline (Task 1.2 requirement).
*   **Output**: `forecast_corrected.csv`.

### 💰 Stage 4: DSM Settlement
*   **Regulatory Logic**: Applies the **Mandatory CERC 15% Penalty Threshold**.
*   **Calculation**: Computes deviation percentage per 15-min block and translates error into INR financial impact.
*   **Output**: `dsm_report.csv` (Revenue impact audit trail).

### 📊 Stage 5: Reporting & Metrics
*   **Skill Score**: Measures the improvement of XGBoost over the PVLib baseline (Benchmark: Skill > 0.15).
*   **Dashboards**: Generates high-fidelity visual reports for plant operators.

---

## 📈 Final Deliverables (Stage 5)

Each pipeline run generates the following in `results/DSM/{date}/`:

| File | Type | Purpose |
|------|------|---------|
| `performance_dashboard.png` | **Chart** | Power curves vs CERC 15% Safe Zone. |
| `daily_penalty_chart.png` | **Chart** | Penalty saved by using XGBoost vs PVLib only. |
| `metrics_comparison.json` | **Data** | Precision metrics: nMAE < 5%, RMSE < 4MW. |
| `hindcast_audit.csv` | **Data** | Block-by-block comparison for regulatory submission. |

---

## 🚀 How to Run

### Automated (Airflow)
The pipeline is fully orchestrated via the `solar_mvp_dsm_production_pipeline` DAG.
1. Build the image: `docker-compose build`
2. Start: `docker-compose up -d`
3. Trigger the DAG in the UI at `localhost:8080`.

### Manual Reporting
To regenerate reports for an existing run:
```bash
# Path to your daily DSM report
export REPORT_PATH="results/DSM/13-02-2026/dsm_report.csv"

# Run the reporting module
.venv/bin/python3 src/dsm_settlement/reporting.py $REPORT_PATH
```

---
**ClimateForte Solar DSM MVP Execution Directive v1.0** 🛰️☀️
