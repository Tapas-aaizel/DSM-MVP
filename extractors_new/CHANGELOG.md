# GEOS-FP Extractors - Change Log

## December 12, 2025 - Migration to GEOS-FP

### Overview

Migrated from MERRA-2 to GEOS-FP for near real-time weather prediction pipeline.

### Why GEOS-FP?

| Aspect | MERRA-2 | GEOS-FP |
|--------|---------|---------|
| Data Delay | ~35 days | ~1-2 days |
| Resolution | 0.5 x 0.625 deg | 0.25 x 0.25 deg |
| Use Case | Historical reanalysis | Near real-time forecasting |

### Files Removed (MERRA-2)

- mera.py (M2T1NXRAD)
- meraslv.py (M2T1NXSLV)
- meraflx.py (M2T1NXFLX)
- merraroot.py (M2T1NXLND)
- merapres.py (M2I6NPANA)
- merapres_asm.py (M2I3NPASM)
- download_utils.py (earthaccess utilities)
- prithvi_merra2_extraction_dag.py

### Files Added/Updated (GEOS-FP)

- geosfp_prithvi_downloader.py - Main downloader (updated)
- geosfp_inst3_2d_asm.py - Surface atmospheric
- geosfp_tavg1_2d_flx.py - Surface flux
- geosfp_tavg1_2d_lnd.py - Land surface
- geosfp_tavg1_2d_rad.py - Radiation
- geosfp_tavg1_2d_slv.py - Sea ice fraction
- geosfp_const_2d_asm.py - Static constants
- geosfp_inst3_3d_asm_Nv.py - 3D model levels (all vertical vars)
- prithvi_geosfp_extraction_dag.py - Updated DAG

### Key Changes

1. Removed inst3_3d_asm_Np (pressure levels) - Prithvi uses model levels (Nv)
2. All 10 vertical variables now come from inst3_3d_asm_Nv
3. Simplified DAG with 7 products instead of 8
4. No authentication required (GEOS-FP is public data)

### Variable Coverage

Surface Variables (20):
- inst3_2d_asm: PS, QV2M, SLP, T2M, TQI, TQL, TQV, TS, U10M, V10M
- tavg1_2d_flx: EFLUX, HFLUX, Z0M
- tavg1_2d_lnd: GWETROOT, LAI
- tavg1_2d_rad: LWGAB, LWGEM, LWTUP, SWGNT, SWTNT

Static Variables (4):
- tavg1_2d_slv: FRSEAICE (mapped to FRACI)
- const_2d_asm: FRLAND, FROCEAN, PHIS

Vertical Variables (10 x 13 levels = 130):
- inst3_3d_asm_Nv: CLOUD, H, OMEGA, PL, QI, QL, QV, T, U, V

Total: ~154 channels

### Data Source

NCCS Portal: https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/

---

## Pipeline Components Status

### Phase 1: Infrastructure ✅
- docker-compose.yml with MinIO, PostGIS, Airflow, AWS Collector
- .env.example for configuration

### Phase 2: Pre-processing ✅
- transformation/regridder.py - Regrids all collections to MERRA-2 grid
- transformation/assembler.py - Assembles into Prithvi input format
- transformation/merge_surface.py - Legacy surface merger
- transformation/merge_pressure.py - Legacy pressure merger

### Phase 3: Inference ✅ (Container Ready)
- inference/Dockerfile - GPU container with PyTorch/CUDA
- inference/inference.py - Prithvi rollout inference script
- inference/requirements.txt

### Phase 4: Post-processing ✅
- downscaling/downscaler.py - 0.5° to 5km interpolation for India
- downscaling/Dockerfile
- transformation/bias_correction.py - Ground truth correction

### Phase 5: Formatting ✅
- transformation/formatter.py - Zarr/COG conversion + MinIO upload

### Phase 6: Orchestration ✅
- dags/prithvi_pipeline_dag.py - Full pipeline DAG
- dags/prithvi_geosfp_extraction_dag.py - Data extraction DAG

---

Last Updated: December 12, 2025
