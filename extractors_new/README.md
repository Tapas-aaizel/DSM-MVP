# GEOS-FP Extractors for Prithvi-WxC-2.3B

Near real-time GEOS-FP data extraction for the Prithvi weather prediction model.

## Pipeline Context

This is Phase 1 (Ingest) of the ClimateForte pipeline:
1. Ingest (this) - Download GEOS-FP to MinIO scratch-space/raw
2. Pre-process - Regrid to 0.5 deg, Assemble with constants
3. Inference - Run Prithvi model
4. Downscale - Interpolate to 5km, crop to India
5. Bias Correction - Apply ground truth
6. Formatter - Convert to Zarr/COG for API

## GEOS-FP vs MERRA-2

| Aspect | GEOS-FP | MERRA-2 |
|--------|---------|---------|
| Data Delay | 1-2 days | 35 days |
| Resolution | 0.25 deg | 0.5 x 0.625 deg |
| Use Case | Near real-time | Historical |

## Products and Variables

### Surface Variables (20 total)

| Script | Product | Variables |
|--------|---------|-----------|
| geosfp_inst3_2d_asm.py | inst3_2d_asm_Nx | PS, QV2M, SLP, T2M, TQI, TQL, TQV, TS, U10M, V10M |
| geosfp_tavg1_2d_flx.py | tavg1_2d_flx_Nx | EFLUX, HFLUX, Z0M |
| geosfp_tavg1_2d_lnd.py | tavg1_2d_lnd_Nx | GWETROOT, LAI |
| geosfp_tavg1_2d_rad.py | tavg1_2d_rad_Nx | LWGAB, LWGEM, LWTUP, SWGNT, SWTNT |
| geosfp_tavg1_2d_slv.py | tavg1_2d_slv_Nx | FRSEAICE to FRACI |

### Static Variables (3 total)

| Script | Product | Variables |
|--------|---------|-----------|
| geosfp_const_2d_asm.py | const_2d_asm_Nx | FRLAND, FROCEAN, PHIS |

### Vertical Variables (10 vars x 13 levels = 130 channels)

| Script | Product | Variables |
|--------|---------|-----------|
| geosfp_inst3_3d_asm_Nv.py | inst3_3d_asm_Nv | CLOUD, H, OMEGA, PL, QI, QL, QV, T, U, V |

Note: Prithvi-WxC uses model levels (Nv), not pressure levels (Np).

## Total Channels

- Surface: 20 variables
- Static: 4 variables (FRACI from slv + 3 from const)
- Vertical: 10 variables x 13 levels = 130 channels
- Total: ~154 channels

## Usage

### Via Airflow DAG (Recommended)
Trigger from Airflow UI: prithvi_geosfp_extraction
Select date from calendar picker

### Standalone Testing
```bash
python geosfp_prithvi_downloader.py --date 2025-12-10 --dry-run
python geosfp_inst3_2d_asm.py
python geosfp_inst3_3d_asm_Nv.py
```

## Data Source

NCCS Portal: https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/

No authentication required (public data).
