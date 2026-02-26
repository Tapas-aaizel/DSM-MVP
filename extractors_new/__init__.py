"""GEOS-FP Extractors for Prithvi-WxC-2.3B Model.

This package contains extraction scripts for all GEOS-FP products
required by the Prithvi weather prediction model.

Products:
- Surface: inst3_2d_asm, tavg1_2d_flx, tavg1_2d_lnd, tavg1_2d_rad, tavg1_2d_slv
- Static: const_2d_asm
- Vertical: inst3_3d_asm_Nv (model levels)
"""

from .geosfp_inst3_2d_asm import download_geosfp_inst3_2d_asm
from .geosfp_tavg1_2d_flx import download_geosfp_tavg1_2d_flx
from .geosfp_tavg1_2d_lnd import download_geosfp_tavg1_2d_lnd
from .geosfp_tavg1_2d_rad import download_geosfp_tavg1_2d_rad
from .geosfp_tavg1_2d_slv import download_geosfp_tavg1_2d_slv
from .geosfp_const_2d_asm import download_geosfp_const_2d_asm
from .geosfp_inst3_3d_asm_Nv import download_geosfp_inst3_3d_asm_Nv

__all__ = [
    "download_geosfp_inst3_2d_asm",
    "download_geosfp_tavg1_2d_flx",
    "download_geosfp_tavg1_2d_lnd",
    "download_geosfp_tavg1_2d_rad",
    "download_geosfp_tavg1_2d_slv",
    "download_geosfp_const_2d_asm",
    "download_geosfp_inst3_3d_asm_Nv",
]
