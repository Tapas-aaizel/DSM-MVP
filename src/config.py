import os

# Project Constants (Bhadla, Rajasthan Pilot)
LATITUDE = 27.5      # Specified in Section 8
LONGITUDE = 71.9     # Bhadla Region
TIMEZONE = "Asia/Kolkata"
ALTITUDE = 225       # meters

# Plant Configuration (Directive Section 2.3)
FIXED_TILT = 27.5    # Must match Latitude
AZIMUTH = 180        # True South
CAPACITY_MW = 50.0   # Normalisation constant for nMAE

# Module: Generic 400W mono-Si
MODULE_PARAMS = {
    'pdc0': 400,          # Rated power (W)
    'gamma_pdc': -0.0045, # Temperature coefficient (%/°C)
    'A_c': 1.95,          # Module area (m²)
    'eta_m': 0.20,         # Module efficiency
}

# Inverter: Generic 500kW string inverter
INVERTER_PARAMS = {
    'pdc0': 500000,       # DC input limit (W)
    'eta_inv_nom': 0.98,  # Nominal efficiency
}

# Temperature Model: Faiman (accounts for wind cooling)
FAIMAN_PARAMS = {
    'u0': 25.0,           # Combined heat loss factor
    'u1': 6.84,           # Wind-dependent heat loss factor
}

# DSM Logic (Section 4.1)
DSM_RATE_INR = 1.50
BAND_1_DEV = 15
BAND_2_DEV = 25

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DELIVERABLES_DIR = os.path.join(BASE_DIR, "deliverables")

# Create directories
for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, DELIVERABLES_DIR]:
    os.makedirs(path, exist_ok=True)
