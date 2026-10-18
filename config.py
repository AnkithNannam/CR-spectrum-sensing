"""Global experiment configuration (single source of truth).

Change values here, not inside experiment scripts, so every figure in the
report can be regenerated from one place.
"""
import numpy as np

# ---- Reproducibility ------------------------------------------------------
SEED = 2026                       # master seed; each experiment derives streams from it

# ---- Sweep grids (from the project spec) ----------------------------------
SNR_DB_GRID = np.arange(-20, 11, 2.0)          # -20 ... +10 dB
N_GRID = [100, 200, 500, 1000, 2000, 5000]     # samples per sensing window
N_DEFAULT = 1000
PFA_TARGET = 0.1                               # IEEE 802.22-style target
PFA_GRID = np.logspace(-3, 0, 60)              # for ROC sweeps

# ---- Signal / channel options ---------------------------------------------
MODS = ["BPSK", "QPSK", "16QAM", "OFDM"]
CHANNELS = ["awgn", "rayleigh", "rician"]
RICIAN_K_DB = 6.0                              # Rician K-factor
NOISE_UNC_DB = 1.0                             # noise uncertainty bound (+/- x dB)

# ---- Monte Carlo budgets ---------------------------------------------------
# Raise N_MC_THRESHOLD to >=1e5 for the final report-quality low-Pfa thresholds.
N_MC_THRESHOLD = 40_000        # H0 trials used to estimate an empirical threshold
N_MC_TRIALS = 2_000            # trials per (SNR, scenario) point for Pd estimates
