"""M1 validation: signal power, SNR scaling, fading normalisation, noise uncertainty."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import replace
import config as C
from src.utils import make_rng, lin2db
from src.signals import generate_pu, MOD_KAPPA
from src.channels import fading_coeff
from src.simulator import Scenario, observe

rng = make_rng(C.SEED, 0)
print("== PU signal power and fourth moment (expect power=1, kappa as tabulated) ==")
for mod in C.MODS:
    s = generate_pu(mod, 4000, 800, rng)
    p = np.mean(np.abs(s) ** 2)
    kap = np.mean(np.abs(s) ** 4) - 1.0
    print(f"  {mod:6s} power={p:.4f}  kappa_emp={kap:.3f}  kappa_table={MOD_KAPPA[mod]}")

print("== Fading normalisation E|h|^2 (expect 1) ==")
for ch in C.CHANNELS:
    h = fading_coeff(ch, (200000, 1), rng, C.RICIAN_K_DB)
    print(f"  {ch:9s} E|h|^2 = {np.mean(np.abs(h)**2):.4f}")

print("== Measured SNR of H1 windows minus noise (AWGN, expect = nominal) ==")
for snr in (-10, 0, 10):
    sc = Scenario(mod="QPSK", N=2000, snr_db=snr)
    y1 = observe(1, sc, 3000, rng); y0 = observe(0, sc, 3000, rng)
    sig_pow = np.mean(np.abs(y1) ** 2) - np.mean(np.abs(y0) ** 2)
    print(f"  nominal {snr:+d} dB -> measured {lin2db(sig_pow / np.mean(np.abs(y0)**2)):+.2f} dB")

print("== Noise uncertainty: mean true noise power in dB (expect ~0, range +/-1) ==")
y0 = observe(0, Scenario(N=500, nu_db=1.0), 20000, rng)
pw = lin2db(np.mean(np.abs(y0) ** 2, axis=1))
print(f"  mean={pw.mean():+.3f} dB  min={pw.min():+.2f}  max={pw.max():+.2f}")

print("== Reproducibility: same seed -> identical samples ==")
a = observe(1, Scenario(mod="OFDM"), 5, make_rng(C.SEED, 7)); b = observe(1, Scenario(mod="OFDM"), 5, make_rng(C.SEED, 7))
print("  identical:", np.array_equal(a, b))
