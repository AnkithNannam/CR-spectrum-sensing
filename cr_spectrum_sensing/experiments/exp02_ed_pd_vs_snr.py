"""M2 validation: Pd vs SNR for the energy detector, Monte Carlo vs theory.

(a) AWGN, all four PU modulations  -> checks the kappa-dependent variance term
(b) QPSK over AWGN / Rayleigh / Rician (K=6 dB) -> checks the fading average
Threshold: exact chi-square for Pfa = 0.1.  Pd (MC) uses independent H1 draws.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import config as C
from src.utils import make_rng, set_plot_style, save_fig, wilson_ci
from src.simulator import Scenario, collect_stats
from src.detectors import energy_detector as ed

set_plot_style()
N, PFA, snrs = C.N_DEFAULT, C.PFA_TARGET, C.SNR_DB_GRID
lam = ed.threshold_exact(N, PFA)
print(f"N={N}, Pfa={PFA}, lambda={lam:.4f}, trials/point={C.N_MC_TRIALS}")

def mc_pd(mod, channel, tag):
    pd = []
    for i, snr in enumerate(snrs):
        sc = Scenario(mod=mod, N=N, snr_db=snr, channel=channel, k_db=C.RICIAN_K_DB)
        t = collect_stats(ed.statistic, 1, sc, C.N_MC_TRIALS, make_rng(C.SEED, 201, tag, i))
        pd.append(np.mean(t > lam))
    return np.array(pd)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
dense = np.linspace(snrs[0], snrs[-1], 200)
errs = []
print("\n(a) AWGN, modulation dependence")
for j, mod in enumerate(C.MODS):
    pd = mc_pd(mod, "awgn", j); th = ed.pd_theory(snrs, N, thr=lam, mod=mod)
    errs.append(np.max(np.abs(pd - th)))
    l, = a.plot(dense, ed.pd_theory(dense, N, thr=lam, mod=mod), "-")
    a.plot(snrs, pd, "o", color=l.get_color(), mfc="none", label=f"{mod} (sim)")
    print(f"  {mod:6s} max|Pd_sim - Pd_theory| = {errs[-1]:.3f}")
a.plot([], [], "k-", label="theory (CLT)")
a.set_xlabel("SNR (dB)"); a.set_ylabel("Probability of detection, $P_d$")
a.set_title(f"AWGN, $N={N}$, $P_{{fa}}={PFA}$"); a.legend(); a.set_ylim(-0.02, 1.02)

print("\n(b) QPSK, fading dependence")
for j, ch in enumerate(C.CHANNELS):
    pd = mc_pd("QPSK", ch, 10 + j); th = ed.pd_theory(snrs, N, thr=lam, channel=ch, k_db=C.RICIAN_K_DB, mod="QPSK")
    l, = b.plot(dense, ed.pd_theory(dense, N, thr=lam, channel=ch, k_db=C.RICIAN_K_DB, mod="QPSK"), "-")
    b.plot(snrs, pd, "s", color=l.get_color(), mfc="none", label=f"{ch} (sim)")
    print(f"  {ch:9s} max|Pd_sim - Pd_theory| = {np.max(np.abs(pd - th)):.3f}")
b.plot([], [], "k-", label="theory")
b.set_xlabel("SNR (dB)"); b.set_ylabel("Probability of detection, $P_d$")
b.set_title(f"QPSK, $N={N}$, $P_{{fa}}={PFA}$"); b.legend(); b.set_ylim(-0.02, 1.02)
save_fig(fig, "fig02_ed_pd_vs_snr_validation")

# Required SNR for Pd=0.9 : closed form check  N ~ 2*(Q^-1(Pfa)-Q^-1(Pd))^2 / gamma^2 ... (low-SNR approx)
from src.utils import qfuncinv, lin2db
g = (qfuncinv(PFA) - qfuncinv(0.9)) / np.sqrt(N)
print(f"\nLow-SNR rule of thumb: gamma_req = (Q^-1(Pfa)-Q^-1(Pd))/sqrt(N) -> {lin2db(g):.2f} dB for Pd=0.9, N={N}")
print(f"SNR wall for +/-1 dB uncertainty: {ed.snr_wall_db(1.0):.2f} dB")
