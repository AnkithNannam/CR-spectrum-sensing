"""M2 validation gate: does each ED threshold give the target Pfa under H0?

Thresholds compared: exact chi-square, CLT (Gaussian), Monte Carlo.
Test windows are drawn from an RNG stream independent of the one used to
calibrate the MC threshold. Also shows how noise uncertainty breaks the Pfa.
"""
import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
import config as C
from src.utils import make_rng, set_plot_style, save_fig, wilson_ci, TAB_DIR
from src.simulator import Scenario, collect_stats
from src.detectors import energy_detector as ed

set_plot_style()
N_LIST = [100, 500, 1000, 5000]
PFAS = [0.1, 0.01]
N_TEST = C.N_MC_THRESHOLD

rows = []
print(f"{'N':>5} {'Pfa*':>6} {'method':>6} {'lambda':>8} {'Pfa_emp':>8} {'95% CI':>17}  ok")
for i, N in enumerate(N_LIST):
    sc = Scenario(N=N)
    test_stats = collect_stats(ed.statistic, 0, sc, N_TEST, make_rng(C.SEED, 101, N))       # test stream
    for pfa in PFAS:
        thr = {
            "exact": ed.threshold_exact(N, pfa),
            "CLT": ed.threshold_clt(N, pfa),
            "MC": ed.threshold_mc(sc, pfa, C.N_MC_THRESHOLD, make_rng(C.SEED, 102, N)),     # calibration stream
        }
        for m, lam in thr.items():
            k = int(np.sum(test_stats > lam)); p = k / N_TEST
            lo, hi = wilson_ci(k, N_TEST)
            ok = lo <= pfa <= hi
            rows.append(dict(N=N, pfa_target=pfa, method=m, lam=lam, pfa_emp=p, ci_lo=lo, ci_hi=hi, ok=ok))
            print(f"{N:5d} {pfa:6.2f} {m:>6} {lam:8.4f} {p:8.4f} [{lo:.4f},{hi:.4f}]  {'PASS' if ok else 'FAIL'}")

# --- noise uncertainty breaks Pfa (preview of the SNR wall) ------------------
print("\nNoise uncertainty (+/-1 dB), threshold calibrated at nominal noise, N=1000, Pfa*=0.1")
sc_nu = Scenario(N=1000, nu_db=C.NOISE_UNC_DB)
t = collect_stats(ed.statistic, 0, sc_nu, N_TEST, make_rng(C.SEED, 103))
lam = ed.threshold_exact(1000, 0.1)
print(f"  empirical Pfa = {np.mean(t > lam):.3f}   (target 0.1)  -> detector can no longer hold its Pfa")

with open(TAB_DIR / "validation_energy_pfa.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# --- figure ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
for ax, pfa in zip(axes, PFAS):
    for j, (m, mk) in enumerate(zip(["exact", "CLT", "MC"], ["o", "s", "^"])):
        r = [x for x in rows if x["pfa_target"] == pfa and x["method"] == m]
        xs = np.arange(len(N_LIST)) + (j - 1) * 0.12
        ax.errorbar(xs, [x["pfa_emp"] for x in r],
                    yerr=[[x["pfa_emp"] - x["ci_lo"] for x in r], [x["ci_hi"] - x["pfa_emp"] for x in r]],
                    fmt=mk, capsize=3, label=m)
    ax.axhline(pfa, color="k", ls=":", label="target")
    ax.set_xticks(range(len(N_LIST))); ax.set_xticklabels(N_LIST)
    ax.set_xlabel("Samples per window, $N$"); ax.set_ylabel("Empirical $P_{fa}$")
    ax.set_title(f"Energy detector, target $P_{{fa}}={pfa}$"); ax.legend()
save_fig(fig, "fig01_ed_pfa_validation")
