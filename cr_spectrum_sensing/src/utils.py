"""Shared helpers: dB conversions, Q-function, RNG streams, ROC tools, plotting."""
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")            # headless-safe backend; remove for interactive use
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
for _d in (FIG_DIR, TAB_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- basic maths
def db2lin(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def lin2db(x):
    return 10.0 * np.log10(x)


def qfunc(x):
    """Gaussian tail Q(x) = P(Z > x), Z ~ N(0,1)."""
    return norm.sf(x)


def qfuncinv(p):
    """Inverse Q-function."""
    return norm.isf(p)


def make_rng(seed, *tags):
    """Independent, reproducible RNG stream per (seed, tags...).

    Using SeedSequence guarantees different tags give statistically independent
    streams, so e.g. threshold calibration and testing never share samples.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(t) for t in tags]]))


# ------------------------------------------------------ empirical statistics
def empirical_threshold(stats_h0, pfa):
    """Threshold lambda such that the fraction of H0 statistics > lambda equals pfa."""
    return np.quantile(stats_h0, 1.0 - pfa, method="higher")


def empirical_rate(stats, thr):
    """P(stat > thr) estimated from samples (used for both Pfa and Pd)."""
    return float(np.mean(stats > thr))


def wilson_ci(k, n, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion k/n."""
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def roc_from_stats(s0, s1, max_points=2000):
    """Empirical ROC from H0 and H1 test statistics (larger stat => decide H1).

    Returns (pfa, pd) arrays, ordered by increasing threshold.
    """
    s0 = np.sort(np.asarray(s0))
    s1 = np.sort(np.asarray(s1))
    thr = np.sort(np.concatenate([s0, s1]))
    if thr.size > max_points:
        thr = thr[np.linspace(0, thr.size - 1, max_points).astype(int)]
    pfa = 1.0 - np.searchsorted(s0, thr, side="right") / s0.size
    pd = 1.0 - np.searchsorted(s1, thr, side="right") / s1.size
    return pfa, pd


# ------------------------------------------------------------------- plotting
def set_plot_style():
    """Publication-style defaults (IEEE-ish): serif font, grid, 300 dpi export."""
    plt.rcParams.update({
        "figure.figsize": (6.0, 4.2), "figure.dpi": 110, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.framealpha": 0.9,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
        "lines.linewidth": 1.6, "lines.markersize": 5,
    })


def save_fig(fig, name):
    """Save as both PNG (quick view) and PDF (vector, for LaTeX/Word)."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved results/figures/{name}.png|pdf")
