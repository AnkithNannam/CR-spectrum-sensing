"""Energy detector (ED).

Test statistic (normalised by the NOMINAL noise variance sigma_hat^2):

        T(y) = (1/(N sigma_hat^2)) * sum_n |y[n]|^2        decide H1 if T > lambda

Distribution under H0 (complex Gaussian noise, variance sigma^2 = sigma_hat^2):
        (2N) * T  ~  chi-square with 2N degrees of freedom      (exact)
        T ~ N(1, 1/N) for large N                                (CLT)

Thresholds for a target false-alarm probability Pfa:
        exact :  lambda = chi2_isf(Pfa, 2N) / (2N)
        CLT   :  lambda = 1 + Q^{-1}(Pfa) / sqrt(N)
        MC    :  empirical (1-Pfa)-quantile of T simulated under H0

Distribution under H1 (AWGN, instantaneous SNR g = gamma |h|^2), samples iid:
        E[T] = 1 + g
        Var[T] = (1 + 2g + kappa g^2) / N,     kappa = E|s|^4 - 1
   (kappa = 0 for BPSK/QPSK, 0.32 for 16-QAM, 1 for Gaussian-like OFDM)
        Pd(g) = Q( (lambda - 1 - g) / sqrt(Var[T]) )

Under fading, average Pd(g) over the pdf of g = gamma*|h|^2 (block fading).

SNR wall under noise uncertainty (true noise = rho * sigma_hat^2, rho_max = 10^(x/10)):
        gamma_wall = (rho_max^2 - 1) / rho_max          (linear)
   Below this SNR no N can reach the target Pd/Pfa.
"""
import numpy as np
from scipy.stats import chi2, ncx2
from scipy.integrate import trapezoid
from ..utils import qfunc, qfuncinv, db2lin, lin2db, empirical_threshold
from ..signals import MOD_KAPPA
from ..simulator import collect_stats


# ----------------------------------------------------------------- statistic
def statistic(y, sigma2_hat=1.0):
    """Normalised energy per window. y: (n_trials, N) complex -> (n_trials,)."""
    return np.mean(np.abs(y) ** 2, axis=1) / sigma2_hat


# ---------------------------------------------------------------- thresholds
def threshold_clt(N, pfa):
    return 1.0 + qfuncinv(pfa) / np.sqrt(N)


def threshold_exact(N, pfa):
    return chi2.isf(pfa, 2 * N) / (2.0 * N)


def threshold_mc(sc, pfa, n_trials, rng):
    """Monte Carlo threshold: simulate H0 windows, take the (1-Pfa) quantile.
    Calibrated at the NOMINAL noise (sc.nu_db is forced to 0): the detector
    cannot know the true noise level, which is exactly why the SNR wall exists."""
    from dataclasses import replace
    t0 = collect_stats(statistic, 0, replace(sc, nu_db=0.0), n_trials, rng)
    return empirical_threshold(t0, pfa)


# ------------------------------------------------------------------- theory
def pfa_theory(lam, N):
    """Exact Pfa = P(T > lam | H0)."""
    return chi2.sf(2.0 * N * lam, 2 * N)


def pd_theory(snr_db, N, pfa=None, thr=None, channel="awgn", k_db=6.0, mod="BPSK"):
    """Closed-form / semi-analytic Pd (CLT under H1). Accepts scalar or array snr_db."""
    lam = thr if thr is not None else threshold_exact(N, pfa)
    kappa = MOD_KAPPA[mod]
    gamma = np.atleast_1d(db2lin(snr_db)).astype(float)

    def pd_cond(g):
        return qfunc((lam - 1.0 - g) / np.sqrt((1.0 + 2.0 * g + kappa * g ** 2) / N))

    if channel == "awgn":
        out = pd_cond(gamma)
    else:
        K = float(db2lin(k_db)) if channel == "rician" else 0.0
        # |h|^2 pdf: Rician -> scaled noncentral chi2; Rayleigh (K=0) -> exponential
        dist = chi2(df=2, scale=0.5) if K == 0 else ncx2(df=2, nc=2 * K, scale=1.0 / (2 * (K + 1)))
        x = np.linspace(dist.ppf(1e-7), dist.ppf(1 - 1e-7), 4000)
        out = trapezoid(pd_cond(gamma[:, None] * x[None, :]) * dist.pdf(x)[None, :], x, axis=1)
    return out if np.ndim(snr_db) else float(out[0])


def snr_wall_db(nu_db):
    """SNR wall (dB) for +/- nu_db noise uncertainty."""
    rho = db2lin(nu_db)
    return lin2db((rho ** 2 - 1.0) / rho)
