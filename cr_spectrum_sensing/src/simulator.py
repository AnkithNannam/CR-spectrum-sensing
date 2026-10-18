"""Observation generator: produces H0 / H1 sensing windows for a Scenario.

    H0 : y[n] = w[n]
    H1 : y[n] = sqrt(gamma) * h * s[n] + w[n]

gamma = nominal SNR (linear) = E|sqrt(gamma) h s|^2 / sigma_hat^2 with sigma_hat^2 = 1.
"""
from dataclasses import dataclass
import numpy as np
from .utils import db2lin
from .signals import generate_pu
from .channels import cn, fading_coeff, noise_uncertainty_factor


@dataclass(frozen=True)
class Scenario:
    mod: str = "BPSK"
    N: int = 1000
    snr_db: float = -10.0
    channel: str = "awgn"          # awgn | rayleigh | rician
    k_db: float = 6.0              # Rician K-factor
    nu_db: float = 0.0             # noise-uncertainty bound (+/- dB)
    fast_fading: bool = False      # False: block fading (one h per window)
    sps: int = 1                   # samples per symbol


def observe(H, sc, n_trials, rng):
    """Return complex array (n_trials, N) of sensing windows under hypothesis H in {0,1}."""
    rho = noise_uncertainty_factor(n_trials, sc.nu_db, rng)          # (n_trials,1)
    w = np.sqrt(rho) * cn((n_trials, sc.N), rng)
    if H == 0:
        return w
    s = generate_pu(sc.mod, n_trials, sc.N, rng, sps=sc.sps)
    h_shape = (n_trials, sc.N) if sc.fast_fading else (n_trials, 1)
    h = fading_coeff(sc.channel, h_shape, rng, sc.k_db)
    return np.sqrt(db2lin(sc.snr_db)) * h * s + w


def collect_stats(stat_fn, H, sc, n_trials, rng):
    """Monte Carlo: apply stat_fn(y)->(m,) over n_trials windows, in memory-safe chunks."""
    chunk = int(max(50, 4e6 // sc.N))
    out, done = [], 0
    while done < n_trials:
        m = min(chunk, n_trials - done)
        out.append(stat_fn(observe(H, sc, m, rng)))
        done += m
    return np.concatenate(out)
