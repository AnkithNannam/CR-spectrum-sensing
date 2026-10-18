"""Channel and noise models.

Conventions
-----------
* Nominal noise variance is sigma_hat^2 = 1 (the value the detector *believes*).
* Complex noise: w ~ CN(0, sigma^2), i.e. real/imag parts each N(0, sigma^2/2).
* Fading gain h is normalised so that E|h|^2 = 1, hence the average received
  SNR is exactly the nominal SNR in every channel type.
* Noise uncertainty (Tandra & Sahai model): the TRUE noise variance is
  sigma^2 = rho * sigma_hat^2 where 10*log10(rho) ~ Uniform[-x, +x] dB,
  drawn independently per sensing window. The detector is still calibrated
  with sigma_hat^2 = 1.
"""
import numpy as np
from .utils import db2lin


def cn(shape, rng, var=1.0):
    """Circularly-symmetric complex Gaussian CN(0, var)."""
    return np.sqrt(var / 2.0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def fading_coeff(kind, shape, rng, k_db=6.0):
    """Channel gain h with E|h|^2 = 1.

    awgn     : h = 1
    rayleigh : h ~ CN(0,1)
    rician   : h = sqrt(K/(K+1)) + sqrt(1/(K+1)) * CN(0,1)   (LOS + scatter)
    """
    if kind == "awgn":
        return np.ones(shape, dtype=np.complex128)
    if kind == "rayleigh":
        return cn(shape, rng)
    if kind == "rician":
        K = float(db2lin(k_db))
        return np.sqrt(K / (K + 1.0)) + np.sqrt(1.0 / (K + 1.0)) * cn(shape, rng)
    raise ValueError(f"unknown channel {kind}")


def noise_uncertainty_factor(n_trials, nu_db, rng):
    """rho (linear) per trial, log-uniform in [-nu_db, +nu_db] dB. nu_db=0 -> rho=1."""
    if nu_db == 0:
        return np.ones((n_trials, 1))
    return db2lin(rng.uniform(-nu_db, nu_db, size=(n_trials, 1)))
