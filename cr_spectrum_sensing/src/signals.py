"""Primary-user (PU) waveform generators.

All generators return complex baseband samples of shape (n_trials, N) with
UNIT average power, so the SNR is controlled purely by scaling in the channel
stage. Each trial uses fresh random data (independent realisations).

Single-carrier signals use `sps` samples per symbol with a rectangular pulse
(sps=1 => symbol-rate sampling, i.i.d. symbols). The cyclostationary detector
(M3) needs sps>1 to expose a cyclic frequency at fs/sps.
"""
import numpy as np

# Excess fourth moment kappa = E|s|^4 - 1 for unit-power s.  It enters the
# variance of the energy statistic (see energy_detector.py):
#   BPSK/QPSK constant-envelope -> 0 ; 16-QAM -> 0.32 ; OFDM ~ Gaussian -> 1
MOD_KAPPA = {"BPSK": 0.0, "QPSK": 0.0, "16QAM": 0.32, "OFDM": 1.0}

OFDM_NFFT = 64
OFDM_CP = 16


def _symbols(mod, shape, rng):
    """Unit-power i.i.d. constellation symbols of the given shape."""
    if mod == "BPSK":
        return (2 * rng.integers(0, 2, shape) - 1).astype(np.complex128)
    if mod == "QPSK":
        re = 2 * rng.integers(0, 2, shape) - 1
        im = 2 * rng.integers(0, 2, shape) - 1
        return (re + 1j * im) / np.sqrt(2.0)
    if mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        re = lv[rng.integers(0, 4, shape)]
        im = lv[rng.integers(0, 4, shape)]
        return (re + 1j * im) / np.sqrt(10.0)      # E|s|^2 = 10/10 = 1
    raise ValueError(f"unknown modulation {mod}")


def _ofdm(n_trials, N, rng):
    """OFDM: 64 subcarriers of QPSK, 16-sample cyclic prefix, random start offset."""
    sym_len = OFDM_NFFT + OFDM_CP
    n_sym = int(np.ceil(N / sym_len)) + 1
    X = _symbols("QPSK", (n_trials, n_sym, OFDM_NFFT), rng)
    x = np.fft.ifft(X, axis=-1, norm="ortho")        # unitary IFFT keeps unit power
    x = np.concatenate([x[..., -OFDM_CP:], x], axis=-1)   # prepend cyclic prefix
    x = x.reshape(n_trials, n_sym * sym_len)
    off = rng.integers(0, sym_len, size=(n_trials, 1))    # unknown symbol timing
    idx = off + np.arange(N)[None, :]
    return np.take_along_axis(x, idx, axis=1)


def generate_pu(mod, n_trials, N, rng, sps=1):
    """Return (n_trials, N) complex PU samples with unit average power."""
    if mod == "OFDM":
        return _ofdm(n_trials, N, rng)
    n_sym = int(np.ceil(N / sps)) + 1
    s = _symbols(mod, (n_trials, n_sym), rng)
    if sps > 1:                                      # rectangular pulse shaping
        s = np.repeat(s, sps, axis=1)
    off = rng.integers(0, sps, size=(n_trials, 1)) if sps > 1 else np.zeros((n_trials, 1), int)
    idx = off + np.arange(N)[None, :]
    return np.take_along_axis(s, idx, axis=1)
