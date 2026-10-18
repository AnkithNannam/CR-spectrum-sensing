# Spectrum Sensing for Cognitive Radio

Simulation and comparison of spectrum-sensing detectors (energy, matched filter,
cyclostationary, eigenvalue, deep learning, cooperative) for a secondary user
detecting a primary user over AWGN / Rayleigh / Rician channels.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Everything is seeded from `config.SEED`; each experiment derives independent
RNG streams with `make_rng(seed, *tags)`, so results are reproducible.

## Run (from the project root)
```bash
python experiments/exp00_sanity_checks.py       # M1: signal/channel validation
python experiments/exp01_validate_pfa.py        # M2: ED Pfa validation (exact / CLT / MC thresholds)
python experiments/exp02_ed_pd_vs_snr.py        # M2: ED Pd vs SNR, simulation vs theory
```
Figures go to `results/figures/` (PNG + vector PDF); tables to `results/tables/`.

## Project status
| Milestone | Status |
|---|---|
| M0 setup, M1 signals/channels, M2 energy detector | done, validated |
| M3 matched filter + cyclostationary | next |
| M4 eigenvalue (MME/EME/AIC-MDL) | planned |
| M5 deep learning (1D CNN on I/Q, LSTM ablation) | planned |
| M6 cooperative sensing | planned |
| M7 full experiments, M8 report + viva notes | planned |

## Conventions
* Nominal noise variance = 1; PU samples have unit power; SNR is set by scaling.
* Every detector exposes `statistic(y) -> (n_trials,)`, larger = more likely H1,
  so ROC, thresholds and cooperative fusion work uniformly across detectors.
* Block fading by default (one channel gain per sensing window).
