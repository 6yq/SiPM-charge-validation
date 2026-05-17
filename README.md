# SiPM Charge Validation

Fitting and validation pipeline for Hamamatsu PCB6 SiPM charge spectra using the
Generalized-Tweedie / Negative-Afterpulse model.

## Quick start

```bash
make          # download → fit all voltages → validate
make data     # download only
make results  # fit only (requires data/)
```

## Pipeline

```
data/PCB6_MPPC_<V>V_histo.txt   ←  scripts/download.py  (Zenodo)
        ↓
results/PCB6_MPPC_<V>V.json     ←  scripts/fit.py  (NegBinBetaAPFitter)
        ↓
results/fit_results.csv          ←  scripts/validate.py
figures/fits.pdf                  (per-voltage histogram + model overlay)
figures/validation.pdf            (parameter trends, correlation matrices)
```

## Model

The spectrum model is a Compound Generalized-Poisson / Gamma distribution with a
Gaussian pedestal and NegBin (Geom0) per-avalanche afterpulse counts.  The
default AP charge model is bounded: `Q_AP = G X`, `X ~ Beta(2,b)`, with the
mean charge controlled by the fitted `beta` parameter.

Parameters fitted per voltage:

| Parameter | Description |
|-----------|-------------|
| `ped_mean`, `ped_sigma` | Gaussian pedestal |
| `spe_mean`, `spe_sigma` | SPE Gamma mean and width |
| `xi` | Gen-Poisson dispersion |
| `rho` | Per-avalanche afterpulse probability |
| `beta` | Mean afterpulse charge fraction |
| `lam` | Mean PE per event (occupancy) |

## Data

Hamamatsu PCB6 MPPC at 16 bias voltages (53 – 60 V in 0.5 V steps), measured
under illumination.

> Rolph, J. et al. (2023). *PeakOTron* [Software + Dataset].
> Zenodo. <https://doi.org/10.5281/zenodo.10014537>
>
> Source data also available at:
> <https://gitlab.desy.de/jack.rolph/peakotron/-/tree/main/data/hamamatsu_pcb6/Light>

## Requirements

The venv at `/mnt/stage/liuyq/tao/venv` provides JAX, jax-finufft, numpy,
scipy, matplotlib, pandas.  The project root and `/mnt/stage/liuyq/tao/jax_dep`
are exported in `PYTHONPATH` by the Makefile so `import fitter` resolves.
