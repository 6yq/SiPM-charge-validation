import numpy as np
from math import log

from fitter.models.afterpulse import NegBinBetaAPFitter
from fitter.models.gen_tweedie import reparam_from_spe


def estimate_init(charges, counts):
    d = NegBinBetaAPFitter.estimate_from_histogram(charges, counts)
    return d["ped_mean"], d["ped_sigma"], d["spe_mean"], d["spe_sigma"], d["lam_est"]


def estimate_dark_mu(charges, counts, ped_mean, spe_mean, T_gate, t0_pre, tau_slow):
    """Rough mu_dark estimate from 0PE–1PE valley density (PeakOTron-style init).

    PeakOTron estimates a DCR rate from the density near K=0.5, normalised to
    the counts below K<0.5, then converts it to mu_dark = DCR*(T+t0).
    """
    K = (np.asarray(charges, float) - float(ped_mean)) / max(float(spe_mean), 1.0)
    counts = np.asarray(counts, float)
    if len(K) < 2:
        return 0.1
    dK = float(np.median(np.abs(np.diff(K))))
    valley = (K >= 0.45) & (K <= 0.55)
    below_half = K < 0.5
    if dK <= 0 or not np.any(valley) or not np.any(below_half):
        return 0.1
    NN = float(np.mean(counts[valley]))
    Nc = max(float(np.sum(counts[below_half])), 1.0)
    if NN <= 0:
        return 0.1
    dcr = NN / Nc / max(4.0 * tau_slow * dK, 1e-12)
    dcr *= float(np.exp(min(dcr * tau_slow, 5.0)))
    mu_dark = dcr * (float(T_gate) + float(t0_pre))
    return float(np.clip(mu_dark, 1e-4, 10.0))


def dcr_policy(voltage):
    """Return DCR fitting strategy for a given voltage.

    Returns one of:
      "off"   — DCR must be zero (0PE/1PE overlap; cannot constrain DCR)
      "weak"  — DCR weakly constrained; treat result with caution
      "float" — DCR can be floated freely
    """
    if voltage <= 53.5:
        return "off"
    elif voltage <= 54.0:
        return "weak"
    else:
        return "float"


def _logit(p):
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return log(p / (1.0 - p))


def _clip_theta_to_bounds(fitter, theta):
    theta = np.asarray(theta, dtype=float).copy()
    lo = np.array([b[0] for b in fitter.bounds], dtype=float)
    hi = np.array([b[1] for b in fitter.bounds], dtype=float)
    return np.clip(theta, lo, hi)


def _theta_index(fitter, name):
    try:
        return list(fitter.param_names).index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown parameter {name!r}") from exc


def theta_from_initial_values(fitter, initial_values):
    """Map public physical initial values to the fitter's raw theta vector."""
    if initial_values is None:
        return fitter.init.copy()

    if "theta" in initial_values:
        theta = np.asarray(initial_values["theta"], dtype=float)
        if len(theta) != len(fitter.init):
            raise ValueError(
                f"Initial theta has length {len(theta)}, expected {len(fitter.init)}"
            )
        return _clip_theta_to_bounds(fitter, theta)

    theta = fitter.init.copy()
    ped = initial_values.get("ped", {})
    spe = initial_values.get("spe", {})

    # Raw parameter names accepted directly for expert use.
    for name, value in initial_values.items():
        if name in fitter.param_names and np.isscalar(value):
            theta[_theta_index(fitter, name)] = float(value)

    extra_sl = fitter.layout["extra"]
    spe_sl = fitter.layout["spe"]
    lam_idx = fitter.layout["lam"].start

    if "ped_mean" in ped:
        theta[extra_sl.start] = float(ped["ped_mean"])
    elif "ped_mean" in initial_values:
        theta[extra_sl.start] = float(initial_values["ped_mean"])

    if "ped_sigma" in ped:
        theta[extra_sl.start + 1] = float(ped["ped_sigma"])
    elif "ped_sigma" in initial_values:
        theta[extra_sl.start + 1] = float(initial_values["ped_sigma"])

    spe_mean = spe.get("spe_mean", initial_values.get("spe_mean"))
    spe_sigma = spe.get("spe_sigma", initial_values.get("spe_sigma"))
    if spe_mean is not None and spe_sigma is not None:
        theta[spe_sl.start], theta[spe_sl.start + 1] = reparam_from_spe(
            float(spe_mean), float(spe_sigma)
        )

    xi = spe.get("xi", initial_values.get("xi"))
    if xi is not None:
        xi = float(np.clip(float(xi), 1e-12, 1.0 - 1e-12))
        if "log_xi" in fitter.param_names:
            theta[spe_sl.start + 2] = log(xi)
        else:
            theta[spe_sl.start + 2] = xi

    rho = spe.get("rho", initial_values.get("rho"))
    if rho is not None and "log_rho" in fitter.param_names:
        theta[spe_sl.start + 3] = log(float(rho))

    beta = spe.get("beta", initial_values.get("beta"))
    if beta is not None and "logit_beta" in fitter.param_names:
        theta[spe_sl.start + 4] = _logit(float(beta))

    lam = initial_values.get("lam")
    if lam is not None:
        theta[lam_idx] = float(lam) - fitter.lam_dc

    if "dark" in fitter.layout:
        dark_sl = fitter.layout["dark"]
        dc = initial_values.get("dark_count", {})
        mu_dark = dc.get("mu_dark", initial_values.get("mu_dark"))
        log_mu_dark = dc.get("log_mu_dark", initial_values.get("log_mu_dark"))
        if mu_dark is not None:
            theta[dark_sl.start] = log(float(mu_dark))
        elif log_mu_dark is not None:
            theta[dark_sl.start] = float(log_mu_dark)

    return _clip_theta_to_bounds(fitter, theta)
