#!/usr/bin/env python3
"""Fit one Hamamatsu PCB6 charge spectrum with NegBin AP fitter + optax L-BFGS."""
import argparse
import json
import sys
from math import log

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
import scipy.stats

from fitter.core.base import ParamBlock
from fitter.models.afterpulse import NegBinBetaAPFitter
from fitter.models.gen_tweedie import reparam_from_spe
from fitter.tests.plot import make_figure, n_max, plot_histogram_with_fit


# ─── I/O ─────────────────────────────────────────────────────────────────────


def parse_histogram(path):
    charges, counts = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    charges.append(float(parts[0]))
                    counts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(charges), np.array(counts)


def trim_histogram(charges, counts, pad=20):
    nz = np.where(counts > 0)[0]
    if len(nz) == 0:
        return charges, counts
    lo = max(0, nz[0] - pad)
    hi = min(len(counts) - 1, nz[-1] + pad)
    return charges[lo : hi + 1], counts[lo : hi + 1]


# ─── initialisation ──────────────────────────────────────────────────────────


def estimate_init(charges, counts):
    """Thin wrapper; logic lives in NegBinBetaAPFitter.estimate_from_histogram."""
    d = NegBinBetaAPFitter.estimate_from_histogram(charges, counts)
    return d["ped_mean"], d["ped_sigma"], d["spe_mean"], d["spe_sigma"], d["lam_est"]


def _logit(p):
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-float(x)))


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


def load_initial_values(path):
    """Load optional initial values from JSON.

    Accepted forms:
      * {"theta": [...]} with the full raw optimiser vector.
      * A previous fit result JSON with nested "ped", "spe", and "lam" fields.
      * A flat mapping with raw parameter names or physical names.
    """
    if path is None:
        return None
    with open(path) as fh:
        return json.load(fh)


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

    # Raw parameter names are accepted directly for expert use.
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
        theta[spe_sl.start + 2] = float(xi)

    rho = spe.get("rho", initial_values.get("rho"))
    if rho is not None:
        theta[spe_sl.start + 3] = log(float(rho))

    beta = spe.get("beta", initial_values.get("beta"))
    if beta is not None:
        theta[spe_sl.start + 4] = _logit(float(beta))

    lam = initial_values.get("lam")
    if lam is not None:
        theta[lam_idx] = float(lam) - fitter.lam_dc

    return _clip_theta_to_bounds(fitter, theta)


# ─── fitter construction ──────────────────────────────────────────────────────


def make_fitter(
    charges,
    counts,
    ped_mean,
    ped_sigma,
    spe_mean,
    spe_sigma,
    lam_init=None,
):
    """Build one NegBinBetaAPFitter with physical-only bounds."""
    counts_int = np.round(counts).astype(int)
    A = int(counts_int.sum())

    dq = float(np.median(np.diff(charges)))
    bin_edges = np.append(charges - dq / 2, charges[-1] + dq / 2)
    Q_raw = np.repeat(charges, counts_int)
    q_min = min(float(charges[0]), ped_mean - 10 * ped_sigma)

    a0, b0 = reparam_from_spe(spe_mean, spe_sigma)
    log_rho0 = log(0.01)
    beta_0 = max(1e-4, 0.4 * spe_sigma / ((1.0 - 0.01) * spe_mean))
    logit_beta0 = log(beta_0 / (1.0 - beta_0))

    extra_block = ParamBlock(
        name="pedestal",
        names=["ped_mean", "ped_sigma"],
        init=np.array([ped_mean, ped_sigma], dtype=float),
        bounds=[(-500.0, 500.0), (0.5, 500.0)],
    )
    spe_block = ParamBlock(
        name="spe",
        names=["a_logSigma", "b_logDiff", "xi", "log_rho", "logit_beta"],
        init=np.array([a0, b0, 0.04, log_rho0, logit_beta0], dtype=float),
        bounds=[
            (log(1.0), log(1e4)),  # sigma ∈ (1, 10000) ADC
            (log(1.0), log(1e4)),  # mu-sigma ∈ (1, 10000) ADC
            (1e-4, 1.0 - 1e-4),  # xi ∈ (0,1)
            (-15.0, -0.01),  # rho ∈ (7e-7, 0.99)
            (-15.0, 15.0),  # beta ∈ (3e-7, 1-3e-7)
        ],
    )

    return NegBinBetaAPFitter(
        Q_raw=Q_raw,
        A=A,
        bins=bin_edges,
        q_min=q_min,
        extra_block=extra_block,
        spe_block=spe_block,
        lam_init=lam_init,
        mode="binned",
    )


# ─── optax optimizer ─────────────────────────────────────────────────────────


def fit_optax(
    fitter,
    theta0=None,
    maxiter=2000,
    tol_grad=1e-5,
    tol_nll=1e-8,
    memory_size=10,
    optimizer_name="zoom",
    use_linesearch=None,
):
    """Optimize via optax L-BFGS (CPU, Python loop).

    optimizer_name="zoom"  — optax.lbfgs default strong-Wolfe/zoom line search.
    optimizer_name="fixed" — pure L-BFGS step, no line search.

    Returns (theta, logl, converged, n_iter, trace_logl, trace_gnorm).
    Bounds enforced by projected gradient (clip after apply_updates).
    """
    if use_linesearch is not None:
        optimizer_name = "zoom" if use_linesearch else "fixed"
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    bounds_lo = jnp.array([b[0] for b in fitter.bounds], dtype=jnp.float64)
    bounds_hi = jnp.array([b[1] for b in fitter.bounds], dtype=jnp.float64)

    def neg_logl_fn(t):
        return -fitter._logl_from_theta(t)

    neg_logl_fn_jit = jax.jit(neg_logl_fn)

    if optimizer_name == "zoom":
        optimizer = optax.lbfgs(memory_size=memory_size, scale_init_precond=True)
    elif optimizer_name == "fixed":
        optimizer = optax.lbfgs(
            memory_size=memory_size, scale_init_precond=True, linesearch=None
        )
    else:
        raise ValueError("optimizer_name must be one of 'zoom' or 'fixed'")

    theta = theta0_j
    opt_state = optimizer.init(theta)
    vg_fn = jax.jit(jax.value_and_grad(neg_logl_fn))
    value_and_grad = optax.value_and_grad_from_state(neg_logl_fn_jit)

    def eval_value_and_grad(params, state):
        if optimizer_name == "fixed":
            return vg_fn(params)
        return value_and_grad(params, state=state)

    value, grad = eval_value_and_grad(theta, opt_state)

    trace_logl = []
    trace_gnorm = []
    consec = 0
    converged = False
    n_iter = 0

    for i in range(maxiter):
        logl_i = -float(value)
        gnorm_i = float(jnp.linalg.norm(grad))
        trace_logl.append(logl_i)
        trace_gnorm.append(gnorm_i)

        if not np.isfinite(logl_i) or not np.isfinite(gnorm_i):
            print(
                f"[OPTAX] step {i} produced non-finite value/gradient"
                f"  logl={logl_i}  |g|={gnorm_i}",
                flush=True,
            )
            break

        satisfied = gnorm_i < tol_grad or (
            len(trace_logl) > 1 and abs(logl_i - trace_logl[-2]) < tol_nll
        )
        consec = consec + 1 if satisfied else 0
        if consec >= 3:
            converged = True
            n_iter = i + 1
            break

        try:
            updates, opt_state = optimizer.update(
                grad,
                opt_state,
                theta,
                value=value,
                grad=grad,
                value_fn=neg_logl_fn_jit,
            )
        except Exception as exc:
            print(f"[OPTAX] step {i} optimizer.update failed: {exc}", flush=True)
            break

        theta_new = optax.apply_updates(theta, updates)
        theta_new = jnp.clip(theta_new, bounds_lo, bounds_hi)
        theta = theta_new
        value, grad = eval_value_and_grad(theta, opt_state)
        n_iter = i + 1

        if len(trace_logl) > 0 and -float(value) < trace_logl[-1] - 1e-6:
            print(
                f"[OPTAX] warning: logl decreased at step {i}"
                f" ({-float(value):.6g} < {trace_logl[-1]:.6g})",
                flush=True,
            )

    return (
        np.asarray(theta, dtype=np.float64),
        -float(value),
        converged,
        n_iter,
        np.array(trace_logl),
        np.array(trace_gnorm),
    )


# ─── likelihood-space exploration ────────────────────────────────────────────


def _physical_seed_summary(fitter, theta):
    ly = fitter.layout
    spe = theta[ly["spe"]]
    mean_sigma = np.exp(float(spe[0]))
    mean = mean_sigma + np.exp(float(spe[1]))
    rho = np.exp(float(spe[3]))
    beta = _sigmoid(float(spe[4]))
    return {
        "lam": float(theta[ly["lam"].start] + fitter.lam_dc),
        "xi": float(spe[2]),
        "rho": float(rho),
        "beta": float(beta),
        "ap_mean_fraction": float(beta * (1.0 - rho)),
        "spe_mean": float(mean),
        "spe_sigma": float(mean_sigma),
    }


def _set_seed_params(fitter, theta, lam=None, rho=None, beta=None, xi=None):
    ly = fitter.layout
    out = np.asarray(theta, dtype=float).copy()
    spe_start = ly["spe"].start
    if lam is not None:
        out[ly["lam"].start] = float(lam) - fitter.lam_dc
    if xi is not None:
        out[spe_start + 2] = float(xi)
    if rho is not None:
        out[spe_start + 3] = log(float(rho))
    if beta is not None:
        out[spe_start + 4] = _logit(float(beta))
    return _clip_theta_to_bounds(fitter, out)


def _scout_ranges(fitter, base_theta, lam_est):
    ly = fitter.layout
    spe_start = ly["spe"].start
    lam_idx = ly["lam"].start
    rho_idx = spe_start + 3
    beta_idx = spe_start + 4
    xi_idx = spe_start + 2

    lam0 = float(lam_est) if lam_est is not None and lam_est > 0 else float(base_theta[lam_idx])
    lam0 = max(lam0, 0.02)
    lam_lo = max(fitter.bounds[lam_idx][0] + fitter.lam_dc, lam0 * np.exp(-1.25))
    lam_hi = min(fitter.bounds[lam_idx][1] + fitter.lam_dc, lam0 * np.exp(1.25))

    rho_lo = max(fitter.bounds[rho_idx][0], log(5e-5))
    rho_hi = min(fitter.bounds[rho_idx][1], log(0.25))
    beta_lo = max(fitter.bounds[beta_idx][0], _logit(0.01))
    beta_hi = min(fitter.bounds[beta_idx][1], _logit(0.85))
    xi_lo = max(fitter.bounds[xi_idx][0], 0.005)
    xi_hi = min(fitter.bounds[xi_idx][1], 0.25)

    return {
        "log_lam": (log(lam_lo), log(lam_hi)),
        "log_rho": (rho_lo, rho_hi),
        "logit_beta": (beta_lo, beta_hi),
        "xi": (xi_lo, xi_hi),
    }


def make_seed_candidates(fitter, base_theta, lam_est=None, n_probe=64, seed=12345):
    """Build deterministic scout seeds for degenerate AP/count directions."""
    candidates = [{"label": "base", "theta": _clip_theta_to_bounds(fitter, base_theta)}]
    ranges = _scout_ranges(fitter, base_theta, lam_est)
    lam0 = float(candidates[0]["theta"][fitter.layout["lam"].start] + fitter.lam_dc)
    xi0 = float(candidates[0]["theta"][fitter.layout["spe"].start + 2])

    manual = [
        ("small-ap", lam0, 0.005, 0.05, xi0),
        ("high-rho-low-charge", lam0, 0.12, 0.04, xi0),
        ("low-rho-high-charge", lam0, 0.01, 0.60, xi0),
        ("high-dispersion", lam0, 0.03, 0.20, 0.18),
    ]
    for label, lam, rho, beta, xi in manual:
        candidates.append(
            {"label": label, "theta": _set_seed_params(fitter, base_theta, lam, rho, beta, xi)}
        )

    if n_probe <= 0:
        return candidates

    sampler = scipy.stats.qmc.Sobol(d=4, scramble=True, seed=int(seed))
    n_sobol = 1 << int(np.ceil(np.log2(max(n_probe, 2))))
    u = sampler.random_base2(int(np.log2(n_sobol)))[:n_probe]

    log_lam_lo, log_lam_hi = ranges["log_lam"]
    log_rho_lo, log_rho_hi = ranges["log_rho"]
    beta_lo, beta_hi = ranges["logit_beta"]
    xi_lo, xi_hi = ranges["xi"]

    for i, row in enumerate(u):
        lam = np.exp(log_lam_lo + row[0] * (log_lam_hi - log_lam_lo))
        rho = np.exp(log_rho_lo + row[1] * (log_rho_hi - log_rho_lo))
        beta = _sigmoid(beta_lo + row[2] * (beta_hi - beta_lo))
        xi = xi_lo + row[3] * (xi_hi - xi_lo)
        candidates.append(
            {
                "label": f"sobol-{i:03d}",
                "theta": _set_seed_params(fitter, base_theta, lam, rho, beta, xi),
            }
        )

    return candidates


def rank_seed_candidates(fitter, candidates, n_starts=4):
    """Rank seed candidates by jitted batched likelihood and gradient norm."""
    theta_stack = jnp.asarray([c["theta"] for c in candidates], dtype=jnp.float64)

    def neg_logl_fn(t):
        return -fitter._logl_from_theta(t)

    batched_vg = jax.jit(jax.vmap(jax.value_and_grad(neg_logl_fn)))
    values, grads = batched_vg(theta_stack)
    values = np.asarray(values, dtype=float)
    gnorms = np.linalg.norm(np.asarray(grads, dtype=float), axis=1)

    ranked = []
    for cand, value, gnorm in zip(candidates, values, gnorms):
        report = {
            "label": cand["label"],
            "theta": np.asarray(cand["theta"], dtype=float),
            "start_logl": -float(value),
            "start_gnorm": float(gnorm),
            **_physical_seed_summary(fitter, cand["theta"]),
        }
        if np.isfinite(value) and np.isfinite(gnorm):
            ranked.append(report)

    ranked.sort(key=lambda r: (-r["start_logl"], r["start_gnorm"]))
    return ranked[: max(1, int(n_starts))]


def fit_multistart(
    charges,
    counts,
    ped_mean,
    ped_sigma,
    spe_mean,
    spe_sigma,
    lam_est=None,
    maxiter=2000,
    initial_values=None,
    n_probe=64,
    n_starts=4,
    optimizer_name="zoom",
    scout_seed=12345,
):
    """Single fitter, scout seeds, optimize best-ranked basins."""
    fitter = make_fitter(
        charges,
        counts,
        ped_mean,
        ped_sigma,
        spe_mean,
        spe_sigma,
        lam_init=lam_est,
    )
    base_theta = theta_from_initial_values(fitter, initial_values)

    # warm up JIT once on default init
    _ = fitter._logl_jit(jnp.asarray(base_theta))

    candidates = make_seed_candidates(
        fitter, base_theta, lam_est=lam_est, n_probe=n_probe, seed=scout_seed
    )
    seeds = rank_seed_candidates(fitter, candidates, n_starts=n_starts)
    print(
        f"[SCOUT] evaluated={len(candidates)}  optimizing={len(seeds)}"
        f"  ap_model=beta  optimizer={optimizer_name}",
        flush=True,
    )
    for seed_info in seeds:
        print(
            f"[SCOUT] {seed_info['label']:>18s}"
            f"  logl0={seed_info['start_logl']:.2f}"
            f"  |g|0={seed_info['start_gnorm']:.3g}"
            f"  lam={seed_info['lam']:.3g}"
            f"  rho={seed_info['rho']:.3g}"
            f"  beta={seed_info['beta']:.3g}"
            f"  xi={seed_info['xi']:.3g}",
            flush=True,
        )

    best_logl = -np.inf
    best_result = (
        None  # (fitter, theta, logl, trace_logl, trace_gnorm, converged, n_iter)
    )
    seed_reports = []

    for seed_info in seeds:
        try:
            theta, logl, conv, n_it, tr_l, tr_g = fit_optax(
                fitter,
                theta0=seed_info["theta"],
                maxiter=maxiter,
                optimizer_name=optimizer_name,
            )
            tag = "[new best]" if logl > best_logl else ""
            print(
                f"[FITSEED] {seed_info['label']}"
                f"  logl={logl:.2f}  converged={conv}  {tag}",
                flush=True,
            )
            report = {k: v for k, v in seed_info.items() if k != "theta"}
            report.update(
                {
                    "optimized_logl": float(logl),
                    "optimized_gnorm": float(tr_g[-1]) if len(tr_g) else float("nan"),
                    "converged": bool(conv),
                    "n_iter": int(n_it),
                }
            )
            seed_reports.append(report)
            if logl > best_logl:
                best_logl = logl
                best_result = (fitter, theta, logl, tr_l, tr_g, conv, n_it, seed_reports)
        except Exception as exc:
            print(f"[WARN] seed {seed_info['label']} failed: {exc}", flush=True)
        finally:
            jax.clear_caches()

    if best_result is not None:
        fitter, theta, logl, tr_l, tr_g, conv, n_it, _ = best_result
        best_result = (fitter, theta, logl, tr_l, tr_g, conv, n_it, seed_reports)

    return best_result


# ─── post-fit quantities ──────────────────────────────────────────────────────


def spe_phys(fitter, spe_args, spe_err):
    r = fitter.spe_report(spe_args)
    a, b = float(spe_args[0]), float(spe_args[1])
    da, db = float(spe_err[0]), float(spe_err[1])
    sigma = float(np.exp(a))
    r["spe_sigma_err"] = sigma * da
    r["spe_mean_err"] = float(np.sqrt((sigma * da) ** 2 + (float(np.exp(b)) * db) ** 2))
    rho = r["rho"]
    beta = r["beta"]
    drho = rho * float(spe_err[3])
    dbeta = beta * (1.0 - beta) * float(spe_err[4])
    r["rho_err"] = drho
    r["beta_err"] = dbeta
    ap_charge_mean_err = r["ap_charge_mean"] * float(
        np.sqrt(
            (drho / (rho + 1e-30)) ** 2
            + (dbeta / (beta + 1e-30)) ** 2
            + (r["spe_mean_err"] / (r["spe_mean"] + 1e-30)) ** 2
        )
    )
    r["ap_charge_mean_err"] = ap_charge_mean_err
    if "beta_shape_b" in r:
        mean_fraction = beta * (1.0 - rho)
        dmean_fraction = float(
            np.sqrt(((1.0 - rho) * dbeta) ** 2 + (beta * drho) ** 2)
        )
        r["beta_shape_b_err"] = float(
            2.0 * dmean_fraction / (mean_fraction + 1e-30) ** 2
        )
    r["alpha_err"] = r["alpha"] * float(
        np.sqrt((drho / (rho + 1e-30)) ** 2 + (dbeta / (beta + 1e-30)) ** 2)
    )
    spe_mean = r["spe_mean"]
    spe_sigma = r["spe_sigma"]
    spe_res = r["spe_res"]
    r["spe_res_err"] = spe_res * float(
        np.sqrt(
            (r["spe_sigma_err"] / (spe_sigma + 1e-30)) ** 2
            + (r["spe_mean_err"] / (spe_mean + 1e-30)) ** 2
        )
    )
    return r


def compute_chi2(fitter, theta):
    """Neyman-B chi-squared; threshold exp >= 1 to avoid tail blow-up."""
    obs = fitter.hist.astype(float)
    exp = fitter.estimate_bin_counts(theta)
    A = float(np.exp(float(theta[fitter.layout["log_A"].start])))
    z_est = max(A - float(np.sum(exp)), 0.0)
    zero = int(fitter.grid.zero)

    mask = exp >= 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mask, (obs - exp) ** 2 / exp, 0.0)
    chi2 = float(np.sum(terms))
    if z_est > 1.0:
        chi2 += float((z_est - zero) ** 2 / z_est)

    n_bins_used = int(mask.sum()) + (1 if z_est > 1.0 else 0)
    ndf = n_bins_used - len(theta)
    p_val = float(scipy.stats.chi2.sf(chi2, ndf)) if ndf > 0 else float("nan")
    return chi2, ndf, p_val


def compute_cov(fitter, theta):
    try:
        H = np.asarray(jax.hessian(fitter._logl_from_theta)(jnp.asarray(theta)))
        cov = -np.linalg.inv(H)
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(std, std)
        return cov, corr
    except Exception as exc:
        print(f"[WARN] Hessian failed: {exc}", flush=True)
        n = len(theta)
        return np.full((n, n), float("nan")), np.full((n, n), float("nan"))


# ─── likelihood scan ─────────────────────────────────────────────────────────


def scan_logl(fitter, theta, theta_err, param_idx, n_points=61, sigma_range=4.0):
    """Scan logl vs one parameter (others held at MLE)."""
    center = float(theta[param_idx])
    err = (
        float(theta_err[param_idx])
        if np.isfinite(theta_err[param_idx])
        else abs(center) * 0.2
    )
    if err <= 0 or not np.isfinite(err):
        err = abs(center) * 0.2 or 0.5
    lo = center - sigma_range * err
    hi = center + sigma_range * err
    # respect fitter bounds
    blo, bhi = fitter.bounds[param_idx]
    lo = max(lo, blo)
    hi = min(hi, bhi)
    scan_vals = np.linspace(lo, hi, n_points)
    scan_logls = []
    for v in scan_vals:
        t = theta.copy()
        t[param_idx] = v
        scan_logls.append(fitter.logl(t))
    return scan_vals, np.array(scan_logls)


def profile_scan_spec(fitter, name):
    ly = fitter.layout
    spe_start = ly["spe"].start
    specs = {
        "lam": (ly["lam"].start, lambda x: x + fitter.lam_dc, "lambda"),
        "xi": (spe_start + 2, lambda x: x, "xi"),
        "rho": (spe_start + 3, np.exp, "rho"),
        "beta": (spe_start + 4, lambda x: 1.0 / (1.0 + np.exp(-x)), "beta"),
    }
    if name in specs:
        return specs[name]
    if name in fitter.param_names:
        idx = _theta_index(fitter, name)
        return idx, lambda x: x, name
    raise ValueError(f"Unknown profile scan parameter {name!r}")


def make_profile_scans(fitter, theta, theta_err, names, n_points=61):
    scans = {}
    for name in names:
        if not name:
            continue
        idx, transform, label = profile_scan_spec(fitter, name)
        try:
            scan_vals, scan_logls = scan_logl(
                fitter, theta, theta_err, idx, n_points=n_points
            )
            logl_max = float(np.max(scan_logls))
            x_vals = np.asarray([transform(v) for v in scan_vals], dtype=float)
            scans[name] = {
                "label": label,
                "param_index": int(idx),
                "raw": [float(x) for x in scan_vals],
                "x": [float(x) for x in x_vals],
                "logl": [float(x) for x in scan_logls],
                "delta_logl": [float(x - logl_max) for x in scan_logls],
            }
        except Exception as exc:
            print(f"[WARN] {name} scan failed: {exc}", flush=True)
    return scans


# ─── per-voltage fit plot ─────────────────────────────────────────────────────


def _finite_or_zero(v):
    f = float(v)
    return f if np.isfinite(f) else 0.0


def save_fit_plot(fitter, theta, theta_err, rec, out_path, trace_logl, trace_gnorm):
    """PDF with spectrum fit, optimizer trace, and requested profile scans."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ly = fitter.layout
    lam = float(theta[ly["lam"].start]) + fitter.lam_dc
    lam_err_raw = float(theta_err[ly["lam"].start])

    n_pe = n_max(lam)
    hist = fitter.hist.astype(float)
    bins = fitter.bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = float(bins[1] - bins[0])

    xsp = fitter.grid.xsp
    smooth = fitter.estimate_density(theta) * bin_width
    ys = fitter.estimate_bin_counts(theta)
    comps = [fitter.estimate_component_counts(theta, k) for k in range(1, n_pe + 1)]
    labels = [f"{k} PE" for k in range(1, n_pe + 1)]

    spe = rec["spe"]
    occ = float(np.clip(1.0 - np.exp(-lam), 0.0, 1.0 - 1e-9))
    occ_std = float(lam_err_raw * np.exp(-lam)) if np.isfinite(lam_err_raw) else 0.0
    spe_res_pct = spe["spe_res"] * 100.0
    spe_params = np.array(theta[ly["spe"]])

    # gm_std: pass None to suppress ±term (gm_std=None is checked in plot fn)
    # spe_sigma_std/spe_res_std: pass 0.0 for NaN (plot fn always formats them)
    gm_std = _finite_or_zero(spe.get("spe_mean_err", float("nan"))) or None
    spe_sigma_std = _finite_or_zero(spe.get("spe_sigma_err", float("nan")))
    spe_res_std = _finite_or_zero(spe.get("spe_res_err", 0.0) * 100.0)

    with PdfPages(out_path) as pp:
        # ── page 1: charge spectrum ──────────────────────────────────────────
        fig, ax_main, ax_resid, ax_leg = make_figure(n_comps=len(comps))
        # draw with ys so plot_histogram_with_fit sets up axes (xlabel, label_outer)
        plot_histogram_with_fit(
            bins=bins,
            hist=hist,
            xsp=xsp,
            smooth=smooth,
            bin_centers=bin_centers,
            comps=comps,
            labels=labels,
            params=spe_params,
            occ=occ,
            occ_std=occ_std,
            ped_mean=rec["ped"]["ped_mean"],
            gm=spe["spe_mean"],
            gm_std=gm_std,
            spe_sigma=spe["spe_sigma"],
            spe_sigma_std=spe_sigma_std,
            spe_res=spe_res_pct,
            spe_res_std=spe_res_std,
            chiSq=rec["chi_sq"],
            ndf=rec["ndf"],
            ys=ys,
            logscale=True,
            ax_main=ax_main,
            ax_resid=ax_resid,
            ax_leg=ax_leg,
            fig=fig,
        )
        ax_main.set_title(f"Hamamatsu PCB6  {rec['voltage']} V")
        ax_main.set_ylim(bottom=0.5)

        # replace raw residuals with Neyman pull = (data-model)/sqrt(max(model,1))
        pull = (hist - ys) / np.sqrt(np.maximum(ys, 1.0))
        ax_resid.cla()
        ax_resid.axhline(0, color="gray", lw=1, ls="--")
        ax_resid.axhline(+1, color="C0", lw=0.6, ls=":")
        ax_resid.axhline(-1, color="C0", lw=0.6, ls=":")
        ax_resid.axhline(+3, color="C1", lw=0.6, ls=":")
        ax_resid.axhline(-3, color="C1", lw=0.6, ls=":")
        ax_resid.plot(bin_centers, pull, "o", color="black", ms=2)
        ax_resid.set_ylabel("Pull")
        ax_resid.set_xlabel("Q")
        ax_resid.grid(True, alpha=0.3)
        pp.savefig(fig)
        plt.close(fig)

        # ── page 2: optimizer trace ──────────────────────────────────────────
        if len(trace_logl) > 1:
            fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            axes[0].plot(trace_logl, lw=1)
            axes[0].set_ylabel("log L")
            axes[0].set_title(f"Optimizer trace — {rec['voltage']} V")
            axes[0].grid(True, alpha=0.3)
            axes[1].semilogy(trace_gnorm, lw=1, color="C1")
            axes[1].set_ylabel("|∇|")
            axes[1].set_xlabel("Iteration")
            axes[1].grid(True, alpha=0.3)
            pp.savefig(fig)
            plt.close(fig)

        # ── profile scans for key degeneracy directions ─────────────────────
        for name, scan in rec.get("profile_scans", {}).items():
            fig, ax = plt.subplots()
            ax.plot(scan["x"], scan["delta_logl"], lw=1.5)
            ax.axhline(-0.5, color="C0", ls="--", lw=0.8, label="±1σ")
            ax.axhline(-2.0, color="C1", ls="--", lw=0.8, label="±2σ")
            ax.set_xlabel(scan.get("label", name))
            ax.set_ylabel("Δlog L")
            ax.set_title(f"{name} profile scan — {rec['voltage']} V")
            ax.legend()
            ax.grid(True, alpha=0.3)
            pp.savefig(fig)
            plt.close(fig)

    print(f"[PLOT] {out_path}", flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="PeakOTron histogram txt file")
    parser.add_argument("--voltage", type=float, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--out-fig", default=None, help="Per-voltage fit plot PDF")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument(
        "--init-json",
        default=None,
        help="Optional JSON with theta or physical ped/spe/lam initial values",
    )
    parser.add_argument(
        "--explore-seeds",
        type=int,
        default=64,
        help="Number of Sobol scout seeds before optimization",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=4,
        help="Number of gradient-ranked scout seeds to optimize",
    )
    parser.add_argument(
        "--scout-seed",
        type=int,
        default=12345,
        help="Deterministic seed for scrambled Sobol scout points",
    )
    parser.add_argument(
        "--optimizer",
        choices=["zoom", "fixed"],
        default="zoom",
        help="Optax L-BFGS variant; zoom is the default strong-Wolfe line search",
    )
    parser.add_argument(
        "--profile-scans",
        default="lam,rho,beta,xi",
        help="Comma-separated 1D profile scans to save; use 'none' to disable",
    )
    parser.add_argument(
        "--no-linesearch",
        action="store_true",
        help="Alias for --optimizer fixed",
    )
    args = parser.parse_args()
    if args.no_linesearch:
        args.optimizer = "fixed"

    charges, counts = parse_histogram(args.input)

    if len(counts) == 0 or counts.sum() == 0:
        print(f"[SKIP] {args.input} is empty", flush=True)
        with open(args.output, "w") as f:
            json.dump({"voltage": args.voltage, "empty": True, "converged": False}, f)
        sys.exit(0)

    charges, counts = trim_histogram(charges, counts)

    ped_mean, ped_sigma, spe_mean, spe_sigma, lam_est = estimate_init(charges, counts)
    print(
        f"[INIT] V={args.voltage}V  ped_mean={ped_mean:.2f}  ped_sigma={ped_sigma:.2f}"
        f"  spe_mean={spe_mean:.2f}  spe_sigma={spe_sigma:.2f}  lam_est={lam_est:.3f}",
        flush=True,
    )
    initial_values = load_initial_values(args.init_json)
    if initial_values is not None:
        print(f"[INIT] loaded overrides from {args.init_json}", flush=True)

    result = fit_multistart(
        charges,
        counts,
        ped_mean,
        ped_sigma,
        spe_mean,
        spe_sigma,
        lam_est=lam_est,
        maxiter=args.maxiter,
        initial_values=initial_values,
        n_probe=args.explore_seeds,
        n_starts=args.starts,
        optimizer_name=args.optimizer,
        scout_seed=args.scout_seed,
    )

    if result is None:
        print(f"[FAIL] all starts failed for V={args.voltage}V", flush=True)
        with open(args.output, "w") as f:
            json.dump({"voltage": args.voltage, "empty": False, "converged": False}, f)
        sys.exit(1)

    fitter, theta, logl, trace_logl, trace_gnorm, converged, n_iter, seed_reports = result

    # free accumulated LLVM allocations from multi-start before post-fit JAX calls
    jax.clear_caches()

    print(
        f"[FIT ] V={args.voltage}V  converged={converged}"
        f"  logl={logl:.2f}  n_iter={n_iter}",
        flush=True,
    )

    ly = fitter.layout
    spe_args = theta[ly["spe"]]
    extra_args = theta[ly["extra"]]
    lam = float(theta[ly["lam"].start]) + fitter.lam_dc

    cov, corr = compute_cov(fitter, theta)
    theta_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    theta_err = np.where(np.diag(cov) > 0, theta_err, np.full_like(theta, np.nan))

    spe_err = theta_err[ly["spe"]]
    extra_err = theta_err[ly["extra"]]
    lam_err = float(theta_err[ly["lam"].start])

    phys = spe_phys(fitter, spe_args, spe_err)
    chi2, ndf, p_val = compute_chi2(fitter, theta)
    if args.profile_scans.lower() == "none":
        profile_scans = {}
    else:
        profile_names = [x.strip() for x in args.profile_scans.split(",") if x.strip()]
        profile_scans = make_profile_scans(fitter, theta, theta_err, profile_names)

    print(
        f"[CHI2] V={args.voltage}V  chi2={chi2:.1f}  ndf={ndf}  p={p_val:.3f}",
        flush=True,
    )

    output = {
        "voltage": args.voltage,
        "ap_model": "beta",
        "optimizer": args.optimizer,
        "empty": False,
        "converged": bool(converged),
        "logl": float(logl),
        "n_iter": int(n_iter),
        "n_events": int(np.round(counts).astype(int).sum()),
        "param_names": list(fitter.param_names),
        "theta": [float(x) for x in theta],
        "theta_err": [float(x) for x in theta_err],
        "cov": [[float(x) for x in row] for row in cov],
        "corr": [[float(x) for x in row] for row in corr],
        "ped": {
            "ped_mean": float(extra_args[0]),
            "ped_mean_err": float(extra_err[0]),
            "ped_sigma": float(extra_args[1]),
            "ped_sigma_err": float(extra_err[1]),
        },
        "spe": {k: float(v) for k, v in phys.items()},
        "lam": lam,
        "lam_err": lam_err,
        "chi_sq": float(chi2),
        "ndf": int(ndf),
        "p_value": float(p_val),
        "init_estimate": {
            "ped_mean": float(ped_mean),
            "ped_sigma": float(ped_sigma),
            "spe_mean": float(spe_mean),
            "spe_sigma": float(spe_sigma),
            "lam_est": float(lam_est),
        },
        "seed_scout": seed_reports,
        "profile_scans": profile_scans,
        "hist_q": [float(x) for x in charges],
        "hist_counts": [float(x) for x in counts],
        "hist_bin_edges": [float(x) for x in fitter.bins],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVE] {args.output}", flush=True)

    if args.out_fig:
        try:
            save_fit_plot(
                fitter, theta, theta_err, output, args.out_fig, trace_logl, trace_gnorm
            )
        except Exception as exc:
            import traceback

            print(f"[WARN] plot failed: {exc}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
