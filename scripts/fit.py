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
from fitter.models.dark_count import make_dark_ft, make_dark_block
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


def estimate_dark_mu(charges, counts, ped_mean, spe_mean, T_gate, t0_pre, tau_slow):
    """Rough mu_dark estimate from 0PE-1PE valley density (PeakOTron-style init).

    At K=0.5 the deterministic dark-pulse density is f_d^(1)(0.5) = 4*tau/(T+t0),
    so mu_dark = (dN_dark/dK * (T+t0)) / (4*tau*N_total).
    """
    K = (np.asarray(charges, float) - float(ped_mean)) / max(float(spe_mean), 1.0)
    mask = (K >= 0.45) & (K <= 0.55)
    n_valley = float(counts[mask].sum()) if mask.sum() > 0 else 0.0
    n_total = float(counts.sum())
    if n_total <= 0 or n_valley <= 0 or mask.sum() < 2:
        return 0.1
    # dN/dK at K=0.5 (counts per unit K per total event)
    dK_bin = 0.1 / float(mask.sum())
    dN_dK = (n_valley / n_total) / dK_bin
    # Invert f_d^(1)(K=0.5) = 4*tau/(T+t0)
    denominator = 4.0 * max(tau_slow, 1e-12) * max(n_total, 1.0)
    mu_dark = dN_dK * (T_gate + t0_pre) / max(4.0 * tau_slow, 1e-12)
    # PeakOTron self-consistency correction (small for typical mu_dark)
    mu_dark *= float(np.exp(min(mu_dark * tau_slow / max(T_gate + t0_pre, 1.0), 5.0)))
    return float(np.clip(mu_dark, 1e-4, 10.0))


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
      * A previous fit result JSON with nested "ped", "spe", "lam", "dark_count" fields.
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

    # Dark count initial value
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


# ─── DCR voltage policy ───────────────────────────────────────────────────────


def dcr_policy(voltage):
    """Return DCR fitting strategy for a given overvoltage.

    Returns one of:
      "off"   — DCR must be zero (0PE/1PE overlap; cannot constrain DCR)
      "weak"  — DCR weakly constrained; treat result with caution
      "float" — DCR can be floated freely
    """
    if voltage <= 53.0:
        return "off"
    elif voltage <= 54.0:
        return "weak"
    else:
        return "float"


# ─── fitter construction ──────────────────────────────────────────────────────


def make_fitter(
    charges,
    counts,
    ped_mean,
    ped_sigma,
    spe_mean,
    spe_sigma,
    lam_init=None,
    dark_mu_init=None,
    dark_T_gate=200.0,
    dark_t0_pre=100.0,
    dark_tau_slow=100.0,
):
    """Build one NegBinBetaAPFitter with optional dark-count block."""
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
            (log(1.0), log(1e4)),   # sigma ∈ (1, 10000) ADC
            (log(1.0), log(1e4)),   # mu-sigma ∈ (1, 10000) ADC
            (1e-4, 1.0 - 1e-4),    # xi ∈ (0,1)
            (-15.0, -0.01),         # rho ∈ (7e-7, 0.99)
            (-15.0, 15.0),          # beta ∈ (3e-7, 1-3e-7)
        ],
    )

    dark_block = None
    dark_ft_fn = None
    if dark_mu_init is not None:
        dark_ft_fn = make_dark_ft(dark_T_gate, dark_t0_pre, dark_tau_slow)
        dark_block = make_dark_block(mu_dark_init=float(dark_mu_init))

    return NegBinBetaAPFitter(
        Q_raw=Q_raw,
        A=A,
        bins=bin_edges,
        q_min=q_min,
        extra_block=extra_block,
        spe_block=spe_block,
        dark_block=dark_block,
        dark_ft=dark_ft_fn,
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
):
    """Optimize via optax L-BFGS + Armijo backtracking linesearch (CPU, Python loop).

    Returns (theta, logl, converged, n_iter, trace_logl, trace_gnorm).
    Bounds enforced by projected gradient (clip after apply_updates).
    """
    if theta0 is None:
        theta0 = fitter.init.copy()
    theta0_j = jnp.asarray(theta0, dtype=jnp.float64)
    bounds_lo = jnp.array([b[0] for b in fitter.bounds], dtype=jnp.float64)
    bounds_hi = jnp.array([b[1] for b in fitter.bounds], dtype=jnp.float64)

    def neg_logl_fn(t):
        return -fitter._logl_from_theta(t)

    neg_logl_fn_jit = jax.jit(neg_logl_fn)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1e6),
        optax.scale_by_lbfgs(memory_size=memory_size, scale_init_precond=True),
        optax.scale(-1.0),
        optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30,
            slope_rtol=1e-4,
            decrease_factor=0.5,
            increase_factor=1.5,
        ),
    )

    theta = theta0_j
    opt_state = optimizer.init(theta)
    vg_fn = jax.jit(jax.value_and_grad(neg_logl_fn))
    value, grad = vg_fn(theta)

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
                f"[OPTAX] step {i} non-finite  logl={logl_i}  |g|={gnorm_i}",
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
        value, grad = vg_fn(theta)
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
        dmean_fraction = float(np.sqrt(((1.0 - rho) * dbeta) ** 2 + (beta * drho) ** 2))
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


def dark_phys(fitter, theta, theta_err):
    """Extract dark-count physical quantities from fit result.

    Returns a dict with mu_dark and its uncertainty; returns None when
    no dark block is present.
    """
    if "dark" not in fitter.layout:
        return None
    dark_sl = fitter.layout["dark"]
    log_mu = float(theta[dark_sl.start])
    log_mu_err = float(theta_err[dark_sl.start])
    mu_dark = float(np.exp(log_mu))
    mu_dark_err = float(mu_dark * log_mu_err) if np.isfinite(log_mu_err) else float("nan")
    return {
        "enabled": True,
        "mu_dark": mu_dark,
        "mu_dark_err": mu_dark_err,
        "log_mu_dark": log_mu,
        "log_mu_dark_err": log_mu_err,
    }


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


# ─── per-voltage fit plot ─────────────────────────────────────────────────────


def _finite_or_zero(v):
    f = float(v)
    return f if np.isfinite(f) else 0.0


def save_fit_plot(fitter, theta, theta_err, rec, out_path, trace_logl, trace_gnorm):
    """PDF with spectrum fit and optimizer trace."""
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

    gm_std = _finite_or_zero(spe.get("spe_mean_err", float("nan"))) or None
    spe_sigma_std = _finite_or_zero(spe.get("spe_sigma_err", float("nan")))
    spe_res_std = _finite_or_zero(spe.get("spe_res_err", 0.0) * 100.0)

    # Build title with optional DCR annotation
    dcr_info = rec.get("dark_count", {})
    if dcr_info.get("enabled"):
        mu_str = f"  μ_dark={dcr_info['mu_dark']:.3g}"
    else:
        mu_str = ""

    with PdfPages(out_path) as pp:
        # ── page 1: charge spectrum ──────────────────────────────────────────
        fig, ax_main, ax_resid, ax_leg = make_figure(n_comps=len(comps))
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
        ax_main.set_title(f"Hamamatsu PCB6  {rec['voltage']} V{mu_str}")
        ax_main.set_ylim(bottom=0.5)

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
        help="Optional JSON with theta or physical ped/spe/lam/dark initial values",
    )

    # Dark count arguments
    dcr_group = parser.add_argument_group("dark count")
    dcr_group.add_argument(
        "--dcr",
        type=float,
        default=None,
        metavar="MU",
        help="Enable dark-count term with initial mu_dark=MU (mean dark pulses per window)."
             " If omitted, DCR is disabled regardless of voltage.",
    )
    dcr_group.add_argument(
        "--dcr-fixed",
        action="store_true",
        help="Fix mu_dark at --dcr value (not fitted).",
    )
    dcr_group.add_argument(
        "--dcr-auto",
        action="store_true",
        help="Auto-estimate mu_dark init from valley density; enables DCR for V>=54.5.",
    )
    dcr_group.add_argument(
        "--gate-T", type=float, default=200.0, metavar="NS",
        help="Gate length in ns (default: 200)",
    )
    dcr_group.add_argument(
        "--gate-t0", type=float, default=100.0, metavar="NS",
        help="Pre-gate window in ns (default: 100)",
    )
    dcr_group.add_argument(
        "--tau-slow", type=float, default=100.0, metavar="NS",
        help="Slow-pulse time constant in ns (default: 100)",
    )
    args = parser.parse_args()

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

    # ─── DCR policy ───────────────────────────────────────────────────────────
    policy = dcr_policy(args.voltage)
    dcr_mu_init = None

    if args.dcr is not None or args.dcr_auto:
        if policy == "off":
            print(
                f"[DCR ] V={args.voltage}V: DCR forced off — 0PE/1PE overlap,"
                f" cannot constrain dark counts",
                flush=True,
            )
        else:
            if policy == "weak":
                print(
                    f"[DCR ] V={args.voltage}V: DCR weakly constrained —"
                    f" treat uncertainty with caution",
                    flush=True,
                )
            if args.dcr_auto and args.dcr is None:
                dcr_mu_init = estimate_dark_mu(
                    charges, counts, ped_mean, spe_mean,
                    args.gate_T, args.gate_t0, args.tau_slow,
                )
                print(f"[DCR ] auto-estimated mu_dark={dcr_mu_init:.4g}", flush=True)
            else:
                dcr_mu_init = args.dcr
            print(
                f"[DCR ] mu_dark_init={dcr_mu_init:.4g}"
                f"  fixed={args.dcr_fixed}"
                f"  T={args.gate_T}ns  t0={args.gate_t0}ns  tau={args.tau_slow}ns",
                flush=True,
            )

    initial_values = load_initial_values(args.init_json)
    if initial_values is not None:
        print(f"[INIT] loaded overrides from {args.init_json}", flush=True)

    fitter = make_fitter(
        charges,
        counts,
        ped_mean,
        ped_sigma,
        spe_mean,
        spe_sigma,
        lam_init=lam_est,
        dark_mu_init=dcr_mu_init,
        dark_T_gate=args.gate_T,
        dark_t0_pre=args.gate_t0,
        dark_tau_slow=args.tau_slow,
    )

    # When --dcr-fixed, freeze log_mu_dark by tightening bounds to a single point
    if args.dcr_fixed and dcr_mu_init is not None and "dark" in fitter.layout:
        dark_idx = fitter.layout["dark"].start
        log_mu = log(float(dcr_mu_init))
        fitter.bounds[dark_idx] = (log_mu, log_mu + 1e-10)
        print(f"[DCR ] log_mu_dark fixed at {log_mu:.4g}", flush=True)

    theta0 = theta_from_initial_values(fitter, initial_values)

    # warm up JIT
    _ = fitter._logl_jit(jnp.asarray(theta0, dtype=jnp.float64))

    print(
        f"[FIT ] V={args.voltage}V  n_params={len(theta0)}"
        f"  dcr={'on' if dcr_mu_init is not None else 'off'}",
        flush=True,
    )

    try:
        theta, logl, converged, n_iter, trace_logl, trace_gnorm = fit_optax(
            fitter, theta0=theta0, maxiter=args.maxiter
        )
    except Exception as exc:
        print(f"[FAIL] optimization failed for V={args.voltage}V: {exc}", flush=True)
        with open(args.output, "w") as f:
            json.dump({"voltage": args.voltage, "empty": False, "converged": False}, f)
        sys.exit(1)

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
    dc_phys = dark_phys(fitter, theta, theta_err)
    chi2, ndf, p_val = compute_chi2(fitter, theta)

    print(
        f"[CHI2] V={args.voltage}V  chi2={chi2:.1f}  ndf={ndf}  p={p_val:.3f}",
        flush=True,
    )
    if dc_phys is not None:
        print(
            f"[DCR ] V={args.voltage}V  mu_dark={dc_phys['mu_dark']:.4g}"
            f"  ±{dc_phys['mu_dark_err']:.2g}",
            flush=True,
        )

    output = {
        "voltage": args.voltage,
        "ap_model": "beta",
        "optimizer": "backtracking-lbfgs",
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
        "dark_count": dc_phys if dc_phys is not None else {"enabled": False},
        "dark_model": {
            "T_gate": args.gate_T,
            "t0_pre": args.gate_t0,
            "tau_slow": args.tau_slow,
            "policy": policy,
        } if dcr_mu_init is not None else None,
        "init_estimate": {
            "ped_mean": float(ped_mean),
            "ped_sigma": float(ped_sigma),
            "spe_mean": float(spe_mean),
            "spe_sigma": float(spe_sigma),
            "lam_est": float(lam_est),
        },
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
