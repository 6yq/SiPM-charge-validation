#!/usr/bin/env python3
"""Likelihood slice scans for a completed fit result.

For each parameter, fix all others at MLE and evaluate logl on a grid.
Plots Δ(2 ln L) = 2*(logl_MLE - logl_slice) vs. the physical parameter value;
the 1σ contour is where Δ(2 ln L) = 1.
"""
import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_build import make_fitter
from fit_defaults import (
    PEAKOTRON_TAU_SLOW_NS,
    PEAKOTRON_T0_PRE_NS,
    PEAKOTRON_T_GATE_NS,
)


# ─── physical transforms ──────────────────────────────────────────────────────


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _raw_to_phys(name, raw_vals, theta, fitter):
    """Map raw theta values for `name` to physical units for axis labelling."""
    if name == "lam":
        return raw_vals
    if name == "log_rho":
        return np.exp(raw_vals)
    if name == "log_xi":
        return np.exp(raw_vals)
    if name == "logit_beta":
        return _sigmoid(raw_vals)
    if name == "log_mu_dark":
        return np.exp(raw_vals)
    if name == "b_logDiff":
        # spe_mean = exp(a_logSigma) + exp(b_logDiff)
        idx_a = list(fitter.param_names).index("a_logSigma")
        return np.exp(float(theta[idx_a])) + np.exp(raw_vals)
    if name == "a_logSigma":
        return np.exp(raw_vals)
    return raw_vals  # fallback: raw == physical


# ─── parameters to scan and their axis labels ────────────────────────────────

_SCAN_DEFS = [
    # (theta param name, physical x-axis label)
    ("lam", r"$\lambda$"),
    ("b_logDiff", r"$G^*$"),
    ("log_xi", r"$\xi$"),
    ("log_rho", r"$\rho$"),
    ("logit_beta", r"$\beta$"),
    ("log_mu_dark", r"$\mu_\mathrm{dark}$"),
]


# ─── core scan ────────────────────────────────────────────────────────────────


def _scan_limits(fitter, theta, theta_err, idx, param_name, n_sigma):
    center = float(theta[idx])
    err = (
        float(theta_err[idx])
        if np.isfinite(theta_err[idx]) and theta_err[idx] > 0
        else None
    )
    lo_b, hi_b = fitter.bounds[idx]

    if err is not None:
        lo = max(center - n_sigma * err, lo_b)
        hi = min(center + n_sigma * err, hi_b)
    else:
        lo, hi = lo_b, hi_b

    if param_name == "b_logDiff":
        # Optimizer bounds are deliberately broad; profile plots should stay local.
        # Scan G* in a ±50% physical window around the MLE, then convert back to raw b.
        names = list(fitter.param_names)
        idx_a = names.index("a_logSigma")
        sigma = float(np.exp(theta[idx_a]))
        diff = float(np.exp(center))
        g_mle = sigma + diff
        g_lo = max(0.5 * g_mle, sigma + float(np.exp(lo_b)))
        g_hi = min(1.5 * g_mle, sigma + float(np.exp(hi_b)))
        if g_hi > g_lo and g_lo > sigma:
            lo = max(lo, float(np.log(g_lo - sigma)))
            hi = min(hi, float(np.log(g_hi - sigma)))

    return lo, hi


def scan_param(fitter, theta, theta_err, param_name, n_sigma=5.0, n_points=101):
    """Evaluate logl on a 1-D grid with one parameter varied.

    Returns (raw_grid, logl_vals), or (None, None) if param absent.
    """
    names = list(fitter.param_names)
    if param_name not in names:
        return None, None
    idx = names.index(param_name)
    theta_j = jnp.asarray(theta, dtype=jnp.float64)

    lo, hi = _scan_limits(fitter, theta, theta_err, idx, param_name, n_sigma)

    if hi - lo < 1e-10:
        return None, None

    raw_grid = jnp.linspace(lo, hi, n_points, dtype=jnp.float64)

    @jax.jit
    def _do_scan(vals):
        return jax.vmap(lambda v: fitter._logl_from_theta(theta_j.at[idx].set(v)))(vals)

    logl_vals = np.asarray(_do_scan(raw_grid))
    return np.asarray(raw_grid), logl_vals


# ─── plot ─────────────────────────────────────────────────────────────────────


def _find_crossings(x, y, level=1.0):
    """Linear interpolation for crossings of y = level."""
    crossings = []
    for i in range(len(y) - 1):
        if (y[i] - level) * (y[i + 1] - level) < 0:
            t = (level - y[i]) / (y[i + 1] - y[i])
            crossings.append(float(x[i] + t * (x[i + 1] - x[i])))
    return crossings


def _make_scan_figure(fitter, theta, theta_err, rec, n_sigma=5.0, n_points=101):
    """Build and return the scan figure (caller owns plt.close)."""
    import matplotlib.pyplot as plt

    names = list(fitter.param_names)
    dc_enabled = rec.get("dark_count", {}).get("enabled", False)

    to_scan = [
        (p, lbl)
        for p, lbl in _SCAN_DEFS
        if p in names and (p != "log_mu_dark" or dc_enabled)
    ]
    if not to_scan:
        return None

    ncols = 2
    nrows = (len(to_scan) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    logl_mle = float(fitter._logl_jit(jnp.asarray(theta, dtype=jnp.float64)))

    for k, (param_name, phys_label) in enumerate(to_scan):
        ax = axes_flat[k]
        print(f"[SCAN] scanning {param_name}...", flush=True)

        raw_grid, logl_vals = scan_param(
            fitter,
            theta,
            theta_err,
            param_name,
            n_sigma=n_sigma,
            n_points=n_points,
        )
        if raw_grid is None:
            ax.set_visible(False)
            continue

        x_phys = _raw_to_phys(param_name, raw_grid, theta, fitter)
        delta = np.clip(2.0 * (logl_mle - logl_vals), 0.0, None)
        crossings = _find_crossings(x_phys, delta, level=1.0)

        ax.fill_between(
            x_phys,
            0,
            delta,
            where=(delta <= 1.0),
            alpha=0.15,
            color="C0",
            label=r"$1\sigma$ band",
        )
        ax.plot(x_phys, delta, color="C0", lw=1.5)
        ax.axhline(1.0, color="C1", lw=1.0, ls="--", label=r"$\Delta(2\ln L)=1$")

        mle_phys = float(
            _raw_to_phys(
                param_name,
                np.array([float(theta[names.index(param_name)])]),
                theta,
                fitter,
            )[0]
        )
        ax.axvline(mle_phys, color="gray", lw=1.0, ls=":")

        for xc in crossings:
            ax.axvline(xc, color="C1", lw=0.8, ls="--", alpha=0.8)

        ax.set_xlabel(phys_label)
        ax.set_ylabel(r"$\Delta(2\ln L)$")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, frameon=False)

    for ax in axes_flat[len(to_scan) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Likelihood scans — {rec['voltage']} V"
        f"  (converged={rec.get('converged')},"
        f" logl={rec.get('logl', float('nan')):.1f})",
        fontsize=18,
    )
    return fig


def append_to_pdf(pp, fitter, theta, theta_err, rec, n_sigma=5.0, n_points=101):
    """Append likelihood scan page(s) to an already-open PdfPages object."""
    import matplotlib.pyplot as plt

    fig = _make_scan_figure(
        fitter, theta, theta_err, rec, n_sigma=n_sigma, n_points=n_points
    )
    if fig is not None:
        pp.savefig(fig)
        plt.close(fig)


def plot_scans(fitter, theta, theta_err, rec, out_path, n_sigma=5.0, n_points=101):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    fig = _make_scan_figure(
        fitter, theta, theta_err, rec, n_sigma=n_sigma, n_points=n_points
    )
    if fig is None:
        return
    with PdfPages(out_path) as pp:
        pp.savefig(fig)
    plt.close(fig)
    print(f"[SCAN] {out_path}", flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────


def _should_skip_scan(rec):
    return bool(rec.get("empty"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Fit result JSON")
    parser.add_argument("--out-fig", required=True, help="Output scan PDF")
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=5.0,
        help="Scan ±n_sigma around MLE (default: 5)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Grid points per parameter (default: 101)",
    )
    args = parser.parse_args()

    with open(args.input) as fh:
        rec = json.load(fh)

    if _should_skip_scan(rec):
        print(f"[SKIP] {args.input}: empty fit result", flush=True)
        sys.exit(0)

    charges = np.array(rec["hist_q"])
    counts = np.array(rec["hist_counts"])
    init = rec["init_estimate"]
    dc_model = rec.get("dark_model") or {}
    dc_info = rec.get("dark_count", {})

    dark_mu_init = dc_info.get("mu_dark") if dc_info.get("enabled") else None

    fitter = make_fitter(
        charges,
        counts,
        init["ped_mean"],
        init["ped_sigma"],
        init["spe_mean"],
        init["spe_sigma"],
        lam_init=rec.get("lam"),
        dark_mu_init=dark_mu_init,
        dark_T_gate=dc_model.get("T_gate", PEAKOTRON_T_GATE_NS),
        dark_t0_pre=dc_model.get("t0_pre", PEAKOTRON_T0_PRE_NS),
        dark_tau_slow=dc_model.get("tau_slow", PEAKOTRON_TAU_SLOW_NS),
        bin_edges=np.array(rec["hist_bin_edges"]),
        total_events=rec.get("n_events"),
        display_charges=np.array(rec.get("hist_full_q", rec["hist_q"])),
        display_counts=np.array(rec.get("hist_full_counts", rec["hist_counts"])),
    )

    theta = np.array(rec["theta"])
    theta_err = np.array(rec["theta_err"])

    print(
        f"[SCAN] V={rec['voltage']}V  n_params={len(theta)}"
        f"  n_sigma={args.n_sigma}  n_points={args.n_points}",
        flush=True,
    )
    plot_scans(
        fitter,
        theta,
        theta_err,
        rec,
        args.out_fig,
        n_sigma=args.n_sigma,
        n_points=args.n_points,
    )


if __name__ == "__main__":
    main()
