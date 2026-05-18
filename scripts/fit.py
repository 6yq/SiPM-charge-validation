#!/usr/bin/env python3
"""Fit one Hamamatsu PCB6 charge spectrum with NegBin AP fitter + optax L-BFGS."""

import argparse

import os
import sys
import jax
import json
import numpy as np
import jax.numpy as jnp

from math import log

from fit_io import (
    parse_histogram,
    trim_histogram,
    roi_lower_sigma_for_voltage,
    select_fit_roi,
    load_initial_values,
)
from fit_init import (
    estimate_init,
    estimate_dark_mu,
    dcr_policy,
    theta_from_initial_values,
)
from fit_defaults import (
    PEAKOTRON_TAU_SLOW_NS,
    PEAKOTRON_T0_PRE_NS,
    PEAKOTRON_T_GATE_NS,
)
from fit_build import make_fitter
from fit_optim import _is_gpu, fit_multistart
from fit_analysis import (
    spe_phys,
    dark_phys,
    compute_chi2,
    compute_cov,
    print_theta_table,
)
from fit_plot import save_fit_plot

jax.config.update("jax_enable_x64", True)

# Allow sibling imports from scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="PeakOTron histogram txt file")
    parser.add_argument("--voltage", type=float, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--out-fig", default=None, help="Per-voltage fit plot PDF")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument(
        "--n-seeds", type=int, default=16, help="Number of multi-start seeds"
    )
    parser.add_argument(
        "--init-json",
        default=None,
        help="Optional JSON with theta or physical ped/spe/lam/dark initial values",
    )

    dcr_group = parser.add_argument_group("dark count")
    dcr_group.add_argument(
        "--dcr",
        type=float,
        default=None,
        metavar="MU",
        help="Enable dark-count term with initial mu_dark=MU."
        " Overrides --dcr-auto estimate.",
    )
    dcr_group.add_argument(
        "--dcr-fixed",
        action="store_true",
        help="Fix mu_dark at --dcr value (not fitted).",
    )
    dcr_group.add_argument(
        "--dcr-auto",
        action="store_true",
        help="Auto-estimate mu_dark from 0PE-1PE valley; voltage policy applies"
        " (off <=53V, weak 53.5-54V, float >=54.5V).",
    )
    dcr_group.add_argument(
        "--gate-T",
        type=float,
        default=PEAKOTRON_T_GATE_NS,
        metavar="NS",
        help=f"Gate length in ns (default: {PEAKOTRON_T_GATE_NS:g}, PeakOTron PCB6)",
    )
    dcr_group.add_argument(
        "--gate-t0",
        type=float,
        default=PEAKOTRON_T0_PRE_NS,
        metavar="NS",
        help=f"Pre-gate window in ns (default: {PEAKOTRON_T0_PRE_NS:g}, PeakOTron PCB6)",
    )
    dcr_group.add_argument(
        "--tau-slow",
        type=float,
        default=PEAKOTRON_TAU_SLOW_NS,
        metavar="NS",
        help=f"Slow-pulse time constant in ns (default: {PEAKOTRON_TAU_SLOW_NS:g}, PeakOTron PCB6)",
    )
    args = parser.parse_args()

    charges, counts = parse_histogram(args.input)

    if len(counts) == 0 or counts.sum() == 0:
        print(f"[SKIP] {args.input} is empty", flush=True)
        with open(args.output, "w") as f:
            json.dump({"voltage": args.voltage, "empty": True, "converged": False}, f)
        sys.exit(0)

    charges, counts = trim_histogram(charges, counts)
    charges_full = charges
    counts_full = counts
    n_events_full = int(np.round(counts_full).astype(int).sum())

    ped_mean, ped_sigma, spe_mean, spe_sigma, lam_est = estimate_init(charges, counts)
    print(
        f"[INIT] V={args.voltage}V  ped_mean={ped_mean:.2f}  ped_sigma={ped_sigma:.2f}"
        f"  spe_mean={spe_mean:.2f}  spe_sigma={spe_sigma:.2f}  lam_est={lam_est:.3f}",
        flush=True,
    )

    roi_lower_sigma = roi_lower_sigma_for_voltage(args.voltage)
    charges_fit, counts_fit, roi_q_min = select_fit_roi(
        charges_full,
        counts_full,
        ped_mean,
        ped_sigma,
        lower_sigma=roi_lower_sigma,
    )
    n_events_fit = int(np.round(counts_fit).astype(int).sum())
    print(
        f"[ROI ] fit range Q >= {roi_q_min:.2f}"
        f"  fit_entries={n_events_fit}  total_entries={n_events_full}",
        flush=True,
    )

    # ─── DCR policy ───────────────────────────────────────────────────────────
    policy = dcr_policy(args.voltage)
    dcr_mu_init = None

    if args.dcr is not None or args.dcr_auto:
        if policy == "off":
            print(
                f"[DCR ] V={args.voltage}V: forced off — 0PE/1PE overlap at this voltage",
                flush=True,
            )
        else:
            if policy == "weak":
                print(
                    f"[DCR ] V={args.voltage}V: weakly constrained —"
                    f" treat DCR uncertainty with caution",
                    flush=True,
                )
            if args.dcr_auto and args.dcr is None:
                dcr_mu_init = estimate_dark_mu(
                    charges,
                    counts,
                    ped_mean,
                    spe_mean,
                    args.gate_T,
                    args.gate_t0,
                    args.tau_slow,
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
        charges_fit,
        counts_fit,
        ped_mean,
        ped_sigma,
        spe_mean,
        spe_sigma,
        lam_init=lam_est,
        dark_mu_init=dcr_mu_init,
        dark_T_gate=args.gate_T,
        dark_t0_pre=args.gate_t0,
        dark_tau_slow=args.tau_slow,
        total_events=n_events_full,
        display_charges=charges_full,
        display_counts=counts_full,
    )

    if args.dcr_fixed and dcr_mu_init is not None and "dark" in fitter.layout:
        dark_idx = fitter.layout["dark"].start
        log_mu = log(float(dcr_mu_init))
        fitter.bounds[dark_idx] = (log_mu, log_mu + 1e-10)
        print(f"[DCR ] log_mu_dark fixed at {log_mu:.4g}", flush=True)

    theta0 = theta_from_initial_values(fitter, initial_values)

    print(f"[INIT] initial theta — V={args.voltage}V", flush=True)
    print_theta_table(fitter, theta0, "INIT")

    _backend = jax.default_backend()
    _use_lax = _is_gpu()
    _loop_tag = "lax.while_loop" if _use_lax else "python"

    # warm up JIT; for lax path this also pre-compiles the loop body
    _ = fitter._logl_jit(jnp.asarray(theta0, dtype=jnp.float64))

    print(
        f"[FIT ] V={args.voltage}V  n_params={len(theta0)}"
        f"  dcr={'on' if dcr_mu_init is not None else 'off'}"
        f"  backend={_backend}  loop={_loop_tag}"
        f"  seeds={args.n_seeds}",
        flush=True,
    )

    try:
        theta, logl, converged, n_iter, trace_logl, trace_gnorm = fit_multistart(
            fitter,
            theta0=theta0,
            n_seeds=args.n_seeds,
            use_lax=_use_lax,
            maxiter=args.maxiter,
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

    print(f"[MLE ] final theta — V={args.voltage}V", flush=True)
    print_theta_table(fitter, theta, "MLE ", theta_err)

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
        "optimizer": f"zoom-lbfgs/{_loop_tag}",
        "empty": False,
        "converged": bool(converged),
        "logl": float(logl),
        "n_iter": int(n_iter),
        "n_events": n_events_full,
        "n_fit_events": n_events_fit,
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
        "dark_model": (
            {
                "T_gate": args.gate_T,
                "t0_pre": args.gate_t0,
                "tau_slow": args.tau_slow,
                "policy": policy,
            }
            if dcr_mu_init is not None
            else None
        ),
        "init_estimate": {
            "ped_mean": float(ped_mean),
            "ped_sigma": float(ped_sigma),
            "spe_mean": float(spe_mean),
            "spe_sigma": float(spe_sigma),
            "lam_est": float(lam_est),
        },
        "fit_roi": {
            "q_min": float(roi_q_min),
            "lower_sigma": float(roi_lower_sigma),
        },
        "hist_q": [float(x) for x in charges_fit],
        "hist_counts": [float(x) for x in counts_fit],
        "hist_full_q": [float(x) for x in charges_full],
        "hist_full_counts": [float(x) for x in counts_full],
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
