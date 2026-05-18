import numpy as np
import scipy.stats
import jax
import jax.numpy as jnp


def spe_phys(fitter, spe_args, spe_err):
    r = fitter.spe_report(spe_args)
    a, b = float(spe_args[0]), float(spe_args[1])
    da, db = float(spe_err[0]), float(spe_err[1])
    sigma = float(np.exp(a))
    r["spe_sigma_err"] = sigma * da
    r["spe_mean_err"] = float(np.sqrt((sigma * da) ** 2 + (float(np.exp(b)) * db) ** 2))
    rho = r["rho"]
    beta = r["beta"]
    if "log_xi" in fitter.param_names:
        log_xi_idx = list(fitter.param_names).index("log_xi") - fitter.layout["spe"].start
        r["xi_err"] = r["xi"] * float(spe_err[log_xi_idx])
    else:
        r["xi_err"] = float(spe_err[2])
    drho = rho * float(spe_err[3])
    dbeta = beta * (1.0 - beta) * float(spe_err[4])
    r["rho_err"] = drho
    r["beta_err"] = dbeta
    r["ap_charge_mean_err"] = r["ap_charge_mean"] * float(
        np.sqrt(
            (drho / (rho + 1e-30)) ** 2
            + (dbeta / (beta + 1e-30)) ** 2
            + (r["spe_mean_err"] / (r["spe_mean"] + 1e-30)) ** 2
        )
    )
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
    """Extract dark-count physical quantities; returns None when no dark block."""
    if "dark" not in fitter.layout:
        return None
    dark_sl = fitter.layout["dark"]
    log_mu = float(theta[dark_sl.start])
    log_mu_err = float(theta_err[dark_sl.start])
    mu_dark = float(np.exp(log_mu))
    mu_dark_err = (
        float(mu_dark * log_mu_err) if np.isfinite(log_mu_err) else float("nan")
    )
    return {
        "enabled": True,
        "mu_dark": mu_dark,
        "mu_dark_err": mu_dark_err,
        "log_mu_dark": log_mu,
        "log_mu_dark_err": log_mu_err,
    }


def compute_chi2(fitter, theta):
    """Neyman-B chi-squared over the fitted histogram ROI."""
    obs = fitter.hist.astype(float)
    exp = fitter.estimate_bin_counts(theta)

    mask = exp >= 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mask, (obs - exp) ** 2 / exp, 0.0)
    chi2 = float(np.sum(terms))

    n_bins_used = int(mask.sum())
    ndf = n_bins_used - len(theta)
    p_val = float(scipy.stats.chi2.sf(chi2, ndf)) if ndf > 0 else float("nan")
    return chi2, ndf, p_val


def print_theta_table(fitter, theta, label, theta_err=None):
    """Print raw + physical parameter values to stdout, one line per parameter."""
    names = list(fitter.param_names)
    theta = np.asarray(theta, dtype=float)
    errs = np.asarray(theta_err, dtype=float) if theta_err is not None else None

    a_idx = names.index("a_logSigma") if "a_logSigma" in names else None

    for i, name in enumerate(names):
        raw = float(theta[i])
        err = float(errs[i]) if errs is not None else None

        if name == "a_logSigma":
            phys = f"sigma={np.exp(raw):.3f}"
        elif name == "b_logDiff":
            if a_idx is not None:
                phys = f"spe_mean={np.exp(float(theta[a_idx])) + np.exp(raw):.3f}"
            else:
                phys = f"exp={np.exp(raw):.3f}"
        elif name == "log_xi":
            phys = f"xi={np.exp(raw):.6g}"
        elif name == "log_rho":
            phys = f"rho={np.exp(raw):.5f}"
        elif name == "logit_beta":
            phys = f"beta={1.0 / (1.0 + np.exp(-raw)):.5f}"
        elif name == "log_mu_dark":
            phys = f"mu_dark={np.exp(raw):.4g}"
        elif name == "log_A":
            phys = f"A={np.exp(raw):.0f}"
        else:
            phys = ""

        if err is not None and np.isfinite(err) and err > 0:
            err_str = f"±{err:.4f}"
        else:
            err_str = ""

        print(
            f"[{label}]   {name:<18s}  {raw:+12.5f}  {err_str:<12s}  {phys}",
            flush=True,
        )


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
