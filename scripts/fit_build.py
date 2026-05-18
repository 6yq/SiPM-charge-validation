import numpy as np
from math import log

from fitter.core.base import ParamBlock
from fitter.models.afterpulse import NegBinBetaAPFitter
from fitter.models.dark_count import make_dark_ft, make_dark_block
from fitter.models.gen_tweedie import reparam_from_spe


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
    bin_edges=None,
):
    """Build one NegBinBetaAPFitter with optional dark-count block."""
    counts_int = np.round(counts).astype(int)
    A = int(counts_int.sum())

    dq = float(np.median(np.diff(charges)))
    if bin_edges is None:
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
            (1e-4, 1.0 - 1e-4),    # xi ∈ (0, 1)
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
