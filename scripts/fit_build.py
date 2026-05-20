import numpy as np
from math import log

from fitter.core.base import ParamBlock
from fitter.models.afterpulse import NegBinBetaAPFitter
from fitter.models.dark_count import make_dark_ft, make_dark_block
from fitter.models.gen_tweedie import GenTweedieFitter, reparam_from_spe

import jax.numpy as jnp

try:
    from fit_defaults import (
        PEAKOTRON_TAU_SLOW_NS,
        PEAKOTRON_T0_PRE_NS,
        PEAKOTRON_T_GATE_NS,
    )
except ImportError:
    from .fit_defaults import (
        PEAKOTRON_TAU_SLOW_NS,
        PEAKOTRON_T0_PRE_NS,
        PEAKOTRON_T_GATE_NS,
    )


def _spe_with_physical_xi(spe):
    return spe.at[2].set(jnp.exp(spe[2]))


class LogXiNegBinBetaAPFitter(NegBinBetaAPFitter):
    """NegBinBetaAP fitter using log_xi as the raw optimizer parameter."""

    def _model_callables(self):
        ft_extra, ser_ft, count_pgf, dark_ft = super()._model_callables()

        def count_pgf_log_xi(s, lam, spe):
            return count_pgf(s, lam, _spe_with_physical_xi(spe))

        return ft_extra, ser_ft, count_pgf_log_xi, dark_ft

    def _single_n_pgf(self, n):
        base_pgf = super()._single_n_pgf(n)

        def pgf(s, lam, spe):
            return base_pgf(s, lam, _spe_with_physical_xi(spe))

        return pgf

    def spe_report(self, spe_args) -> dict:
        spe_phys = np.asarray(spe_args, dtype=float).copy()
        spe_phys[2] = float(np.exp(spe_phys[2]))
        return super().spe_report(spe_phys)


class LogXiGenTweedieFitter(GenTweedieFitter):
    """GenTweedie fitter using log_theta and optional dark-count term."""

    def __init__(self, *args, dark_ft=None, **kwargs):
        self._dark_ft_opt = dark_ft
        super().__init__(*args, **kwargs)

    def _model_callables(self):
        ft_extra, ser_ft, count_pgf, _dark_ft = super()._model_callables()

        def count_pgf_log_theta(s, lam, spe):
            return count_pgf(s, lam, _spe_with_physical_xi(spe))

        return ft_extra, ser_ft, count_pgf_log_theta, self._dark_ft_opt

    def _single_n_pgf(self, n):
        base_pgf = super()._single_n_pgf(n)

        def pgf(s, lam, spe):
            return base_pgf(s, lam, _spe_with_physical_xi(spe))

        return pgf

    def spe_report(self, spe_args) -> dict:
        spe_phys = np.asarray(spe_args, dtype=float).copy()
        spe_phys[2] = float(np.exp(spe_phys[2]))
        return super().spe_report(spe_phys)


def make_fitter(
    charges,
    counts,
    ped_mean,
    ped_sigma,
    spe_mean,
    spe_sigma,
    lam_init=None,
    dark_mu_init=None,
    dark_T_gate=PEAKOTRON_T_GATE_NS,
    dark_t0_pre=PEAKOTRON_T0_PRE_NS,
    dark_tau_slow=PEAKOTRON_TAU_SLOW_NS,
    bin_edges=None,
    total_events=None,
    display_charges=None,
    display_counts=None,
    ap_enabled=True,
):
    """Build one NegBinBetaAPFitter with optional dark-count block."""
    counts_int = np.round(counts).astype(int)
    A = int(total_events) if total_events is not None else int(counts_int.sum())

    dq = float(np.median(np.diff(charges)))
    if bin_edges is None:
        bin_edges = np.append(charges - dq / 2, charges[-1] + dq / 2)
    Q_raw = np.repeat(charges, counts_int)
    q_min = min(float(charges[0]), ped_mean - 10 * ped_sigma)

    a0, b0 = reparam_from_spe(spe_mean, spe_sigma)
    extra_block = ParamBlock(
        name="pedestal",
        names=["ped_mean", "ped_sigma"],
        init=np.array([ped_mean, ped_sigma], dtype=float),
        bounds=[(-500.0, 500.0), (0.5, 500.0)],
    )
    spe_names = ["a_logSigma", "b_logDiff", "log_xi"]
    spe_init = [a0, b0, log(0.04)]
    spe_bounds = [
        (log(1.0), log(1e4)),   # sigma in (1, 10000) ADC
        (log(1.0), log(1e4)),   # mu-sigma in (1, 10000) ADC
        (log(1e-4), log(1.0 - 1e-4)),  # theta in (0, 1)
    ]
    if ap_enabled:
        log_rho0 = log(0.01)
        beta_0 = max(1e-4, 0.4 * spe_sigma / ((1.0 - 0.01) * spe_mean))
        logit_beta0 = log(beta_0 / (1.0 - beta_0))
        spe_names += ["log_rho", "logit_beta"]
        spe_init += [log_rho0, logit_beta0]
        spe_bounds += [
            (-15.0, -0.01),  # rho in (7e-7, 0.99)
            (-15.0, 15.0),   # AP charge fraction transform
        ]

    spe_block = ParamBlock(
        name="spe",
        names=spe_names,
        init=np.array(spe_init, dtype=float),
        bounds=spe_bounds,
    )

    dark_block = None
    dark_ft_fn = None
    if dark_mu_init is not None:
        dark_ft_fn = make_dark_ft(dark_T_gate, dark_t0_pre, dark_tau_slow)
        dark_block = make_dark_block(mu_dark_init=float(dark_mu_init))

    fitter_cls = LogXiNegBinBetaAPFitter if ap_enabled else LogXiGenTweedieFitter
    fitter = fitter_cls(
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

    if display_charges is not None and display_counts is not None:
        display_charges = np.asarray(display_charges, dtype=float)
        display_counts = np.asarray(display_counts, dtype=float)
        display_counts_int = np.round(display_counts).astype(int)
        display_dq = float(np.median(np.diff(display_charges)))
        fitter.display_hist = display_counts_int
        fitter.display_bins = np.append(
            display_charges - display_dq / 2,
            display_charges[-1] + display_dq / 2,
        )
        fitter.display_charges = display_charges
        fitter.display_counts = display_counts

    return fitter
