import numpy as np

from fitter.tests.plot import make_figure, n_max, plot_histogram_with_fit


def _finite_or_zero(v):
    f = float(v)
    return f if np.isfinite(f) else 0.0


def save_fit_plot(fitter, theta, theta_err, rec, out_path, trace_logl, trace_gnorm):
    """PDF with charge spectrum fit (page 1) and optimizer trace (page 2, CPU only)."""
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

    dcr_info = rec.get("dark_count", {})
    mu_str = f"  μ_dark={dcr_info['mu_dark']:.3g}" if dcr_info.get("enabled") else ""

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

        # ── page 2: optimizer trace (empty on GPU/lax path) ──────────────────
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
