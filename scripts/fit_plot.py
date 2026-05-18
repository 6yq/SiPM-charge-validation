import numpy as np

from fitter.tests.plot import make_figure, n_max, plot_histogram_with_fit


def _finite_or_zero(v):
    f = float(v)
    return f if np.isfinite(f) else 0.0


def _rebin(hist, bins, comps, ys, target=250):
    """Merge adjacent bins for display; returns (hist, bins, comps, ys)."""
    n = len(hist)
    if n <= target:
        return hist, bins, comps, ys
    factor = max(2, n // target)
    n_trim = (n // factor) * factor

    def _merge(arr):
        return arr[:n_trim].reshape(-1, factor).sum(axis=1)

    new_hist = _merge(hist)
    new_ys = _merge(ys) if ys is not None else None
    new_comps = [_merge(c) for c in comps]
    new_bins = np.concatenate([bins[:n_trim:factor], [bins[n_trim]]])
    return new_hist, new_bins, new_comps, new_ys


def _clip_curves_to_roi(bin_centers, comps, xsp, smooth, x_lo, x_hi):
    """Clip smooth x-grid and NaN-mask component curves outside the ROI."""
    mask_sm = (xsp >= x_lo) & (xsp <= x_hi)
    mask_bc = (bin_centers >= x_lo) & (bin_centers <= x_hi)
    comps_roi = []
    for comp in comps:
        comp_roi = np.asarray(comp, dtype=float).copy()
        comp_roi[~mask_bc] = np.nan
        comps_roi.append(comp_roi)
    return xsp[mask_sm], smooth[mask_sm], comps_roi


def _plot_roi_lower(ped_mean, ped_sigma, lower_sigma=3.5):
    return float(ped_mean) - float(lower_sigma) * float(ped_sigma)


def _style_project_fit_lines(ax_main, fit_label=None):
    """Use project-local styling: red total fit, black PE components."""
    for line in ax_main.lines:
        linestyle = line.get_linestyle()
        marker = line.get_marker()
        if linestyle in ("None", "none", ""):
            continue
        if marker not in (None, "", "None"):
            continue
        if linestyle == "--":
            line.set_color("k")
        else:
            line.set_color("red")


def _pull_points_in_roi(bin_centers, hist, ys, x_lo, x_hi):
    mask = (bin_centers >= x_lo) & (bin_centers <= x_hi)
    pull = (hist - ys) / np.sqrt(np.maximum(ys, 1.0))
    return bin_centers[mask], pull[mask]


def _histogram_data_range(bins, hist):
    nz = np.where(np.asarray(hist) > 0)[0]
    if len(nz) == 0:
        return float(bins[0]), float(bins[-1])
    return float(bins[nz[0]]), float(bins[nz[-1] + 1])


def _place_main_legend(ax_main, extra_info):
    from matplotlib.lines import Line2D

    handles, labels = ax_main.get_legend_handles_labels()
    if extra_info:
        for info_line in extra_info:
            handles.append(Line2D([], [], linestyle="none", color="none"))
            labels.append(info_line)
    if not handles:
        return None
    return ax_main.legend(
        handles,
        labels,
        frameon=False,
        fancybox=False,
        framealpha=0.78,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.2,
        labelspacing=0.4,
        handlelength=2.0,
        prop={"size": 14},
    )


def _hide_component_legend(ax_leg):
    if ax_leg is None:
        return
    ax_leg.clear()
    ax_leg.axis("off")
    ax_leg.set_visible(False)


def _build_extra_info(rec):
    """Build physics-parameter annotation list for the legend."""
    spe = rec.get("spe", {})
    ped = rec.get("ped", {})
    lines = []

    gm = spe.get("spe_mean", float("nan"))
    gm_err = spe.get("spe_mean_err", float("nan"))
    if np.isfinite(gm):
        if np.isfinite(gm_err) and gm_err > 0:
            lines.append(rf"$G^*={gm:.2f}\pm{gm_err:.2f}$")
        else:
            lines.append(rf"$G^*={gm:.2f}$")

    s0 = ped.get("ped_sigma", float("nan"))
    s0_err = ped.get("ped_sigma_err", float("nan"))
    if np.isfinite(s0):
        if np.isfinite(s0_err) and s0_err > 0:
            lines.append(rf"$\sigma_0={s0:.2f}\pm{s0_err:.2f}$")
        else:
            lines.append(rf"$\sigma_0={s0:.2f}$")

    rho = spe.get("rho", float("nan"))
    rho_err = spe.get("rho_err", float("nan"))
    if np.isfinite(rho):
        if np.isfinite(rho_err) and rho_err > 0:
            lines.append(rf"$\rho={rho:.4f}\pm{rho_err:.4f}$")
        else:
            lines.append(rf"$\rho={rho:.4f}$")

    beta = spe.get("beta", float("nan"))
    beta_err = spe.get("beta_err", float("nan"))
    if np.isfinite(beta):
        if np.isfinite(beta_err) and beta_err > 0:
            lines.append(rf"$\beta={beta:.4f}\pm{beta_err:.4f}$")
        else:
            lines.append(rf"$\beta={beta:.4f}$")

    dc = rec.get("dark_count", {})
    if dc.get("enabled"):
        mu_d = dc.get("mu_dark", float("nan"))
        mu_d_err = dc.get("mu_dark_err", float("nan"))
        if np.isfinite(mu_d):
            if np.isfinite(mu_d_err) and mu_d_err > 0:
                lines.append(rf"$\mu_\mathrm{{dark}}={mu_d:.3g}\pm{mu_d_err:.2g}$")
            else:
                lines.append(rf"$\mu_\mathrm{{dark}}={mu_d:.3g}$")

    return lines


def save_fit_plot(fitter, theta, theta_err, rec, out_path, trace_logl, trace_gnorm):
    """PDF: charge spectrum fit, optimizer trace (CPU only), likelihood scans."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from fit_scan import append_to_pdf

    ly = fitter.layout
    lam = float(theta[ly["lam"].start]) + fitter.lam_dc
    lam_err_raw = float(theta_err[ly["lam"].start])

    n_pe = n_max(lam)
    hist = getattr(fitter, "display_hist", fitter.hist).astype(float)
    bins = getattr(fitter, "display_bins", fitter.bins)
    bin_centers_full = (bins[:-1] + bins[1:]) / 2

    # ── rebin for display ──────────────────────────────────────────────────
    ys_full = fitter.estimate_bin_counts_at(theta, bins)
    comps_full = [
        fitter.estimate_component_counts_at(theta, k, bins) for k in range(1, n_pe + 1)
    ]
    labels = [f"{k} PE" for k in range(1, n_pe + 1)]

    hist_d, bins_d, comps_d, ys_d = _rebin(hist, bins, comps_full, ys_full, target=250)
    bin_centers_d = (bins_d[:-1] + bins_d[1:]) / 2

    # ── smooth: normalize by the display bin width, not the original ───────
    display_bin_width = float(bins_d[1] - bins_d[0])
    smooth_full = fitter.estimate_density(theta) * display_bin_width
    xsp = fitter.grid.xsp

    # ── clip smooth/comps curves to ROI [Q0 − 3.5σ₀, Qmax] ────────────────
    # Only the fit/component lines are clipped; histogram is full.
    ped_mean_v = rec["ped"]["ped_mean"]
    ped_sigma_v = rec["ped"]["ped_sigma"]
    roi_info = rec.get("fit_roi", {})
    roi_lower_sigma = float(roi_info.get("lower_sigma", 3.5))
    x_lo_roi = _plot_roi_lower(ped_mean_v, ped_sigma_v, roi_lower_sigma)
    _, x_hi_roi = _histogram_data_range(bins_d, hist_d)

    xsp_d, smooth_d, comps_roi = _clip_curves_to_roi(
        bin_centers_d, comps_d, xsp, smooth_full, x_lo_roi, x_hi_roi
    )

    occ = float(np.clip(1.0 - np.exp(-lam), 0.0, 1.0 - 1e-9))
    occ_std = float(lam_err_raw * np.exp(-lam)) if np.isfinite(lam_err_raw) else 0.0
    extra_info = _build_extra_info(rec)

    with PdfPages(out_path) as pp:
        # ── page 1: charge spectrum ──────────────────────────────────────────
        fig, ax_main, ax_resid, ax_leg = make_figure(n_comps=0)
        fig.set_size_inches(8.0, 6.2, forward=True)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.10, hspace=0.08)
        plot_histogram_with_fit(
            bins=bins_d,
            hist=hist_d,
            xsp=xsp_d,
            smooth=smooth_d,
            bin_centers=bin_centers_d,
            comps=comps_roi,
            labels=labels,
            params=np.array(theta[ly["spe"]]),
            occ=occ,
            occ_std=occ_std,
            ped_mean=ped_mean_v,
            gm=rec["spe"]["spe_mean"],
            chiSq=rec["chi_sq"],
            ndf=rec["ndf"],
            ys=ys_d,
            logscale=True,
            ax_main=ax_main,
            ax_resid=ax_resid,
            ax_leg=ax_leg,
            fig=fig,
            extra_info=[],
        )
        _style_project_fit_lines(ax_main)
        _hide_component_legend(ax_leg)
        ax_main.set_title(f"Hamamatsu PCB6  {rec['voltage']} V")
        ax_main.set_ylim(bottom=0.5)
        ax_main.set_xlim(float(bins_d[0]), float(bins_d[-1]))
        leg = ax_main.get_legend()
        if leg is not None:
            leg.remove()
        _place_main_legend(ax_main, extra_info)

        # ── pull panel: same ROI as the fitted line/chi-square ───────────────
        pull_x, pull = _pull_points_in_roi(
            bin_centers_d, hist_d, ys_d, x_lo_roi, x_hi_roi
        )
        ax_resid.cla()
        ax_resid.axhline(0, color="gray", lw=1, ls="--")
        ax_resid.axhline(+1, color="C0", lw=0.6, ls=":")
        ax_resid.axhline(-1, color="C0", lw=0.6, ls=":")
        ax_resid.axhline(+3, color="C1", lw=0.6, ls=":")
        ax_resid.axhline(-3, color="C1", lw=0.6, ls=":")
        ax_resid.plot(pull_x, pull, "o", color="black", ms=2)
        ax_resid.set_ylabel("Pull")
        ax_resid.set_xlabel("Q")
        ax_resid.set_xlim(float(bins_d[0]), float(bins_d[-1]))
        ax_resid.grid(True, alpha=0.3)
        pp.savefig(fig)
        plt.close(fig)

        # ── page 2: optimizer trace (CPU path only) ──────────────────────────
        if len(trace_logl) > 1:
            fig, axes = plt.subplots(2, 1, sharex=True)
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

        # ── page 3+: likelihood scan pages ──────────────────────────────────
        append_to_pdf(pp, fitter, theta, theta_err, rec)

    print(f"[PLOT] {out_path}", flush=True)
