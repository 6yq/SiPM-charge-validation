#!/usr/bin/env python3
"""Aggregate fit results, write CSV, and produce validation figures."""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ─── helpers ──────────────────────────────────────────────────────────────────


def load_results(paths):
    records = []
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        if rec.get("empty", False):
            print(f"[SKIP] {p} is empty", flush=True)
            continue
        records.append(rec)
    records.sort(key=lambda r: r["voltage"])
    return records


def build_dataframe(records):
    rows = []
    for r in records:
        row = {
            "voltage": r["voltage"],
            "converged": r["converged"],
            "logl": r["logl"],
            "n_iter": r["n_iter"],
            "n_events": r["n_events"],
            "chi_sq": r["chi_sq"],
            "ndf": r["ndf"],
            "p_value": r["p_value"],
            "lam": r["lam"],
            "lam_err": r["lam_err"],
        }
        row.update({f"ped_{k}": v for k, v in r["ped"].items()})
        row.update({f"spe_{k}": v for k, v in r["spe"].items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("voltage")


# ─── per-parameter validation plots ───────────────────────────────────────────


def _one_param_page(pp, vs, y, ye, ylabel, title):
    """Single errorbar plot on one PDF page."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(vs, y, yerr=ye, fmt="o-", capsize=4, markersize=5)
    ax.set_xlabel("Bias voltage (V)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pp.savefig(fig)
    plt.close(fig)


def plot_validation(df, records, pp):
    vs = df["voltage"].values

    def _col(key):
        return df[key].values if key in df.columns else np.zeros(len(vs))

    # SPE parameters
    _one_param_page(pp, vs, _col("spe_spe_mean"), _col("spe_spe_mean_err"),
                    "SPE mean (ADC)", "Gain vs bias voltage")
    _one_param_page(pp, vs, _col("spe_spe_sigma"), _col("spe_spe_sigma_err"),
                    "SPE σ (ADC)", "SPE sigma vs bias voltage")
    _one_param_page(pp, vs, _col("spe_spe_res") * 100, _col("spe_spe_res_err") * 100,
                    "SPE resolution σ/μ (%)", "SPE resolution vs bias voltage")
    _one_param_page(pp, vs, _col("spe_xi"), None,
                    "ξ (gen-Poisson dispersion)", "Gen-Poisson dispersion ξ vs bias voltage")

    # Afterpulse parameters
    _one_param_page(pp, vs, _col("spe_rho"), _col("spe_rho_err"),
                    "ρ (afterpulse probability)", "Afterpulse probability ρ vs bias voltage")
    _one_param_page(pp, vs, _col("spe_beta"), _col("spe_beta_err"),
                    "β = <Q_AP>/((1-ρ)G)", "Afterpulse mean-charge parameter β vs bias voltage")
    _one_param_page(pp, vs, _col("spe_beta_shape_b"), _col("spe_beta_shape_b_err"),
                    "Beta(2,b) shape b", "Afterpulse charge shape b vs bias voltage")
    _one_param_page(pp, vs, _col("spe_ap_charge_mean"), _col("spe_ap_charge_mean_err"),
                    "<Q_AP> (ADC)", "Mean afterpulse charge vs bias voltage")
    _one_param_page(pp, vs, _col("spe_alpha"), _col("spe_alpha_err"),
                    "α = ρβ (mean AP fraction)", "Mean AP fraction α vs bias voltage")

    # Occupancy
    _one_param_page(pp, vs, _col("lam"), _col("lam_err"),
                    "λ (mean PE / trigger)", "Occupancy λ vs bias voltage")

    # Pedestal
    _one_param_page(pp, vs, _col("ped_ped_mean"), _col("ped_ped_mean_err"),
                    "Pedestal mean (ADC)", "Pedestal mean vs bias voltage")
    _one_param_page(pp, vs, _col("ped_ped_sigma"), _col("ped_ped_sigma_err"),
                    "Pedestal σ (ADC)", "Pedestal sigma vs bias voltage")

    # Fit quality
    chi2_ndf = _col("chi_sq") / np.maximum(_col("ndf"), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vs, chi2_ndf, "o-", markersize=5)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="ideal")
    ax.set_xlabel("Bias voltage (V)")
    ax.set_ylabel("χ²/ndf")
    ax.set_title("Fit quality χ²/ndf vs bias voltage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pp.savefig(fig)
    plt.close(fig)

    _one_param_page(pp, vs, _col("p_value"), None,
                    "p-value", "Fit p-value vs bias voltage")

    # Mean parameter correlation matrix
    all_corrs = []
    param_names = None
    for rec in records:
        if not rec.get("converged", False):
            continue
        corr = np.array(rec["corr"])
        if np.any(np.isfinite(corr)):
            all_corrs.append(corr)
            if param_names is None:
                param_names = rec["param_names"]

    if all_corrs and param_names:
        mean_corr = np.nanmean(np.stack(all_corrs), axis=0)
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        im = ax.imshow(mean_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(param_names, fontsize=9)
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                val = mean_corr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > 0.6 else "black")
        ax.set_title(f"Mean parameter correlation matrix (N={len(all_corrs)} fits)")
        pp.savefig(fig)
        plt.close(fig)

    # Per-voltage correlation matrix — one page per voltage
    if param_names:
        for rec in records:
            corr = np.array(rec["corr"])
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im, ax=ax, label="Correlation")
            ax.set_xticks(range(len(param_names)))
            ax.set_yticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(param_names, fontsize=8)
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    val = corr[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if abs(val) > 0.6 else "black")
            ax.set_title(f"Parameter correlations — {rec['voltage']} V")
            pp.savefig(fig)
            plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Fit result JSON files")
    parser.add_argument("--out-csv", default="results/fit_results.csv")
    parser.add_argument("--out-val", default="figures/validation.pdf")
    args = parser.parse_args()

    records = load_results(args.inputs)
    if not records:
        print("[WARN] no valid records to process", flush=True)
        return

    df = build_dataframe(records)
    df.to_csv(args.out_csv, index=False)
    print(f"[CSV ] {args.out_csv}  ({len(df)} rows)", flush=True)

    with PdfPages(args.out_val) as pp:
        plot_validation(df, records, pp)
    print(f"[PDF ] {args.out_val}", flush=True)


if __name__ == "__main__":
    main()
