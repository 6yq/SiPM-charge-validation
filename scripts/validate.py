#!/usr/bin/env python3
"""Aggregate fit results, write CSV, and produce validation figures."""
import argparse
import json
import os

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


def _dcr_enabled(rec):
    return bool(rec.get("dark_count", {}).get("enabled", False))


def _model_label(rec):
    return "dcr_on" if _dcr_enabled(rec) else "dcr_off"


def _bic(rec):
    logl = float(rec.get("logl", float("-inf")))
    k = len(rec.get("param_names", rec.get("theta", [])))
    n = int(rec.get("n_fit_events", rec.get("n_events", 0)))
    if not np.isfinite(logl) or k <= 0 or n <= 1:
        return float("inf")
    return float(k * np.log(n) - 2.0 * logl)


def _with_model_metrics(rec):
    out = dict(rec)
    out["dcr_enabled"] = _dcr_enabled(rec)
    out["model"] = _model_label(rec)
    out["n_params"] = len(rec.get("param_names", rec.get("theta", [])))
    out["bic"] = float(rec.get("bic", _bic(rec)))
    return out


def select_bic_records(records):
    """Return one BIC-preferred record per voltage plus all annotated models."""
    all_models = [_with_model_metrics(r) for r in records]
    grouped = {}
    for rec in all_models:
        grouped.setdefault(float(rec["voltage"]), []).append(rec)

    selected = []
    for voltage in sorted(grouped):
        candidates = grouped[voltage]
        best = min(candidates, key=lambda r: (r["bic"], not r.get("converged", False)))
        for rec in candidates:
            rec["bic_selected"] = rec is best
        selected.append(best)
    return selected, all_models


def build_dataframe(records):
    rows = []
    for r in records:
        spe = dict(r["spe"])
        if "ap_charge_mean" in spe and "spe_mean" in spe:
            gain = float(spe["spe_mean"])
            if np.isfinite(gain) and gain != 0.0:
                ap_frac = float(spe["ap_charge_mean"]) / gain
                spe["ap_charge_fraction"] = ap_frac
                ap_err = spe.get("ap_charge_mean_err")
                gain_err = spe.get("spe_mean_err")
                if (
                    ap_err is not None
                    and gain_err is not None
                    and np.isfinite(ap_err)
                    and np.isfinite(gain_err)
                ):
                    spe["ap_charge_fraction_err"] = abs(ap_frac) * float(
                        np.sqrt(
                            (float(ap_err) / (float(spe["ap_charge_mean"]) + 1e-30))
                            ** 2
                            + (float(gain_err) / gain) ** 2
                        )
                    )
        row = {
            "voltage": r["voltage"],
            "model": r.get("model", _model_label(r)),
            "dcr_enabled": bool(r.get("dcr_enabled", _dcr_enabled(r))),
            "n_params": int(r.get("n_params", len(r.get("param_names", [])))),
            "bic": float(r.get("bic", _bic(r))),
            "bic_selected": bool(r.get("bic_selected", False)),
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
        row.update({f"spe_{k}": v for k, v in spe.items()})
        row.update({f"dark_{k}": v for k, v in r.get("dark_count", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("voltage")


def correlation_param_label(name):
    labels = {
        "log_A": r"$\log A$",
        "ped_mean": r"$Q_0$",
        "ped_sigma": r"$\sigma_0$",
        "a_logSigma": r"$\log\sigma_{\mathrm{SPE}}$",
        "b_logDiff": r"$\log(G^*-\sigma_{\mathrm{SPE}})$",
        "log_xi": r"$\log\xi$",
        "xi": r"$\xi$",
        "log_rho": r"$\log\rho$",
        "logit_beta": r"$\mathrm{logit}\,m_{\mathrm{AP}}$",
        "lam": r"$\lambda$",
        "log_mu_dark": r"$\log\mu_{\mathrm{dark}}$",
    }
    escaped = name.replace("_", r"\_")
    return labels.get(name, rf"$\mathrm{{{escaped}}}$")


# ─── per-parameter validation plots ───────────────────────────────────────────

_VOLTAGE_LABEL = r"$V_{\mathrm{bias}}\;(\mathrm{V})$"


def _one_param_page(pp, vs, y, ye, ylabel, title=None):
    """Single errorbar plot on one PDF page."""
    fig, ax = plt.subplots()
    ax.plot(vs, y, linestyle="--", color="0.4", lw=1.0)
    ax.errorbar(
        vs,
        y,
        yerr=ye,
        fmt="o",
        color="k",
        ecolor="k",
        capsize=4,
        markersize=5,
        linestyle="none",
    )
    ax.set_xlabel(_VOLTAGE_LABEL)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    pp.savefig(fig)
    plt.close(fig)


def plot_validation(df, records, pp):
    vs = df["voltage"].values

    def _col(key):
        return df[key].values if key in df.columns else np.zeros(len(vs))

    # SPE parameters
    _one_param_page(
        pp,
        vs,
        _col("spe_spe_mean"),
        _col("spe_spe_mean_err"),
        r"$G^*$ (ADC)",
    )
    _one_param_page(
        pp,
        vs,
        _col("spe_spe_sigma"),
        _col("spe_spe_sigma_err"),
        r"$\sigma_{\mathrm{SPE}}$ (ADC)",
    )
    _one_param_page(
        pp,
        vs,
        _col("spe_spe_res") * 100,
        _col("spe_spe_res_err") * 100,
        r"$\sigma_{\mathrm{SPE}}/G^*$ (%)",
    )
    _one_param_page(
        pp,
        vs,
        _col("spe_xi"),
        None,
        r"$\xi$",
    )

    # Afterpulse parameters
    _one_param_page(
        pp,
        vs,
        _col("spe_rho"),
        _col("spe_rho_err"),
        r"$\rho$",
    )
    _one_param_page(
        pp,
        vs,
        _col("spe_ap_charge_fraction"),
        _col("spe_ap_charge_fraction_err"),
        r"$\langle Q_{\mathrm{AP}}\rangle/G^*$",
    )

    # Occupancy
    _one_param_page(
        pp,
        vs,
        _col("lam"),
        _col("lam_err"),
        r"$\lambda$",
    )

    # Pedestal
    _one_param_page(
        pp,
        vs,
        _col("ped_ped_mean"),
        _col("ped_ped_mean_err"),
        r"$Q_0$ (ADC)",
    )
    _one_param_page(
        pp,
        vs,
        _col("ped_ped_sigma"),
        _col("ped_ped_sigma_err"),
        r"$\sigma_0$ (ADC)",
    )

    # Fit quality
    chi2_ndf = _col("chi_sq") / np.maximum(_col("ndf"), 1)
    fig, ax = plt.subplots()
    ax.plot(vs, chi2_ndf, linestyle="--", color="0.4", lw=1.0)
    ax.plot(vs, chi2_ndf, "o", color="k", markersize=5)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label="ideal")
    ax.set_xlabel(_VOLTAGE_LABEL)
    ax.set_ylabel(r"$\chi^2/\mathrm{ndf}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    pp.savefig(fig)
    plt.close(fig)

    # Per-voltage correlation matrix — one page per voltage
    if records:
        for rec in records:
            corr = np.array(rec["corr"])
            param_labels = [
                correlation_param_label(name) for name in rec["param_names"]
            ]
            fig, ax = plt.subplots(constrained_layout=True)
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            fig.colorbar(im, ax=ax, label=r"$\mathrm{corr}(\theta_i,\theta_j)$")
            ax.set_xticks(range(len(param_labels)))
            ax.set_yticks(range(len(param_labels)))
            ax.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=12)
            ax.set_yticklabels(param_labels, fontsize=12)
            ax.tick_params(axis="both", which="both", length=0)
            n_param = corr.shape[0]
            for i in range(n_param):
                for j in range(n_param):
                    val = corr[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="white" if abs(val) > 0.6 else "black",
                        )
            ax.set_title(
                rf"Parameter correlations, $V_{{\mathrm{{bias}}}}={rec['voltage']:g}\,\mathrm{{V}}$"
            )
            pp.savefig(fig)
            plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Fit result JSON files")
    parser.add_argument("--out-csv", default="results/fit_results.csv")
    parser.add_argument(
        "--out-model-csv",
        default=None,
        help="All candidate model rows before BIC selection"
        " (default: <out-csv stem>_models.csv)",
    )
    parser.add_argument("--out-val", default="figures/validation.pdf")
    args = parser.parse_args()

    records = load_results(args.inputs)
    if not records:
        print("[WARN] no valid records to process", flush=True)
        return

    selected, all_models = select_bic_records(records)
    out_model_csv = args.out_model_csv
    if out_model_csv is None:
        root, ext = os.path.splitext(args.out_csv)
        out_model_csv = f"{root}_models{ext or '.csv'}"

    model_df = build_dataframe(all_models)
    model_df.to_csv(out_model_csv, index=False)
    print(f"[CSV ] {out_model_csv}  ({len(model_df)} model rows)", flush=True)

    df = build_dataframe(selected)
    df.to_csv(args.out_csv, index=False)
    print(f"[CSV ] {args.out_csv}  ({len(df)} rows)", flush=True)

    with PdfPages(args.out_val) as pp:
        plot_validation(df, selected, pp)
    print(f"[PDF ] {args.out_val}", flush=True)


if __name__ == "__main__":
    main()
