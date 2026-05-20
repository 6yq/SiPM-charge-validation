#!/usr/bin/env python3
"""Run DCR/AP model comparisons and validate selected results."""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
VOLTAGES = [53, 53.5, 54, 54.5, 55, 55.5, 56, 56.5, 57, 57.5, 58, 58.5, 59, 59.5, 60]
GRID_TEST_VOLTAGES = [53, 53.5, 54, 54.5, 55]


def _stem(voltage):
    return f"{voltage:g}"


def _data_path(voltage):
    return f"data/PCB6_MPPC_{_stem(voltage)}V_histo.txt"


def _result_path(voltage, suffix=""):
    return f"results/PCB6_MPPC_{_stem(voltage)}V{suffix}.json"


def _figure_path(voltage, suffix=""):
    return f"figures/PCB6_MPPC_{_stem(voltage)}V{suffix}.pdf"


def _run_fit(args, voltage, out_path, fig_path, fit_args, force=False):
    if os.path.exists(out_path) and not (args.force or force):
        print(f"[KEEP] {out_path}", flush=True)
        return
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/fit.py",
        _data_path(voltage),
        "-o",
        out_path,
        "--out-fig",
        fig_path,
        "--voltage",
        _stem(voltage),
        "--n-seeds",
        str(args.seeds),
        "--maxiter",
        str(args.maxiter),
        *fit_args,
    ]
    print(f"[RUN ] {' '.join(cmd)}", flush=True)
    with open(f"{out_path}.log", "w") as stdout, open(f"{out_path}.time.log", "w") as stderr:
        subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr)


def _ensure_policy_fit(args, voltage):
    out = _result_path(voltage)
    _run_fit(args, voltage, out, _figure_path(voltage), ["--dcr-auto"])
    return out


def _ensure_grid_fits(args, voltage):
    specs = [
        ("", ["--dcr-auto"]),
        ("_nodcr" if voltage > 53.5 else "_dcr", [] if voltage > 53.5 else ["--dcr-auto", "--dcr-force"]),
        ("_apoff", ["--ap-off"]),
        ("_dcr_apoff", ["--dcr-auto", "--dcr-force", "--ap-off"]),
    ]
    paths = []
    for suffix, fit_args in specs:
        out = _result_path(voltage, suffix)
        _run_fit(
            args,
            voltage,
            out,
            _figure_path(voltage, suffix),
            fit_args,
            force=args.force_grid,
        )
        paths.append(out)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="rerun fits even if output exists")
    parser.add_argument(
        "--force-grid",
        action="store_true",
        help="rerun only the 53.0-55.0 V 2x2 comparison fits",
    )
    parser.add_argument("--out-csv", default="results/fit_results.csv")
    parser.add_argument("--out-model-csv", default="results/fit_results_models.csv")
    parser.add_argument("--out-val", default="figures/validation.pdf")
    args = parser.parse_args()

    result_paths = []
    for voltage in VOLTAGES:
        if voltage in GRID_TEST_VOLTAGES:
            result_paths.extend(_ensure_grid_fits(args, voltage))
        else:
            result_paths.append(_ensure_policy_fit(args, voltage))

    validate_cmd = [
        sys.executable,
        "scripts/validate.py",
        *result_paths,
        "--out-csv",
        args.out_csv,
        "--out-model-csv",
        args.out_model_csv,
        "--out-val",
        args.out_val,
    ]
    print(f"[RUN ] {' '.join(validate_cmd)}", flush=True)
    subprocess.run(validate_cmd, check=True)


if __name__ == "__main__":
    main()
