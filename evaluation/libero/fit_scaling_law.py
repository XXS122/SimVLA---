#!/usr/bin/env python3
"""
Inference-Time Scaling Law Fitting (Task-2)

Reads the sweep results CSV produced by run_scaling_eval.sh and fits, per
task suite, a power law on the best-of-K error rate:

    err(K) = a * K^(-b) + c

where err = 1 - success_rate. Reports (a, b, c), R², and the success-latency
Pareto table. The pre-registered falsifiable prediction of Task-2 is R² >= 0.9
on the bok rows; if the fit fails that threshold, the scaling-law hypothesis
is rejected.

CSV columns: mode,K,suite,success_rate,episodes,mean_latency_ms

Usage:
    python fit_scaling_law.py --csv results/scaling_results.csv \
        --out_json results/scaling_fit.json --out_plot results/scaling_fit.png
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def power_law(K, a, b, c):
    return a * np.power(K, -b) + c


def r_squared(y, y_pred):
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "mode": row["mode"],
                "K": int(row["K"]),
                "suite": row["suite"],
                "success_rate": float(row["success_rate"]),
                "episodes": int(row.get("episodes", 0) or 0),
                "mean_latency_ms": float(row.get("mean_latency_ms", 0) or 0),
            })
    return rows


def fit_suite(points):
    """points: list of (K, err) with K >= 1, sorted."""
    K = np.array([p[0] for p in points], dtype=np.float64)
    err = np.array([p[1] for p in points], dtype=np.float64)
    if len(K) < 3:
        return None
    # init: a = err(K=1) - err(K=max), b = 0.5, c = err(K=max)
    p0 = [max(err[0] - err[-1], 1e-3), 0.5, max(err[-1], 1e-3)]
    try:
        popt, _ = curve_fit(
            power_law, K, err, p0=p0,
            bounds=([0, 0, 0], [1.0, 5.0, 1.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    pred = power_law(K, *popt)
    return {
        "a": float(popt[0]),
        "b": float(popt[1]),
        "c": float(popt[2]),
        "r_squared": r_squared(err, pred),
        "K": K.tolist(),
        "err": err.tolist(),
        "err_pred": pred.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_plot", type=str, default=None)
    parser.add_argument("--r2_threshold", type=float, default=0.9,
                        help="Pre-registered falsification threshold")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    # ------------- scaling fit on bok rows -------------
    by_suite = defaultdict(list)
    for r in rows:
        if r["mode"] == "bok":
            by_suite[r["suite"]].append((r["K"], 1.0 - r["success_rate"]))

    fits = {}
    for suite, pts in sorted(by_suite.items()):
        pts = sorted(set(pts))
        fit = fit_suite(pts)
        fits[suite] = fit
        if fit is None:
            print(f"[{suite}] not enough K points to fit (need >= 3)")
            continue
        verdict = "SUPPORTED" if fit["r_squared"] >= args.r2_threshold else "REJECTED"
        print(f"[{suite}] err(K) = {fit['a']:.4f} * K^(-{fit['b']:.3f}) + {fit['c']:.4f}"
              f"   R²={fit['r_squared']:.4f}  -> scaling-law hypothesis {verdict}")

    # ------------- success-latency Pareto table -------------
    print("\n=== Success / latency Pareto table ===")
    print(f"{'suite':<16}{'mode':<9}{'K':>4}{'success':>10}{'latency(ms)':>14}")
    pareto = []
    for r in sorted(rows, key=lambda x: (x["suite"], x["mode"], x["K"])):
        print(f"{r['suite']:<16}{r['mode']:<9}{r['K']:>4}"
              f"{r['success_rate']:>10.3f}{r['mean_latency_ms']:>14.1f}")
        pareto.append(r)

    result = {
        "r2_threshold": args.r2_threshold,
        "fits": fits,
        "rows": pareto,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved fit summary to {args.out_json}")

    # ------------- optional plot -------------
    if args.out_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
            return

        n = max(1, len([f for f in fits.values() if f]))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        i = 0
        for suite, fit in sorted(fits.items()):
            if fit is None:
                continue
            ax = axes[0][i]
            i += 1
            K = np.array(fit["K"])
            ax.plot(K, fit["err"], "o", label="measured error")
            K_dense = np.logspace(0, math.log10(K.max()), 64)
            ax.plot(K_dense, power_law(K_dense, fit["a"], fit["b"], fit["c"]),
                    "-", label=f"fit  R²={fit['r_squared']:.3f}")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("K (candidates)")
            ax.set_ylabel("1 - success rate")
            ax.set_title(suite)
            ax.legend()
        fig.tight_layout()
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_plot, dpi=150)
        print(f"Saved plot to {args.out_plot}")


if __name__ == "__main__":
    main()
