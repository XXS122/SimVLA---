#!/usr/bin/env python
"""
plot_suite_bars.py
==================
Stacked per-suite bar chart at one budget point: the grey base of each bar is
the Uniform success rate, the blue segment stacked on top is the TDS gain, so
the total height is the TDS success rate. An "Avg" bar is appended
automatically.

Defaults are the real same-machine 40k numbers (docs/results_log.md sec.7);
override via --uniform/--tds for other budgets.

Usage:
  python plot_suite_bars.py --out figures_method/suite_bars_40k.png
  python plot_suite_bars.py --budget 100k \
      --uniform 92.5 96.0 90.5 87.5 --tds 97.5 98.5 96.0 90.5 \
      --out figures_method/suite_bars_100k.png
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNI = "#9aa0a6"    # grey  (Uniform base)
TDS = "#1a73e8"    # blue  (TDS gain segment)
SUITES = ["Spatial", "Object", "Goal", "Long"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform", nargs=4, type=float,
                    default=[73.0, 95.5, 38.5, 40.5],
                    help="Uniform SR for Spatial Object Goal Long")
    ap.add_argument("--tds", nargs=4, type=float,
                    default=[97.5, 98.5, 92.0, 71.0],
                    help="TDS SR for Spatial Object Goal Long")
    ap.add_argument("--budget", default="40k")
    ap.add_argument("--out", default="figures_method/suite_bars_40k.png")
    args = ap.parse_args()

    uni = np.array(args.uniform + [np.mean(args.uniform)])
    tds = np.array(args.tds + [np.mean(args.tds)])
    if np.any(tds < uni):
        print("[note] some suites have TDS < Uniform; those bars will show "
              "no gain segment and get a text label instead")
    gain = np.clip(tds - uni, 0, None)
    labels = SUITES + ["Avg"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x, uni, width=0.58, color=UNI, label="Uniform")
    ax.bar(x, gain, width=0.58, bottom=uni, color=TDS, label="TDS gain")

    for i in range(len(labels)):
        # total (TDS) on top of the bar
        ax.text(x[i], max(tds[i], uni[i]) + 1.5, f"{tds[i]:.1f}",
                ha="center", fontsize=9, color=TDS, fontweight="bold")
        # uniform value just below the junction, inside the grey base
        ax.text(x[i], uni[i] - 2.0, f"{uni[i]:.1f}", ha="center", va="top",
                fontsize=8, color="white")
        # delta beside the gain segment
        if gain[i] >= 4:
            ax.text(x[i], uni[i] + gain[i] / 2, f"+{gain[i]:.1f}",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(100, color="grey", lw=0.6, ls=":")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"LIBERO per-suite success rate @ {args.budget} steps")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out, {l: f"{u:.1f}->{t:.1f}" for l, u, t in zip(labels, uni, tds)})


if __name__ == "__main__":
    main()
