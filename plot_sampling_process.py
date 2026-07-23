#!/usr/bin/env python
"""
plot_sampling_process.py
========================
Three-panel figure showing how difficulty scores become the sampling
distribution (the "Sampling Distribution" module of the method figure):

  (1) difficulty scores d_k            sorted easy -> hard, diverging colors
  (2) softmax(d_k / T)                 sharpness-controlled normalization
  (3) (1-rho)*softmax + rho*(1/N)      uniform-floor mixing -> final p_k

Same tasks, same order, same colors in every panel -- only the heights
change. Dashed lines: uniform 1/N (grey) and floor rho/N (blue). Arrows
between panels carry the operation labels.

Depends only on numpy + matplotlib (no pandas).

Usage:
  python plot_sampling_process.py --difficulty task_difficulty_2comp.csv \
      --temperature 2.0 --rho 0.5 --out figures_method/sampling_process.png
"""

import argparse
import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch

GREY = "#9aa0a6"
BLUE = "#1a73e8"
LABEL_BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True, help="csv with a d_k column")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", default="figures_method/sampling_process.png")
    args = ap.parse_args()

    with open(args.difficulty, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "d_k" not in rows[0]:
        raise SystemExit(f"{args.difficulty} has no 'd_k' column")
    dk = np.sort(np.array([float(r["d_k"]) for r in rows]))  # easy -> hard
    N = len(dk)
    T, rho = args.temperature, args.rho

    z = np.exp(dk / T)
    p_soft = z / z.sum()                    # sharpness-controlled softmax
    p_final = (1 - rho) * p_soft + rho / N  # uniform-floor mixing
    uniform, floor = 1.0 / N, rho / N

    # Same diverging colors (blue = easy, red = hard) in all panels
    vmin, vmax = float(dk.min()), float(dk.max())
    norm = TwoSlopeNorm(0.0, vmin, vmax) if vmin < 0 < vmax else Normalize(vmin, vmax)
    colors = plt.cm.RdBu_r(norm(dk))
    x = np.arange(N)
    ymax = max(p_soft.max(), p_final.max()) * 1.15

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    ax = axes[0]
    ax.bar(x, dk, color=colors, width=0.7)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_title(r"difficulty scores $d_k$", fontsize=10)
    ax.set_ylabel(r"$d_k$")

    ax = axes[1]
    ax.bar(x, p_soft, color=colors, width=0.7)
    ax.axhline(uniform, color=GREY, ls="--", lw=1.0)
    ax.text(0.02, uniform, r" uniform $1/N$", fontsize=8, color=GREY,
            va="bottom", transform=ax.get_yaxis_transform(), bbox=LABEL_BOX)
    ax.set_ylim(0, ymax)
    ax.set_title(rf"softmax$(d_k/T)$,  $T={T:g}$", fontsize=10)
    ax.set_ylabel("prob.")

    ax = axes[2]
    ax.bar(x, p_final, color=colors, width=0.7)
    ax.axhline(floor, color=BLUE, ls="--", lw=1.2)
    ax.axhline(uniform, color=GREY, ls="--", lw=1.0)
    ax.text(0.02, floor, r" floor $\rho/N$", fontsize=8, color=BLUE,
            va="top", transform=ax.get_yaxis_transform(), bbox=LABEL_BOX)
    ax.text(0.02, uniform, r" uniform $1/N$", fontsize=8, color=GREY,
            va="bottom", transform=ax.get_yaxis_transform(), bbox=LABEL_BOX)
    ax.set_ylim(0, ymax)
    ax.set_title(rf"$p_k=(1-\rho)\,$softmax$+\rho/N$,  $\rho={rho:g}$", fontsize=10)
    ax.set_ylabel(r"$p_k$")

    for ax in axes:
        ax.set_xticks([])
        ax.set_xlabel(r"tasks (sorted easy $\rightarrow$ hard)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for x0, x1, label in [(0.352, 0.392, "sharpen"), (0.678, 0.718, "mix with uniform")]:
        fig.patches.append(FancyArrowPatch(
            (x0, 0.52), (x1, 0.52), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=16, color="#444", lw=1.4))
        fig.text((x0 + x1) / 2, 0.58, label, ha="center", fontsize=8, color="#444")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}  (N={N}, hardest p_k = {p_final.max()/uniform:.2f}x uniform, "
          f"floor = {floor:.4f})")


if __name__ == "__main__":
    main()
