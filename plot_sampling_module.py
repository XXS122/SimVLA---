#!/usr/bin/env python
"""
plot_sampling_module.py
=======================
Three standalone schematic bar charts for the "Sampling Distribution" module
(arrows / knobs / frame are added by hand in the drawing tool):

  1_dk.png        difficulty d_k       (10 bars, negatives below zero line)
  2_softmax.png   after softmax        (uniform 1/N dashed line)
  3_final_pk.png  final p_k            (uniform + floor rho/N dashed lines)

Same 10 tasks, same blue->red diverging colors in all three charts; only the
bar heights change. Saved with transparent background for easy compositing.

Usage:
  python plot_sampling_module.py --outdir figures_method
  python plot_sampling_module.py --temperature 2 --rho 0.5 --outdir figures_method
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY = "#9aa0a6"
BLUE = "#1a73e8"

# 10-bar schematic difficulty profile: 6 easy (negative), 1 neutral, 3 hard tail
DK = np.array([-1.1, -0.9, -0.7, -0.5, -0.3, -0.15, 0.0, 0.6, 1.4, 2.2])


def style(ax, x):
    ax.set_xticks(x)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("", xy=(0.78, -0.26), xytext=(0.22, -0.26),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.text(0.10, -0.26, "easier", transform=ax.transAxes,
            ha="center", va="center", fontsize=9)
    ax.text(0.91, -0.26, "harder", transform=ax.transAxes,
            ha="center", va="center", fontsize=9)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--outdir", default="figures_method")
    args = ap.parse_args()
    T, rho = args.temperature, args.rho

    dk = DK
    N = len(dk)
    z = np.exp(dk / T)
    p_soft = z / z.sum()
    p_final = (1 - rho) * p_soft + rho / N
    uniform, floor = 1.0 / N, rho / N

    # Diverging blue->red keyed to d_k, identical across the three charts
    vmax = max(abs(dk.min()), abs(dk.max()))
    colors = plt.cm.RdBu_r(0.5 + dk / (2 * vmax))
    x = np.arange(1, N + 1)
    ymax = max(p_soft.max(), p_final.max()) * 1.18
    os.makedirs(args.outdir, exist_ok=True)

    # ---- 1. difficulty d_k ----
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(x, dk, color=colors, width=0.72, edgecolor="white", lw=0.4)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title(r"difficulty $d_k$", fontsize=12)
    ax.set_ylabel("value")
    style(ax, x)
    fig.tight_layout()
    save(fig, os.path.join(args.outdir, "1_dk.png"))

    # ---- 2. after softmax ----
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(x, p_soft, color=colors, width=0.72, edgecolor="white", lw=0.4)
    ax.axhline(uniform, color=GREY, ls="--", lw=1.2)
    ax.text(0.03, uniform * 1.05, r"uniform $1/N$", fontsize=9, color=GREY,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, ymax)
    ax.set_title("after softmax", fontsize=12)
    ax.set_ylabel("probability")
    style(ax, x)
    fig.tight_layout()
    save(fig, os.path.join(args.outdir, "2_softmax.png"))

    # ---- 3. final p_k ----
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(x, p_final, color=colors, width=0.72, edgecolor="white", lw=0.4)
    ax.axhline(uniform, color=GREY, ls="--", lw=1.2)
    ax.axhline(floor, color=BLUE, ls="--", lw=1.4)
    ax.text(0.03, uniform * 1.05, r"uniform $1/N$", fontsize=9, color=GREY,
            transform=ax.get_yaxis_transform())
    ax.text(0.03, floor * 1.10, r"floor $\rho/N$", fontsize=9, color=BLUE,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(0, ymax)
    ax.set_title(r"final $p_k$", fontsize=12)
    ax.set_ylabel("probability")
    style(ax, x)
    fig.tight_layout()
    save(fig, os.path.join(args.outdir, "3_final_pk.png"))

    print(f"(schematic N={N}, hardest = {p_final.max()/uniform:.2f}x uniform)")


if __name__ == "__main__":
    main()
