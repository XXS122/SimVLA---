#!/usr/bin/env python
"""
plot_difficulty_dist.py
=======================
Two views of the offline difficulty score d_k over the LIBERO tasks, straight
from task_difficulty*.csv (columns: suite, task_name, d_k, ...):

  dk_hist.png     histogram of d_k -- difficulty is a continuous spectrum, with
                  a hard tail; bars coloured blue (easy) -> red (hard).
  dk_heatmap.png  a 4 x N "difficulty landscape": rows = suites, columns =
                  tasks sorted by d_k within each suite; red = hard, blue = easy.

Usage:
  python plot_difficulty_dist.py --difficulty task_difficulty_2comp.csv \
      --outdir figures_difficulty
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize

# suite display order (hardest first) + pretty names
SUITE_ORDER = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
SUITE_LABEL = {"libero_10": "LIBERO-Long", "libero_goal": "Goal",
               "libero_object": "Object", "libero_spatial": "Spatial"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True,
                    help="csv with columns suite, task_name, d_k")
    ap.add_argument("--outdir", default="figures_difficulty")
    args = ap.parse_args()

    d = pd.read_csv(args.difficulty)
    if "d_k" not in d.columns:
        raise SystemExit(f"{args.difficulty} has no 'd_k' column")
    dk = d["d_k"].to_numpy()
    os.makedirs(args.outdir, exist_ok=True)
    vmin, vmax = float(dk.min()), float(dk.max())
    # d_k is a z-score centred near 0; use a diverging norm when it straddles 0,
    # else fall back to a plain linear norm so the script never crashes.
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # ---------- 1) histogram: continuous difficulty spectrum ----------
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    counts, bins, patches = ax.hist(dk, bins=12, edgecolor="white", linewidth=0.6)
    for c, p in zip(0.5 * (bins[:-1] + bins[1:]), patches):
        p.set_facecolor(cmap(norm(c)))
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel(r"difficulty $d_k$")
    ax.set_ylabel("# tasks")
    ax.set_title("Task difficulty is a continuous spectrum", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out1 = os.path.join(args.outdir, "dk_hist.png")
    fig.savefig(out1, dpi=200); plt.close(fig)
    print("HIST:", out1)

    # ---------- 2) heatmap: difficulty landscape (suite x task) ----------
    if "suite" in d.columns:
        suites = [s for s in SUITE_ORDER if s in d["suite"].unique()]
        suites += [s for s in d["suite"].unique() if s not in suites]  # any extras
        rows = [np.sort(d.loc[d.suite == s, "d_k"].to_numpy())[::-1] for s in suites]
        ncol = max(len(r) for r in rows)
        M = np.full((len(suites), ncol), np.nan)
        for i, r in enumerate(rows):
            M[i, :len(r)] = r
        fig, ax = plt.subplots(figsize=(4.6, 0.55 * len(suites) + 1.0))
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_yticks(range(len(suites)))
        ax.set_yticklabels([SUITE_LABEL.get(s, s) for s in suites], fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel("tasks (sorted by difficulty within each suite)", fontsize=8)
        ax.set_title("Difficulty landscape across LIBERO", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$d_k$", fontsize=8)
        fig.tight_layout()
        out2 = os.path.join(args.outdir, "dk_heatmap.png")
        fig.savefig(out2, dpi=200); plt.close(fig)
        print("HEATMAP:", out2)
    else:
        print("[skip heatmap] no 'suite' column in", args.difficulty)


if __name__ == "__main__":
    main()
