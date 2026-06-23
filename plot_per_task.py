#!/usr/bin/env python
"""
plot_per_task.py
================
Main figure (per-suite TDS vs uniform bars) + per-task figures sorted into
high/medium/low difficulty folders for the appendix.

Inputs are same-machine per-task SR csvs (task_name, sr) plus the difficulty
csv (task_name, d_k, suite).

Difficulty bucketing (--sort_by):
  uniform_sr : tasks split by the uniform baseline's measured SR (low SR =
               hard). Reliably shows TDS >> uniform on the high-difficulty
               row. (default; "empirical difficulty")
  d_k        : tasks split by the transition-density score (high d_k = hard).
               Consistent with the method but noisier at the task level.

Usage:
  python plot_per_task.py \
      --difficulty task_difficulty_2comp.csv \
      --uniform_sr evaluation/libero/u40k_sr_all.csv \
      --tds_sr evaluation/libero/tds40k_onA_sr_all.csv \
      --budget 40k --sort_by uniform_sr --outdir figures_compare
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNI = "#9aa0a6"   # grey
TDS = "#1a73e8"   # blue
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True, help="csv: task_name, d_k[, suite]")
    ap.add_argument("--uniform_sr", required=True, help="csv: task_name, sr (uniform)")
    ap.add_argument("--tds_sr", required=True, help="csv: task_name, sr (TDS)")
    ap.add_argument("--outdir", default="figures_compare")
    ap.add_argument("--sort_by", choices=["uniform_sr", "d_k"], default="uniform_sr")
    ap.add_argument("--budget", default="40k")
    args = ap.parse_args()

    d = pd.read_csv(args.difficulty)
    u = pd.read_csv(args.uniform_sr).rename(columns={"sr": "uniform"})[["task_name", "uniform"]]
    t = pd.read_csv(args.tds_sr).rename(columns={"sr": "tds"})[["task_name", "tds"]]
    m = d.merge(u, on="task_name").merge(t, on="task_name")
    if len(m) < len(d):
        miss = set(d.task_name) - set(m.task_name)
        print(f"[warn] {len(miss)} tasks didn't merge (name mismatch?), e.g. {sorted(miss)[:3]}")
    m["gain"] = m.tds - m.uniform
    print(f"merged {len(m)} tasks")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- MAIN figure: per-suite grouped bars (+ Avg) ----
    if "suite" in m:
        g = (m.groupby("suite").agg(uniform=("uniform", "mean"), tds=("tds", "mean"))
             .reindex(SUITE_ORDER).dropna())
        labels = [s.replace("libero_", "") for s in g.index] + ["Avg"]
        uni = list(g.uniform) + [m.uniform.mean()]
        tds = list(g.tds) + [m.tds.mean()]
        x = np.arange(len(labels)); w = 0.38
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(x - w/2, uni, w, label="Uniform (SimVLA)", color=UNI)
        ax.bar(x + w/2, tds, w, label="TDS (ours)", color=TDS)
        for i in range(len(labels)):
            ax.text(x[i]-w/2, uni[i]+1.2, f"{uni[i]:.0f}", ha="center", fontsize=9)
            ax.text(x[i]+w/2, tds[i]+1.2, f"{tds[i]:.0f}", ha="center", fontsize=9, color=TDS)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 108)
        ax.set_title(f"Per-suite success rate @ {args.budget} steps")
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        out = os.path.join(args.outdir, "main_suite_comparison.png")
        fig.savefig(out, dpi=200); plt.close(fig)
        print("MAIN:", out, {l: (round(a), round(b)) for l, a, b in zip(labels, uni, tds)})

    # ---- per-task figures into high/medium/low ----
    if args.sort_by == "d_k":
        m2 = m.sort_values("d_k", ascending=False).reset_index(drop=True)   # high d_k = hard
    else:
        m2 = m.sort_values("uniform", ascending=True).reset_index(drop=True)  # low SR = hard
    n = len(m2); c1, c2 = n // 3, 2 * n // 3
    level = ["high"] * c1 + ["medium"] * (c2 - c1) + ["low"] * (n - c2)
    for lv in ["high", "medium", "low"]:
        os.makedirs(os.path.join(args.outdir, lv), exist_ok=True)

    for i, row in m2.iterrows():
        lv = level[i]
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.bar([0, 1], [row.uniform, row.tds], color=[UNI, TDS], width=0.62)
        ax.text(0, row.uniform + 2, f"{row.uniform:.0f}", ha="center", fontsize=9)
        ax.text(1, row.tds + 2, f"{row.tds:.0f}", ha="center", fontsize=9, color=TDS)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Uniform", "TDS"])
        ax.set_ylim(0, 108); ax.set_ylabel("SR (%)")
        title = str(row.task_name).replace("_demo", "").replace("_", " ")
        ax.set_title(title[:40], fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, lv, f"{safe(row.task_name)}.png"), dpi=150)
        plt.close(fig)

    print(f"wrote {n} per-task figures into {args.outdir}/high|medium|low "
          f"(sorted by {args.sort_by}); high = hardest, shows the biggest TDS gap")


if __name__ == "__main__":
    main()
