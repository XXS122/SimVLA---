#!/usr/bin/env python
"""
plot_curves.py
==============
Multi-budget figures using the full per-task SR sweep (e.g. 20k..100k).

Main figure : per-suite efficiency curves (2x2 panels + overall Avg),
              uniform vs TDS, SR-vs-steps.
Per-task    : one line figure per task (uniform vs TDS across budgets),
              sorted into high/medium/low difficulty folders for the appendix.

Pass the per-budget sr_all csvs (task_name, sr) in step order for each method.

Usage:
  python plot_curves.py \
      --difficulty task_difficulty_2comp.csv \
      --steps 20 40 60 80 100 \
      --uniform_sr u20k_sr_all.csv u40k_sr_all.csv u60k_sr_all.csv u80k_sr_all.csv u100k_sr_all.csv \
      --tds_sr tds20k_sr_all.csv tds40k_sr_all.csv tds60k_sr_all.csv tds80k_sr_all.csv tds100k_sr_all.csv \
      --sort_by uniform_sr --outdir figures_curves
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNI = "#9aa0a6"
TDS = "#1a73e8"
SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True)
    ap.add_argument("--steps", nargs="+", type=int, required=True, help="e.g. 20 40 60 80 100 (k steps)")
    ap.add_argument("--uniform_sr", nargs="+", required=True, help="uniform sr_all csvs in step order")
    ap.add_argument("--tds_sr", nargs="+", required=True, help="TDS sr_all csvs in step order")
    ap.add_argument("--sort_by", choices=["uniform_sr", "d_k"], default="uniform_sr")
    ap.add_argument("--outdir", default="figures_curves")
    args = ap.parse_args()

    assert len(args.steps) == len(args.uniform_sr) == len(args.tds_sr), \
        "steps / uniform_sr / tds_sr must have the same length"

    d = pd.read_csv(args.difficulty)
    keep = ["task_name", "d_k"] + (["suite"] if "suite" in d.columns else [])
    d = d[keep]

    recs = []
    for step, uf, tf in zip(args.steps, args.uniform_sr, args.tds_sr):
        u = pd.read_csv(uf).rename(columns={"sr": "uniform"})[["task_name", "uniform"]]
        t = pd.read_csv(tf).rename(columns={"sr": "tds"})[["task_name", "tds"]]
        mm = d.merge(u, on="task_name").merge(t, on="task_name")
        mm["step"] = step
        recs.append(mm)
    L = pd.concat(recs, ignore_index=True)
    print(f"loaded {L.step.nunique()} budgets x {L.task_name.nunique()} tasks")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- MAIN: per-suite efficiency curves + overall Avg ----
    if "suite" in L:
        suites = [s for s in SUITE_ORDER if s in L.suite.unique()]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
        axes = axes.flat
        for ax, s in zip(axes, suites):
            sub = L[L.suite == s].groupby("step").agg(u=("uniform", "mean"), t=("tds", "mean"))
            ax.plot(sub.index, sub.u, "o-", color=UNI, lw=2, label="Uniform")
            ax.plot(sub.index, sub.t, "s-", color=TDS, lw=2, label="TDS (ours)")
            ax.set_title(s.replace("libero_", ""), fontsize=11)
            ax.grid(alpha=0.3)
        # overall avg
        ax = axes[len(suites)]
        sub = L.groupby("step").agg(u=("uniform", "mean"), t=("tds", "mean"))
        ax.plot(sub.index, sub.u, "o-", color=UNI, lw=2, label="Uniform")
        ax.plot(sub.index, sub.t, "s-", color=TDS, lw=2, label="TDS (ours)")
        ax.set_title("Average", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(frameon=False)
        for ax in axes[len(suites) + 1:]:
            ax.axis("off")
        fig.supxlabel("Training steps (k)"); fig.supylabel("Success rate (%)")
        fig.tight_layout()
        out = os.path.join(args.outdir, "main_suite_curves.png")
        fig.savefig(out, dpi=200); plt.close(fig)
        print("MAIN:", out)

    # ---- difficulty bucketing ----
    if args.sort_by == "d_k":
        rank = d.set_index("task_name").d_k          # high = hard
        order = rank.sort_values(ascending=False).index.tolist()
    else:
        rank = L.groupby("task_name").uniform.mean()  # low mean SR = hard
        order = rank.sort_values(ascending=True).index.tolist()
    n = len(order); c1, c2 = n // 3, 2 * n // 3
    level = {t: ("high" if i < c1 else "medium" if i < c2 else "low")
             for i, t in enumerate(order)}
    for lv in ["high", "medium", "low"]:
        os.makedirs(os.path.join(args.outdir, lv), exist_ok=True)

    # ---- per-task efficiency curves ----
    for task, sub in L.groupby("task_name"):
        sub = sub.sort_values("step")
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        ax.plot(sub.step, sub.uniform, "o-", color=UNI, lw=1.8, label="Uniform")
        ax.plot(sub.step, sub.tds, "s-", color=TDS, lw=1.8, label="TDS")
        ax.set_ylim(0, 105); ax.grid(alpha=0.3)
        ax.set_xlabel("steps (k)"); ax.set_ylabel("SR (%)")
        title = str(task).replace("_demo", "").replace("_", " ")
        ax.set_title(title[:38], fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, level[task], f"{safe(task)}.png"), dpi=150)
        plt.close(fig)

    counts = {lv: sum(v == lv for v in level.values()) for lv in ["high", "medium", "low"]}
    print(f"wrote per-task curves into {args.outdir}/high|medium|low {counts} "
          f"(sorted by {args.sort_by}; high = hardest)")


if __name__ == "__main__":
    main()
