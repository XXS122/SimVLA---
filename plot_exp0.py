#!/usr/bin/env python
"""
plot_exp0.py
============
Stronger Experiment-0 figures than the raw d_k-vs-SR scatter.

Produces (given a difficulty csv + uniform per-task SR, and optionally TDS
per-task SR):
  1. exp0_scatter.png  - raw d_k vs baseline SR (reference; usually noisy).
  2. exp0_binned.png   - tasks binned by d_k, mean baseline SR +- SEM per bin.
                         Same data, far cleaner monotonic trend.
  3. exp0_gain.png     - d_k vs TDS gain (TDS_SR - uniform_SR). Shows the
                         method helps exactly the high-difficulty tasks it
                         upweights (ties Exp 0 to the mechanism).

Usage:
  python plot_exp0.py \
      --difficulty task_difficulty_2comp.csv \
      --uniform_sr evaluation/libero/u40k_sr_all.csv \
      --tds_sr evaluation/libero/tds40k_onA_sr_all.csv \
      --bins 5

Difficulty csv needs columns task_name, d_k (and optionally suite).
SR csvs need columns task_name, sr.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True, help="csv with task_name, d_k")
    ap.add_argument("--uniform_sr", required=True, help="csv with task_name, sr (uniform)")
    ap.add_argument("--tds_sr", default=None, help="csv with task_name, sr (TDS)")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    d = pd.read_csv(args.difficulty)[["task_name", "d_k"] +
                                     (["suite"] if "suite" in pd.read_csv(args.difficulty).columns else [])]
    u = pd.read_csv(args.uniform_sr).rename(columns={"sr": "uniform_sr"})
    m = d.merge(u, on="task_name")
    print(f"merged {len(m)} tasks")

    # ---- 1. raw scatter (reference) ----
    rho, p = spearmanr(m.d_k, m.uniform_sr)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    c = m.suite.astype("category").cat.codes if "suite" in m else None
    ax.scatter(m.d_k, m.uniform_sr, c=c, cmap="tab10", s=40)
    ax.set_xlabel("Transition density score $d_k$")
    ax.set_ylabel("Baseline success rate (%)")
    ax.set_title(f"Raw: Spearman $\\rho$={rho:.2f} (p={p:.3f})")
    fig.tight_layout(); fig.savefig("exp0_scatter.png", dpi=200)
    print(f"[1] exp0_scatter.png  rho={rho:.3f} p={p:.3f}")

    # ---- 2. binned: mean SR per difficulty bin ----
    m2 = m.copy()
    m2["bin"] = pd.qcut(m2.d_k, args.bins, labels=False, duplicates="drop")
    g = m2.groupby("bin").agg(dk=("d_k", "mean"),
                              sr=("uniform_sr", "mean"),
                              se=("uniform_sr", lambda s: s.std() / max(np.sqrt(len(s)), 1)),
                              n=("uniform_sr", "size"))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(g.dk, g.sr, yerr=g["se"], marker="o", capsize=4, lw=2, color="C0")
    ax.set_xlabel("Transition density score $d_k$ (bin mean)")
    ax.set_ylabel("Baseline success rate (%)")
    ax.set_title(f"Binned ({args.bins} groups, ~{int(g.n.mean())} tasks each)")
    fig.tight_layout(); fig.savefig("exp0_binned.png", dpi=200)
    print(f"[2] exp0_binned.png   bin means SR: {g.sr.round(1).tolist()}")

    # ---- 3. d_k vs TDS gain (mechanism) ----
    if args.tds_sr:
        t = pd.read_csv(args.tds_sr).rename(columns={"sr": "tds_sr"})
        mg = m.merge(t, on="task_name")
        mg["gain"] = mg.tds_sr - mg.uniform_sr
        rho_g, p_g = spearmanr(mg.d_k, mg.gain)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cc = mg.suite.astype("category").cat.codes if "suite" in mg else None
        ax.scatter(mg.d_k, mg.gain, c=cc, cmap="tab10", s=40)
        coef = np.polyfit(mg.d_k, mg.gain, 1)
        xs = np.linspace(mg.d_k.min(), mg.d_k.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "k--", lw=1)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlabel("Transition density score $d_k$")
        ax.set_ylabel("TDS gain (TDS $-$ uniform SR, %)")
        ax.set_title(f"Difficulty predicts where TDS helps: $\\rho$={rho_g:.2f} (p={p_g:.3f})")
        fig.tight_layout(); fig.savefig("exp0_gain.png", dpi=200)
        print(f"[3] exp0_gain.png     rho={rho_g:.3f} p={p_g:.3f} "
              f"(positive = TDS helps high-difficulty tasks most)")


if __name__ == "__main__":
    main()
