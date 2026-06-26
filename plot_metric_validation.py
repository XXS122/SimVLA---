#!/usr/bin/env python
"""
plot_metric_validation.py
=========================
Validate that the three data-intrinsic difficulty signals used by TDS actually
track task difficulty. We bucket the 40 LIBERO tasks into high/medium/low
difficulty by an *independent* ground truth -- the uniform baseline's measured
per-task success rate (low SR = hard) -- and show how each signal behaves
across the buckets.

Bucketing by measured SR (not by d_k) is deliberate: d_k is computed *from*
these signals, so bucketing by d_k would be circular. Sorting by the
independently-measured SR makes this a genuine validation.

The three signals (already computed per task by transition_density_stats.py and
stored in task_difficulty*.csv):
  E_events   gripper open/close events  (count)
  P_plateau  low-speed fine-alignment plateau ratio  (fraction)
  L_length   trajectory length  (steps)

On LIBERO, P_plateau and L_length rise with difficulty while E_events is flat
(no signal) -- which is exactly why the adopted 2-component score drops E
(alpha=0). This figure shows that honestly.

Outputs (into --outdir):
  metric_by_difficulty_panels.png   1x3 raw-unit panels, one per metric (main)
  metric_by_difficulty_grouped.png  single grouped-bar, metrics mean-normalised

Usage:
  python plot_metric_validation.py \
      --difficulty task_difficulty_2comp.csv \
      --uniform_sr evaluation/libero/u40k_sr_all.csv \
      --sort_by uniform_sr --outdir figures_metrics
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-metric colours (consistent with the grey/blue palette of the other plots)
METRIC_STYLE = {
    "E_events":  ("#9aa0a6", "Gripper events", "count"),
    "P_plateau": ("#1a73e8", "Plateau ratio", "fraction"),
    "L_length":  ("#f9ab00", "Trajectory length", "steps"),
}
LEVELS = ["high", "medium", "low"]          # high = hardest (left)
LEVEL_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


def spearman(x, y):
    """Spearman rho and p-value; falls back to a p-less estimate if no scipy."""
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        rho = pd.Series(x).corr(pd.Series(y), method="spearman")
        return float(rho), float("nan")


def assign_levels(m, sort_by):
    """Return m with a 'level' column (high/medium/low), high = hardest."""
    if sort_by == "d_k":
        if "d_k" not in m.columns:
            raise SystemExit("--sort_by d_k needs a 'd_k' column in --difficulty")
        ordered = m.sort_values("d_k", ascending=False).reset_index(drop=True)
    else:
        ordered = m.sort_values("sr", ascending=True).reset_index(drop=True)
    n = len(ordered)
    c1, c2 = n // 3, 2 * n // 3
    lv = ["high"] * c1 + ["medium"] * (c2 - c1) + ["low"] * (n - c2)
    ordered["level"] = lv
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", required=True,
                    help="csv with task_name + E_events/P_plateau/L_length (+ d_k)")
    ap.add_argument("--uniform_sr", required=True,
                    help="uniform-40k per-task SR csv: task_name, sr")
    ap.add_argument("--sort_by", choices=["uniform_sr", "d_k"], default="uniform_sr",
                    help="how to bucket difficulty (default: independent measured SR)")
    ap.add_argument("--metrics", nargs="+",
                    default=["P_plateau", "L_length"])
    ap.add_argument("--outdir", default="figures_metrics")
    args = ap.parse_args()

    d = pd.read_csv(args.difficulty)
    missing = [c for c in args.metrics if c not in d.columns]
    if missing:
        raise SystemExit(f"--difficulty {args.difficulty} is missing columns {missing}; "
                         f"regenerate it with transition_density_stats.py")
    s = pd.read_csv(args.uniform_sr).rename(columns={"sr": "sr"})[["task_name", "sr"]]

    m = d.merge(s, on="task_name")
    if len(m) < min(len(d), len(s)):
        miss = (set(d.task_name) ^ set(s.task_name))
        print(f"[warn] {len(miss)} task names didn't merge (name mismatch?), "
              f"e.g. {sorted(miss)[:3]}")
    print(f"merged {len(m)} tasks; bucketing by {args.sort_by}")
    os.makedirs(args.outdir, exist_ok=True)

    m = assign_levels(m, args.sort_by)
    counts = m.level.value_counts().reindex(LEVELS)
    print("group sizes:", {lv: int(counts[lv]) for lv in LEVELS})

    # Per-group mean + SEM, and Spearman rho(metric vs SR) over all tasks.
    stats = {}
    for col in args.metrics:
        g = m.groupby("level")[col]
        mean = g.mean().reindex(LEVELS)
        sem = (g.std() / np.sqrt(g.count())).reindex(LEVELS)
        rho, p = spearman(m[col], m["sr"])
        stats[col] = dict(mean=mean, sem=sem, rho=rho, p=p)
        print(f"  {col:10s}: high/med/low mean = "
              f"{mean['high']:.3g}/{mean['medium']:.3g}/{mean['low']:.3g}  "
              f"rho(vs SR)={rho:+.3f} (p={p:.3f})")

    x = np.arange(len(LEVELS))
    xt = [LEVEL_LABEL[lv] for lv in LEVELS]

    # ---- MAIN: 1xK raw-unit panels (one per metric) ----
    K = len(args.metrics)
    fig, axes = plt.subplots(1, K, figsize=(3.4 * K, 3.4))
    if K == 1:
        axes = [axes]
    for ax, col in zip(axes, args.metrics):
        color, name, unit = METRIC_STYLE.get(col, ("#1a73e8", col, ""))
        st = stats[col]
        ax.bar(x, st["mean"].values, yerr=st["sem"].values, capsize=4,
               color=color, edgecolor="black", linewidth=0.6, width=0.62)
        ax.set_xticks(x); ax.set_xticklabels(xt)
        ax.set_xlabel("Task difficulty (by measured SR)")
        ax.set_ylabel(f"{name} ({unit})")
        prho = "" if np.isnan(st["p"]) else f", $p$={st['p']:.3f}"
        ax.set_title(f"{name}\nSpearman $\\rho$={st['rho']:+.2f}{prho}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(args.outdir, "metric_by_difficulty_panels.png")
    fig.savefig(out1, dpi=200); plt.close(fig)
    print("MAIN:", out1)

    # ---- ALT: single grouped bar, metrics mean-normalised to a common scale ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.8 / len(args.metrics)
    for j, col in enumerate(args.metrics):
        color, name, _ = METRIC_STYLE.get(col, ("#1a73e8", col, ""))
        grand = m[col].mean()
        norm = grand if abs(grand) > 1e-12 else 1.0
        mean = (stats[col]["mean"] / norm).values
        sem = (stats[col]["sem"] / norm).values
        ax.bar(x + (j - (len(args.metrics) - 1) / 2) * w, mean, w, yerr=sem,
               capsize=3, color=color, edgecolor="black", linewidth=0.5,
               label=name)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(xt)
    ax.set_xlabel("Task difficulty (by measured SR)")
    ax.set_ylabel("Signal value (relative to dataset mean)")
    ax.set_title("Data-intrinsic difficulty signals vs. measured difficulty")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(args.outdir, "metric_by_difficulty_grouped.png")
    fig.savefig(out2, dpi=200); plt.close(fig)
    print("ALT :", out2)

    print(f"\nDone. high = hardest (lowest measured SR). P_plateau / L_length should "
          f"rise toward 'High'; E_events is expected to stay roughly flat on LIBERO "
          f"(justifying alpha=0).")


if __name__ == "__main__":
    main()
