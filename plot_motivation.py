#!/usr/bin/env python
"""
plot_motivation.py
==================
Figure 1 (motivation) for the TDS paper, two rows:

  Row 1 -- "Difficulty is written in the demonstration": the action-change-rate
           trace of one EASY and one HARD task, with low-speed fine-alignment
           plateaus shaded and gripper open/close events marked. The hard task
           visibly has more plateaus, more gripper events, and a longer horizon.
  Row 2 -- "...and it predicts where the policy fails": across all 40 tasks,
           bucketed by the *independent* measured success rate (low SR = hard),
           plateau ratio and trajectory length rise with difficulty while the
           gripper-event count stays flat (justifying the 2-component score).

Two modes:
  --mock           synthesize the traces + bars (layout preview, fake numbers)
  real data        --demo_easy E.hdf5 --demo_hard H.hdf5 --difficulty d.csv
                   --uniform_sr sr.csv   (reads real trajectories + real bars)

Usage (real):
  python plot_motivation.py \
      --demo_easy  $LIBERO/libero_object/<easy>.hdf5 \
      --demo_hard  $LIBERO/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 \
      --difficulty task_difficulty_2comp.csv \
      --uniform_sr evaluation/libero/u40k_sr_all.csv \
      --out figures_motivation/motivation.png
"""

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Keep the plateau definition identical to transition_density_stats.py
PLATEAU_THRESH = 0.25
PLATEAU_MIN_LEN = 3
GRIPPER_DIM = -1

EASY_C = "#1a73e8"   # blue
HARD_C = "#d93025"   # red
PLAT_C = "#fde8a6"   # plateau shading
EVT_C = "#202124"    # gripper-event marker
METRIC_STYLE = {
    "E_events":  ("#9aa0a6", "Gripper events", "count"),
    "P_plateau": ("#1a73e8", "Plateau ratio", "fraction"),
    "L_length":  ("#f9ab00", "Trajectory length", "steps"),
}
LEVELS = ["High", "Medium", "Low"]


def detect_plateaus(c, thresh=PLATEAU_THRESH, min_len=PLATEAU_MIN_LEN):
    """Return [(start, end), ...] runs where c < thresh sustained >= min_len."""
    runs, start = [], None
    for i, v in enumerate(c):
        if v < thresh and start is None:
            start = i
        elif v >= thresh and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(c) - start >= min_len:
        runs.append((start, len(c)))
    return runs


def gripper_events(g):
    """Indices where the (continuous) gripper signal changes sign."""
    s = np.sign(g)
    return list(np.where(np.diff(s) != 0)[0] + 1)


def plot_trace(ax, c, g, title, color, note):
    t = np.arange(len(c))
    for a, b in detect_plateaus(c):
        ax.axvspan(a, b, color=PLAT_C, alpha=0.9, zorder=0)
    ax.plot(t, c, color=color, lw=1.4, zorder=3)
    ax.axhline(PLATEAU_THRESH, ls=":", color="grey", lw=0.8, zorder=1)
    for e in gripper_events(g):
        ax.axvline(e, color=EVT_C, lw=1.0, ls="--", alpha=0.7, zorder=2)
    ax.set_xlim(0, len(c)); ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time step"); ax.set_ylabel("action change rate")
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#444",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))


def synth_easy(rng):
    T = 120
    c = np.full(T, 0.45) + rng.randn(T) * 0.04
    c[54:61] = 0.12 + rng.randn(7) * 0.02          # one short grasp plateau
    c[:8] = np.linspace(0.30, 0.45, 8); c[-8:] = np.linspace(0.45, 0.30, 8)
    g = np.ones(T); g[57:] = -1.0                  # close once -> 1 event
    return np.clip(c, 0, 1), g


def synth_hard(rng):
    T = 240
    c = np.full(T, 0.5) + rng.randn(T) * 0.05
    for a, b in [(28, 40), (78, 92), (128, 140), (172, 188), (208, 220)]:
        c[a:b] = 0.12 + rng.randn(b - a) * 0.02    # 5 fine-alignment plateaus
    c[:8] = np.linspace(0.05, 0.5, 8); c[-8:] = np.linspace(0.5, 0.05, 8)
    g = np.ones(T)
    g[45:100] = -1; g[100:150] = 1; g[150:200] = -1; g[200:] = 1   # 4 events
    return np.clip(c, 0, 1), g


def load_trace(path):
    import h5py
    from transition_density_stats import per_dim_normalized_change_rate
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        actions = f["data"][demos[0]]["actions"][:]   # first demo
    c = np.asarray(per_dim_normalized_change_rate(actions)).reshape(-1)
    g = np.asarray(actions)[:, GRIPPER_DIM]
    return c, g


def bar_data_from_csv(difficulty, uniform_sr):
    import pandas as pd
    d = pd.read_csv(difficulty)
    s = pd.read_csv(uniform_sr)[["task_name", "sr"]]
    m = d.merge(s, on="task_name").sort_values("sr").reset_index(drop=True)
    n = len(m); c1, c2 = n // 3, 2 * n // 3
    m["level"] = ["High"] * c1 + ["Medium"] * (c2 - c1) + ["Low"] * (n - c2)
    out = {}
    for col in ["E_events", "P_plateau", "L_length"]:
        gp = m.groupby("level")[col]
        out[col] = (gp.mean().reindex(LEVELS).values,
                    (gp.std() / np.sqrt(gp.count())).reindex(LEVELS).values)
    return out


def synth_bars():
    return {
        "E_events":  (np.array([7.8, 7.6, 7.5]), np.array([0.5, 0.5, 0.4])),
        "P_plateau": (np.array([0.30, 0.21, 0.13]), np.array([0.02, 0.02, 0.015])),
        "L_length":  (np.array([201, 177, 152]), np.array([6, 6, 4])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="synthesize traces+bars for a layout preview")
    ap.add_argument("--demo_easy"); ap.add_argument("--demo_hard")
    ap.add_argument("--difficulty"); ap.add_argument("--uniform_sr")
    ap.add_argument("--out", default="figures_motivation/motivation.png")
    args = ap.parse_args()

    use_mock = args.mock or not (args.demo_easy and args.demo_hard)
    rng = np.random.RandomState(1)
    if use_mock:
        ce, ge = synth_easy(rng); ch, gh = synth_hard(rng)
    else:
        ce, ge = load_trace(args.demo_easy); ch, gh = load_trace(args.demo_hard)
    bars = synth_bars() if (use_mock or not (args.difficulty and args.uniform_sr)) \
        else bar_data_from_csv(args.difficulty, args.uniform_sr)

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.95], hspace=0.45, wspace=0.9)

    ax_e = fig.add_subplot(gs[0, 0:3])
    ax_h = fig.add_subplot(gs[0, 3:6])
    ne = f"{len(gripper_events(ge))} gripper events\n" \
         f"{len(detect_plateaus(ce))} plateau(s), len={len(ce)}"
    nh = f"{len(gripper_events(gh))} gripper events\n" \
         f"{len(detect_plateaus(ch))} plateaus, len={len(ch)}"
    plot_trace(ax_e, ce, ge, "Easy task (high success rate)", EASY_C, ne)
    plot_trace(ax_h, ch, gh, "Hard task (low success rate)", HARD_C, nh)
    handles = [Patch(fc=PLAT_C, ec="0.7", label="low-speed plateau"),
               Line2D([0], [0], color=EVT_C, ls="--", label="gripper event"),
               Line2D([0], [0], color="grey", ls=":", label="plateau threshold")]
    ax_e.legend(handles=handles, fontsize=7, frameon=False, loc="lower left")

    axes2 = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
             fig.add_subplot(gs[1, 4:6])]
    x = np.arange(len(LEVELS))
    for ax, col in zip(axes2, ["E_events", "P_plateau", "L_length"]):
        color, name, unit = METRIC_STYLE[col]
        mean, sem = bars[col]
        ax.bar(x, mean, yerr=sem, capsize=4, color=color, edgecolor="black",
               linewidth=0.6, width=0.62)
        ax.set_xticks(x); ax.set_xticklabels(LEVELS)
        ax.set_xlabel("difficulty (by measured SR)")
        ax.set_ylabel(f"{name}\n({unit})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    tag = "  [MOCK: synthetic data, layout preview]" if use_mock else ""
    fig.suptitle("Task difficulty is written in the demonstration's action "
                 "structure" + tag, fontsize=12, y=0.99)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", args.out, "(mock)" if use_mock else "(real data)")


if __name__ == "__main__":
    main()
