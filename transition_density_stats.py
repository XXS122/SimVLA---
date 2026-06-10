#!/usr/bin/env python
"""
Transition-Density Statistics (TDS Experiment 0)
=================================================
Offline per-task difficulty scoring for LIBERO from the demo data alone.

For each task hdf5 (one file = one task, ~50 demos) we compute three
data-intrinsic components:
  E_events   - mean number of gripper open/close events per demo
  P_plateau  - mean fraction of low-speed "fine alignment" steps per demo
  L_length   - mean demo length (steps)
z-score each across tasks, combine into a difficulty score d_k, and turn
d_k into sampling weights p_k via a temperature softmax with a uniform floor.

Outputs:
  task_difficulty.csv  - per-task components, z-scores, d_k, p_k
  density_vs_sr.png    - (optional) d_k vs baseline SR scatter, Spearman test
                         (requires --sr_csv: two columns task_name, sr;
                          produced by libero_client.py --per_task_csv)

Usage:
  python transition_density_stats.py \
      --data_root $LIBERO_DATASETS \
      --suites libero_spatial libero_object libero_goal libero_10 \
      --temperature 2.0 --clip_rho 0.5 \
      [--sr_csv baseline_per_task_sr.csv]

The resulting csv plugs into training via --tds_weights_csv
(see datasets/tds_sampling.py and docs/tds_design.md).
"""

import argparse
import glob
import os

import h5py
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Core statistics
# ----------------------------------------------------------------------------

# Shared constants with the step-level weighting / BGE trigger: the plateau
# definition must stay identical across the three granularities (step / chunk
# / task) — that consistency is the point of the method.
PLATEAU_THRESH = 0.25   # low-speed plateau: normalized change rate < 0.25
PLATEAU_MIN_LEN = 3     # ... sustained for >= 3 steps
GRIPPER_DIM = -1        # LIBERO action layout: last dim is the gripper


def per_dim_normalized_change_rate(actions: np.ndarray) -> np.ndarray:
    """Per-dimension normalized change rate; keeps the gripper step function
    from dominating an L2 norm.

    actions: (T, d) action sequence of one episode (raw or normalized)
    returns: (T,)   per-step normalized change rate in [0, 1]
    """
    diff = np.abs(np.diff(actions, axis=0))            # (T-1, d)
    # Each dimension is scaled by its own max magnitude within the episode
    dim_max = diff.max(axis=0, keepdims=True)          # (1, d)
    diff_norm = diff / np.maximum(dim_max, 1e-6)       # (T-1, d) in [0,1]
    # Drop the gripper dim (it is counted separately as an event channel)
    arm = np.delete(diff_norm, GRIPPER_DIM, axis=1)
    c = arm.mean(axis=1)                               # (T-1,)
    c = np.concatenate([c[:1], c])                     # align length -> (T,)
    # Final episode-level normalization to [0,1]
    return c / max(c.max(), 1e-6)


def count_gripper_events(actions: np.ndarray) -> int:
    """Number of gripper open/close events = sign flips of the command.
    Median-centering makes this robust to both 0/1 and -1/1 encodings."""
    g = actions[:, GRIPPER_DIM]
    sign = np.sign(g - np.median(g))
    flips = np.abs(np.diff(sign)) > 0
    return int(flips.sum())


def plateau_ratio(c: np.ndarray) -> float:
    """Fraction of steps in low-speed plateaus: c < PLATEAU_THRESH sustained
    for at least PLATEAU_MIN_LEN consecutive steps."""
    below = c < PLATEAU_THRESH
    total, run = 0, 0
    for b in below:
        if b:
            run += 1
        else:
            if run >= PLATEAU_MIN_LEN:
                total += run
            run = 0
    if run >= PLATEAU_MIN_LEN:
        total += run
    return total / len(c)


def analyze_task_hdf5(path: str) -> dict:
    """Mean of the three components over all demos of one task hdf5."""
    events, plateaus, lengths = [], [], []
    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        for demo in demos:
            actions = f["data"][demo]["actions"][:]    # (T, d)
            c = per_dim_normalized_change_rate(actions)
            events.append(count_gripper_events(actions))
            plateaus.append(plateau_ratio(c))
            lengths.append(len(actions))
    return {
        "n_demos": len(demos),
        "E_events": float(np.mean(events)),      # component 1: gripper events
        "P_plateau": float(np.mean(plateaus)),   # component 2: plateau ratio
        "L_length": float(np.mean(lengths)),     # component 3: mean length
    }


# ----------------------------------------------------------------------------
# Difficulty score and sampling weights
# ----------------------------------------------------------------------------

def compute_difficulty_and_weights(df: pd.DataFrame,
                                   alpha=1/3, beta=1/3, gamma=1/3,
                                   temperature=2.0, clip_rho=0.5) -> pd.DataFrame:
    z = lambda s: (s - s.mean()) / max(s.std(), 1e-8)
    df["zE"], df["zP"], df["zL"] = z(df.E_events), z(df.P_plateau), z(df.L_length)
    df["d_k"] = alpha * df.zE + beta * df.zP + gamma * df.zL

    # Temperature softmax; shift to positive range first (a negative base
    # raised to 1/T would produce NaN)
    shifted = df.d_k - df.d_k.min() + 1.0
    w = shifted ** (1.0 / temperature)
    p = w / w.sum()

    # Uniform floor (protects easy tasks from catastrophic under-sampling),
    # then renormalize
    p_unif = 1.0 / len(df)
    p = np.maximum(p, clip_rho * p_unif)
    df["p_k"] = p / p.sum()
    df["p_over_uniform"] = df.p_k / p_unif   # >1 = up-weighted, <1 = down
    return df


# ----------------------------------------------------------------------------
# (optional) Experiment 0a: density vs SR correlation
# ----------------------------------------------------------------------------

def hypothesis_test(df: pd.DataFrame, sr_csv: str, out_png: str):
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = pd.read_csv(sr_csv)                       # columns: task_name, sr
    m = df.merge(sr, on="task_name")
    if len(m) < len(df):
        missing = set(df.task_name) - set(m.task_name)
        print(f"[warn] {len(missing)} tasks have no SR entry and are excluded "
              f"from the test, e.g. {sorted(missing)[:3]}")
    rho, pval = spearmanr(m.d_k, m.sr)
    print(f"\n[Exp 0a] Spearman rho = {rho:.3f}, p = {pval:.4f}")
    print("  pass criterion: rho < -0.4 and p < 0.05")
    print("  =>", "PASS — TDS hypothesis holds, proceed with plan A"
          if rho < -0.4 and pval < 0.05
          else "FAIL — consider falling back to the two-component design")

    # Per-component discriminative power (Experiment 0b)
    for col in ["E_events", "P_plateau", "L_length"]:
        r, p_ = spearmanr(m[col], m.sr)
        print(f"  component {col:10s}: rho = {r:+.3f} (p={p_:.3f})")
    # Component collinearity (Experiment 0c)
    print("\n[Exp 0c] component correlation matrix:")
    print(m[["E_events", "P_plateau", "L_length"]].corr(method="spearman").round(2))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = m.suite.astype("category").cat.codes if "suite" in m else None
    ax.scatter(m.d_k, m.sr, c=colors, cmap="tab10", s=40)
    coef = np.polyfit(m.d_k, m.sr, 1)
    xs = np.linspace(m.d_k.min(), m.d_k.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), "k--", lw=1)
    ax.set_xlabel("Transition density score $d_k$")
    ax.set_ylabel("Baseline success rate (%)")
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f} (p={pval:.3f})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"\nScatter plot saved: {out_png}")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.environ.get("LIBERO_DATASETS"),
                    help="LIBERO data root (default: $LIBERO_DATASETS from paths.env)")
    ap.add_argument("--suites", nargs="+",
                    default=["libero_spatial", "libero_object",
                             "libero_goal", "libero_10"])
    ap.add_argument("--alpha", type=float, default=1/3,
                    help="weight of the gripper-event component")
    ap.add_argument("--beta", type=float, default=1/3,
                    help="weight of the plateau-ratio component")
    ap.add_argument("--gamma", type=float, default=1/3,
                    help="weight of the length component")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--clip_rho", type=float, default=0.5)
    ap.add_argument("--sr_csv", default=None,
                    help="baseline per-task SR csv (task_name, sr); "
                         "if given, run Experiment 0a. Produce it with "
                         "evaluation/libero/libero_client.py --per_task_csv")
    ap.add_argument("--out", default="task_difficulty.csv")
    args = ap.parse_args()

    if not args.data_root:
        ap.error("--data_root not given and $LIBERO_DATASETS is unset "
                 "(did you `source paths.env`?)")

    rows = []
    for suite in args.suites:
        paths = sorted(glob.glob(os.path.join(args.data_root, suite, "*.hdf5")))
        if not paths:
            print(f"[warn] no hdf5 files under {args.data_root}/{suite}, skipping")
        for path in paths:
            # task_name = hdf5 stem (incl. `_demo`); this key must match the
            # meta datalist paths (tds_sampling.file_stem) and the per-task SR
            # csv from libero_client.py --per_task_csv
            task_name = os.path.splitext(os.path.basename(path))[0]
            print(f"[{suite}] {task_name} ...")
            stats = analyze_task_hdf5(path)
            rows.append({"suite": suite, "task_name": task_name, **stats})

    if not rows:
        raise SystemExit("No tasks found — check --data_root / --suites")

    df = pd.DataFrame(rows)
    df = compute_difficulty_and_weights(
        df, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        temperature=args.temperature, clip_rho=args.clip_rho)
    df.sort_values("d_k", ascending=False).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}; difficulty Top-5:")
    print(df.nlargest(5, "d_k")[["suite", "task_name", "d_k",
                                  "p_over_uniform"]].to_string(index=False))

    if args.sr_csv:
        hypothesis_test(df, args.sr_csv, out_png="density_vs_sr.png")


if __name__ == "__main__":
    main()
