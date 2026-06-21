#!/usr/bin/env python
"""
make_empirical_weights.py
=========================
Build "measured-difficulty" sampling weights — the model-coupled baseline
that TDS is compared against.

Whereas TDS derives task difficulty from the demonstration data alone
(offline, model-free), this baseline derives difficulty from the model's own
per-task success rate (i.e. it requires a full evaluation rollout loop — the
exact cost TDS avoids). difficulty_k = -SR_k, z-scored, then the *same*
temperature-softmax + uniform-floor pipeline as transition_density_stats.py,
so the only thing that differs from TDS is the *source* of difficulty.

If a model trained with these weights performs the same as TDS, that shows the
free data-driven difficulty matches the expensive measured difficulty.

Usage:
  python make_empirical_weights.py \
      --sr_csv evaluation/libero/u40k_sr_all.csv \
      --temperature 2.0 --clip_rho 0.5 \
      --out task_difficulty_empirical.csv
  # then train the baseline exactly like TDS, only swapping the csv:
  #   TDS_WEIGHTS_CSV=./task_difficulty_empirical.csv bash train_smolvlm_small.sh ...

Output columns match what datasets/tds_sampling.load_task_weights expects
(task_name, p_k).
"""

import argparse

import numpy as np
import pandas as pd


def weights_from_difficulty(d_k: np.ndarray, temperature: float,
                            clip_rho: float) -> np.ndarray:
    """Same softmax+floor as transition_density_stats.compute_difficulty_and_weights."""
    shifted = d_k - d_k.min() + 1.0
    w = shifted ** (1.0 / temperature)
    p = w / w.sum()
    p_unif = 1.0 / len(d_k)
    p = np.maximum(p, clip_rho * p_unif)
    return p / p.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr_csv", required=True,
                    help="per-task SR csv (task_name, sr) from a uniform run, "
                         "e.g. evaluation/libero/u40k_sr_all.csv")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--clip_rho", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true",
                    help="oversample EASY tasks instead (direction-control ablation)")
    ap.add_argument("--out", default="task_difficulty_empirical.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.sr_csv)
    if "sr" not in df or "task_name" not in df:
        raise SystemExit("sr_csv must have columns: task_name, sr")

    # difficulty = -SR (low success => hard => sampled more). z-score across tasks.
    sign = +1.0 if args.invert else -1.0   # invert => oversample easy (high SR)
    sr = df["sr"].to_numpy(dtype=float)
    d = sign * sr
    d_k = (d - d.mean()) / max(d.std(), 1e-8)

    df["d_k"] = d_k
    df["p_k"] = weights_from_difficulty(d_k, args.temperature, args.clip_rho)
    df["p_over_uniform"] = df["p_k"] * len(df)

    df.sort_values("d_k", ascending=False).to_csv(args.out, index=False)
    print(f"Wrote {args.out} "
          f"({'EASY-oversampling (inverted)' if args.invert else 'hard-oversampling'}); "
          f"top-5 by difficulty:")
    print(df.nlargest(5, "d_k")[["task_name", "sr", "p_over_uniform"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
