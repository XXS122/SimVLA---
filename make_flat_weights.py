#!/usr/bin/env python
"""
make_flat_weights.py
====================
Generate a FLAT (uniform) sampling-weights csv for the "uniform-interleaved"
control.

TDS differs from the original SimVLA uniform baseline in two ways at once:
  (1) the sampling MECHANISM — TDS draws a task per sample (two-level,
      interleaved), whereas the original reader streams each task file's
      samples in a block;
  (2) the difficulty WEIGHTS.

Feeding this flat csv (all p_k = 1/N) to TDS_WEIGHTS_CSV triggers the TDS
two-level sampler but with uniform weights, i.e. uniform sampling with the
*interleaved* mechanism. Training with it isolates the two effects:
  flat vs TDS              -> contribution of the difficulty weights (the point)
  flat vs original uniform -> whether interleaving alone matters

If flat ~= original uniform, the TDS gain is attributable to the weights, not
to better shuffling.

Usage:
  python make_flat_weights.py --like task_difficulty_2comp.csv \
      --out task_difficulty_flat.csv
  # then train exactly like TDS, only the csv differs:
  #   TDS_WEIGHTS_CSV=./task_difficulty_flat.csv bash train_smolvlm_small.sh ...
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True,
                    help="an existing weights csv to copy the task_name column from "
                         "(e.g. task_difficulty_2comp.csv)")
    ap.add_argument("--out", default="task_difficulty_flat.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.like)
    if "task_name" not in df:
        raise SystemExit("--like csv must have a task_name column")

    n = len(df)
    df["d_k"] = 0.0
    df["p_k"] = 1.0 / n
    df["p_over_uniform"] = 1.0
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {n} tasks, all p_k = {1.0/n:.5f} "
          f"(uniform weights; TDS sampler -> interleaved-uniform control)")


if __name__ == "__main__":
    main()
