"""
Stage 0b: build low-data splits at DEMO level (not file level).

Each LIBERO HDF5 file holds ~50 demos; a p10 split keeps 10% of demos per
file (>=1), so every task stays represented — the split measures data
efficiency, not task coverage. The split is stored as a "demos" whitelist
per datalist item; LiberoHDF5Handler filters on it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np


def get_args_parser():
    p = argparse.ArgumentParser("low-data splits", add_help=False)
    p.add_argument("--meta_path", type=str, required=True, help="full meta JSON")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.1])
    p.add_argument("--output_dir", type=str, default=None,
                   help="default: alongside meta_path")
    p.add_argument("--seed", type=int, default=0)
    return p


def frac_tag(frac: float) -> str:
    return f"p{frac * 100:g}".replace(".", "_")  # 0.01 -> p1, 0.1 -> p10


def main(args):
    with open(args.meta_path) as f:
        meta = json.load(f)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.meta_path).parent

    demo_lists = []
    for item in meta["datalist"]:
        with h5py.File(item["path"], "r") as f:
            keys = sorted(k for k in f["data"].keys() if k.startswith("demo")) \
                if "data" in f else []
        demo_lists.append(keys)

    for frac in args.fractions:
        rng = np.random.RandomState(args.seed)
        sub = copy.deepcopy(meta)
        total = 0
        for item, keys in zip(sub["datalist"], demo_lists):
            if not keys:
                item["demos"] = []
                continue
            n = max(1, math.ceil(frac * len(keys)))
            picked = sorted(rng.choice(keys, size=n, replace=False).tolist())
            item["demos"] = picked
            item["num_demos"] = len(picked)
            total += len(picked)
        sub["num_episodes"] = total
        sub["split_fraction"] = frac
        sub["split_seed"] = args.seed
        tag = frac_tag(frac)
        out = out_dir / (Path(args.meta_path).stem + f"_{tag}.json")
        with open(out, "w") as f:
            json.dump(sub, f, indent=2)
        print(f"{tag}: {total} demos -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("low-data splits", parents=[get_args_parser()])
    main(parser.parse_args())
