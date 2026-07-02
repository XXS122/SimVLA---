"""
Stage 1c: affine probe between latent actions z and ground-truth actions.

This is the go/no-go gate of the project:
  - fits ridge regression z -> a_norm on train episodes, reports per-dim and
    mean R^2 on held-out episodes (identifiability up to affine => high R^2);
  - fits the reverse map a_norm -> z and saves it as the adapter init
    (probe.npz) consumed by LiberoZAdapterActionSpace during fine-tuning.

Actions are normalized with the same norm-stats JSON used for VLA training,
so the probe target space matches the fine-tuning action space exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from latent_action.data import iter_demos, load_meta


def get_args_parser():
    p = argparse.ArgumentParser("affine probe", add_help=False)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--z_labels", type=str, required=True)
    p.add_argument("--norm_stats_path", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="probe .npz output")
    p.add_argument("--report_out", type=str, default=None)
    p.add_argument("--ridge_lambda", type=float, default=1e-3)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p


def _load_action_norm(norm_stats_path: str):
    with open(norm_stats_path) as f:
        data = json.load(f)
    if "norm_stats" in data:
        data = data["norm_stats"]
    stats = data["actions"]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return mean, std


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float):
    """Solve min ||Xw - Y||^2 + lam||w||^2 with bias. Returns (W, b)."""
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    d = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(d, dtype=X.dtype)
    W = np.linalg.solve(A, Xb.T @ Y)  # [d, out]
    return W[:-1], W[-1]


def _r2(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
    sse = ((Y - Y_hat) ** 2).sum(axis=0)
    sst = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    return 1.0 - sse / sst


def main(args):
    rng = np.random.RandomState(args.seed)
    meta = load_meta(args.meta_path)
    a_mean, a_std = _load_action_norm(args.norm_stats_path)

    Z_tr, A_tr, Z_va, A_va = [], [], [], []
    n_missing = 0
    with h5py.File(args.z_labels, "r") as zf:
        stride = int(zf.attrs.get("stride", 1))
        for rec in iter_demos(meta):
            key = rec["key"]
            if key not in zf:
                n_missing += 1
                continue
            z = np.asarray(zf[key], dtype=np.float32)          # [T - stride, z_dim]
            a = rec["actions"][: len(z)].astype(np.float32)    # aligned a_t
            a = (a - a_mean) / (a_std + 1e-6)
            if rng.rand() < args.val_fraction:                 # episode-level split
                Z_va.append(z); A_va.append(a)
            else:
                Z_tr.append(z); A_tr.append(a)
    if n_missing:
        print(f"warning: {n_missing} demos had no z labels")
    if not Z_va or not Z_tr:
        raise RuntimeError("empty train or val probe split — check z labels / meta")

    Z_tr, A_tr = np.concatenate(Z_tr), np.concatenate(A_tr)
    Z_va, A_va = np.concatenate(Z_va), np.concatenate(A_va)
    print(f"probe data: train {len(Z_tr)}, val {len(Z_va)}, "
          f"z_dim {Z_tr.shape[1]}, stride {stride}")

    # ---- z -> a_norm (headline R^2) ----
    W_za, b_za = _ridge(Z_tr, A_tr, args.ridge_lambda)
    r2 = _r2(A_va, Z_va @ W_za + b_za)

    # ---- a_norm -> z (adapter init) ----
    W_az, b_az = _ridge(A_tr, Z_tr, args.ridge_lambda)
    z_r2 = _r2(Z_va, A_va @ W_az + b_az)

    dims = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    print("\n=== affine probe z -> a_norm (val R^2) ===")
    for name, v in zip(dims, r2):
        print(f"  {name:8s}: {v:.4f}")
    print(f"  mean    : {r2.mean():.4f}")
    print(f"(reverse a->z mean R^2: {z_r2.mean():.4f})")
    gate = "PASS" if r2.mean() >= 0.6 else "FAIL"
    print(f"\n>>> go/no-go gate (mean R^2 >= 0.6): {gate}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        # adapter: z = a_norm @ W_az + b_az  (stored transposed for nn use)
        A=W_az.T.astype(np.float32),      # [z_dim, 7]
        b=b_az.astype(np.float32),        # [z_dim]
        W_za=W_za.astype(np.float32),
        b_za=b_za.astype(np.float32),
        r2_za=r2.astype(np.float32),
        r2_az=z_r2.astype(np.float32),
        action_mean=a_mean, action_std=a_std,
    )
    print(f"saved adapter init -> {out}")

    report = {
        "r2_per_dim": {k: float(v) for k, v in zip(dims, r2)},
        "r2_mean": float(r2.mean()),
        "r2_reverse_mean": float(z_r2.mean()),
        "n_train": int(len(Z_tr)), "n_val": int(len(Z_va)),
        "z_labels": str(args.z_labels), "gate": gate,
    }
    report_path = Path(args.report_out) if args.report_out else out.with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved report -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("affine probe", parents=[get_args_parser()])
    main(parser.parse_args())
