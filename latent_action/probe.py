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
from scipy.spatial.transform import Rotation as Rot

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
    p.add_argument("--nonlinear", action="store_true", default=False,
                   help="also fit a small MLP z->a_norm probe (diagnostic only, "
                        "no adapter is derived from it). Disambiguates two "
                        "failure modes when the affine R^2 is low: high "
                        "nonlinear R^2 means the action info IS in z but not "
                        "affinely decodable (adapter/theory needs to change); "
                        "low nonlinear R^2 too means z is not really encoding "
                        "the action at all (LAM itself needs to change). "
                        "Runs in a couple minutes on existing z labels, no GPU "
                        "training required.")
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--mlp_epochs", type=int, default=150)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
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


def _windowed_action_target(acts: np.ndarray, stride: int) -> np.ndarray:
    """Net action over a sliding window of `stride` raw per-step actions,
    on the SAME per-step scale as the 1-step action (so it's comparable to
    the action norm-stats computed on single steps).

    Translation (dims 0:3) and gripper (dim 6) simply average — vector sums
    and near-binary gripper commands compose linearly/trivially.

    Rotation (dims 3:6) does NOT: naive averaging/summing of per-step Euler
    deltas ignores that finite rotations don't commute/add, which biases
    the regression target and can depress R^2 for roll/pitch specifically
    (small rotations about axes with little linear-algebra "room" to
    average correctly). We instead compose the per-step delta rotations as
    proper SO(3) elements (R_total = R_{k-1} @ ... @ R_0) and convert the
    net rotation back to a single equivalent Euler delta.
    """
    T = len(acts)
    if stride <= 1:
        return acts
    n = T - stride
    out = np.empty((n, acts.shape[1]), dtype=np.float32)
    out[:, 0:3] = np.stack(
        [acts[i:i + stride, 0:3].mean(axis=0) for i in range(n)]
    )
    out[:, 6] = np.stack([acts[i:i + stride, 6].mean() for i in range(n)])
    for i in range(n):
        rots = Rot.from_euler("xyz", acts[i:i + stride, 3:6])
        net = rots[stride - 1]
        for j in range(stride - 2, -1, -1):
            net = net * rots[j]
        out[i, 3:6] = net.as_euler("xyz") / stride
    return out


def _r2(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
    sse = ((Y - Y_hat) ** 2).sum(axis=0)
    sst = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    return 1.0 - sse / sst


def _mlp_probe(
    Z_tr: np.ndarray, A_tr: np.ndarray, Z_va: np.ndarray, A_va: np.ndarray,
    hidden: int, epochs: int, lr: float, seed: int = 0,
) -> np.ndarray:
    """Diagnostic-only nonlinear probe z -> a_norm (2-layer MLP, ridge-style
    weight decay). Not used to build the adapter — purely to tell apart
    'no affine map exists but the info is there' from 'the info isn't there'.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ztr = torch.from_numpy(Z_tr).to(device)
    Atr = torch.from_numpy(A_tr).to(device)
    Zva = torch.from_numpy(Z_va).to(device)

    model = nn.Sequential(
        nn.Linear(Z_tr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, A_tr.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n = Ztr.shape[0]
    batch_size = min(4096, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(Ztr[idx])
            loss = nn.functional.mse_loss(pred, Atr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_va = model(Zva).cpu().numpy()
    return _r2(A_va, pred_va)


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
            # z_t summarizes motion over [t, t+stride); regress against the
            # properly SO(3)-composed net action over the same window (see
            # _windowed_action_target — naive Euler-angle averaging is wrong
            # for rotations and depresses R^2 on roll/pitch specifically).
            acts = rec["actions"].astype(np.float32)
            a = _windowed_action_target(acts, stride)[: len(z)]
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

    nl_r2 = None
    if args.nonlinear:
        print("\n=== diagnostic: nonlinear (MLP) probe z -> a_norm (val R^2) ===")
        nl_r2 = _mlp_probe(Z_tr, A_tr, Z_va, A_va,
                           hidden=args.mlp_hidden, epochs=args.mlp_epochs,
                           lr=args.mlp_lr, seed=args.seed)
        for name, v in zip(dims, nl_r2):
            print(f"  {name:8s}: {v:.4f}")
        print(f"  mean    : {nl_r2.mean():.4f}")
        if nl_r2.mean() >= 0.5:
            print("\n>>> diagnosis: action info IS in z, but NOT affinely decodable.")
            print("    -> affine identifiability hypothesis falsified; z needs a")
            print("       nonlinear adapter, or the LAM needs an explicit affine-")
            print("       inducing regularizer.")
        elif nl_r2.mean() >= 0.2:
            print("\n>>> diagnosis: weak/partial action signal in z, still far from")
            print("    usable. Some info, but z is dominated by something else")
            print("    (object motion / contact dynamics / scene context).")
        else:
            print("\n>>> diagnosis: z carries ~no recoverable action information,")
            print("    linearly or nonlinearly. recon improving means z explains")
            print("    SOME feature-space delta, but that delta is likely driven")
            print("    by scene/object dynamics, not the robot's own action.")
            print("    -> LAM objective itself needs to change (e.g. CLAM-style")
            print("       few-shot grounding with a small amount of real action")
            print("       labels), not just probe/adapter tuning.")

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
    if nl_r2 is not None:
        report["r2_nonlinear_per_dim"] = {k: float(v) for k, v in zip(dims, nl_r2)}
        report["r2_nonlinear_mean"] = float(nl_r2.mean())
    report_path = Path(args.report_out) if args.report_out else out.with_suffix(".json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved report -> {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("affine probe", parents=[get_args_parser()])
    main(parser.parse_args())
