"""
Stage 1b: label every LIBERO demo with per-step latent actions z.

Writes an HDF5 file mapping demo keys -> z arrays [T - stride, z_dim],
plus a pretraining meta JSON (dataset_name = "libero_z") that the
existing SimVLA dataloader consumes through the LiberoZHandler.

--ego_aug applies the shared ego-motion warp to every pair before
encoding: the resulting z_labels_stress.h5 is used by the probe stress
test (theory predicts probe R^2 stays high iff the invariance regularizer
was on during LAM training).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from latent_action import config as C
from latent_action.data import _demo_frames, iter_demos, load_meta
from latent_action.models import FrozenVisionBackbone, load_lam, shared_ego_augment


def get_args_parser():
    p = argparse.ArgumentParser("z labeling", add_help=False)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--lam_ckpt", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="output z-labels .h5")
    p.add_argument("--z_meta_out", type=str, default=None,
                   help="write pretraining meta JSON (dataset_name=libero_z)")
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ego_aug", action="store_true", default=False,
                   help="apply shared ego-motion warp (stress-test labels)")
    return p


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = FrozenVisionBackbone(args.smolvlm_model_path).to(device)
    lam = load_lam(args.lam_ckpt).to(device).eval()

    meta = load_meta(args.meta_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_demos, n_frames = 0, 0
    with h5py.File(out_path, "w") as zf:
        zf.attrs["z_dim"] = lam.z_dim
        zf.attrs["stride"] = args.stride
        zf.attrs["ego_aug"] = args.ego_aug

        for rec in iter_demos(meta):
            T = rec["num_frames"]
            if T <= args.stride:
                continue
            zs = []
            ts = np.arange(0, T - args.stride)
            for lo in range(0, len(ts), args.batch_size):
                sub = ts[lo:lo + args.batch_size]
                frames_t = _demo_frames(rec["demo"], sub, args.image_size).to(device)
                frames_tk = _demo_frames(rec["demo"], sub + args.stride, args.image_size).to(device)
                if args.ego_aug:
                    frames_t, frames_tk = shared_ego_augment(frames_t, frames_tk)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                    f_t = backbone(frames_t)
                    f_tk = backbone(frames_tk)
                    z = lam.encode(f_t.float(), f_tk.float())
                    if lam.vq is not None:
                        z, _, _ = lam.vq(z)
                zs.append(z.float().cpu().numpy())
            z_arr = np.concatenate(zs, axis=0)
            zf.create_dataset(rec["key"], data=z_arr.astype(np.float32))
            n_demos += 1
            n_frames += len(z_arr)
            if n_demos % 100 == 0:
                print(f"labeled {n_demos} demos / {n_frames} frames")

    print(f"done: {n_demos} demos, {n_frames} frames -> {out_path}")

    if args.z_meta_out:
        z_meta = dict(meta)
        z_meta["dataset_name"] = "libero_z"
        z_meta["z_labels_path"] = str(out_path.resolve())
        z_meta["z_dim"] = int(lam.z_dim)
        z_meta["z_stride"] = args.stride
        zp = Path(args.z_meta_out)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with open(zp, "w") as f:
            json.dump(z_meta, f, indent=2)
        print(f"wrote pretraining meta -> {zp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("z labeling", parents=[get_args_parser()])
    main(parser.parse_args())
