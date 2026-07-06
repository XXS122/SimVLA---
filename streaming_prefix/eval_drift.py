"""
Phase C, diagnostic 1: recursive drift of the trained PrefixStateUpdater.

Phase B trained the updater teacher-forced (always conditioned on the
REAL previous fused output). At deployment it must run recursively on its
OWN previous prediction between periodic resets. This script measures
whether that recursion stays bounded (the paper's geometric-drift claim,
error ceiling ~ delta_reconstruction / (1 - L)) or diverges.

For each held-out episode, starting from a real full re-encode at step 0
(a "reset"), it applies the updater recursively for up to --horizon
control steps WITHOUT resetting, and at each step k records:
  - drift_k  : normalized MSE between the recursive prediction and the
               REAL full re-encode at step k (the ground truth the model
               would have produced with no shortcut). 0 = perfect,
               1.0-scale is the same normalization as training recon.
  - tf_k     : the same error but teacher-forced (fed the real previous
               state) — the "no exposure bias" reference. The gap
               (drift_k - tf_k) isolates the compounding due to
               self-conditioning.

Reports drift as a function of k (and its empirical ceiling), plus the
implied safe reset period for a target drift budget — the quantity the
paper turns into a design rule.

Usage:
    python -m streaming_prefix.eval_drift \
        --meta_path <libero_train.json> --updater_ckpt <...>/updater_final.pt \
        --smolvlm_model_path $SIMVLA_SMOLVLM_MODEL \
        --horizon 20 --num_episodes 50 --output drift_report.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from latent_action import config as C
from latent_action.data import _demo_frames_raw, gpu_preprocess, load_meta
from streaming_prefix.models import VLMPrefixTeacher, load_updater


def get_args_parser():
    p = argparse.ArgumentParser("recursive drift eval", add_help=False)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--updater_ckpt", type=str, required=True)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=20,
                   help="max control steps to recurse without a reset")
    p.add_argument("--num_episodes", type=int, default=50)
    p.add_argument("--drift_budget", type=float, default=0.3,
                   help="normalized-drift threshold for the implied reset period")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    return p


def _norm_mse(pred: torch.Tensor, target: torch.Tensor, ref_state: torch.Tensor) -> float:
    """Normalized MSE on the same scale as training recon: error energy /
    frame-change energy (relative to the step-0 reset reference)."""
    denom = (target - ref_state).pow(2).mean() + 1e-8
    return (pred - target).pow(2).mean().item() / denom.item()


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)

    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_model_path, dtype=torch.float32, trust_remote_code=True).to(device)
    teacher = VLMPrefixTeacher(vlm).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.smolvlm_model_path, trust_remote_code=True)
    updater = load_updater(args.updater_ckpt).to(device).eval()

    meta = load_meta(args.meta_path)
    items = list(meta["datalist"])
    rng.shuffle(items)

    # drift_by_k[k] and tf_by_k[k] accumulate across episodes
    H = args.horizon
    drift_by_k = [[] for _ in range(H + 1)]
    tf_by_k = [[] for _ in range(H + 1)]

    n_done = 0
    for item in items:
        if n_done >= args.num_episodes:
            break
        try:
            f = h5py.File(item["path"], "r")
        except OSError:
            continue
        with f:
            if "data" not in f:
                continue
            input_ids = tokenizer(item.get("task", ""), return_tensors="pt",
                                  padding="max_length", max_length=args.seq_len,
                                  truncation=True)["input_ids"].to(device)
            mask = torch.ones(1, 2, dtype=torch.bool, device=device)
            for dk in list(f["data"].keys()):
                if n_done >= args.num_episodes:
                    break
                demo = f["data"][dk]
                T = min(len(demo["obs/agentview_rgb"]), len(demo["obs/eye_in_hand_rgb"]))
                if T <= H:
                    continue

                # precompute real full re-encode and cheap features for steps 0..H
                real_states, cheap_feats = [], []
                for k in range(H + 1):
                    frames = gpu_preprocess(_demo_frames_raw(demo, np.array([k])), args.image_size, device)
                    combined, attn = teacher.cheap_forward(frames, mask, input_ids)
                    real_states.append(teacher.expensive_forward(combined, attn).float())
                    cheap_feats.append(combined.float())

                ref = real_states[0]  # step-0 reset reference for normalization
                recursive_state = real_states[0].clone()  # start from a real reset
                drift_by_k[0].append(0.0)
                tf_by_k[0].append(0.0)
                for k in range(1, H + 1):
                    # recursive: feed the model's own previous prediction
                    recursive_state = updater(recursive_state, cheap_feats[k])
                    drift_by_k[k].append(_norm_mse(recursive_state, real_states[k], ref))
                    # teacher-forced reference: feed the real previous state
                    tf_pred = updater(real_states[k - 1], cheap_feats[k])
                    tf_by_k[k].append(_norm_mse(tf_pred, real_states[k], ref))
                n_done += 1

    def summarize(rows):
        return [float(np.mean(r)) if r else float("nan") for r in rows]

    drift_mean = summarize(drift_by_k)
    tf_mean = summarize(tf_by_k)

    # implied reset period: last k whose recursive drift stays under budget
    reset_period = H
    for k in range(1, H + 1):
        if drift_mean[k] > args.drift_budget:
            reset_period = k - 1
            break

    print(f"\nepisodes evaluated: {n_done}")
    print(f"{'k':>4} {'recursive_drift':>16} {'teacher_forced':>16} {'exposure_gap':>14}")
    for k in range(H + 1):
        gap = drift_mean[k] - tf_mean[k]
        print(f"{k:>4} {drift_mean[k]:>16.4f} {tf_mean[k]:>16.4f} {gap:>14.4f}")

    # crude geometric-bound check: is drift sub-linear / saturating?
    tail = [d for d in drift_mean[1:] if not np.isnan(d)]
    saturating = len(tail) >= 3 and (tail[-1] - tail[-2]) < (tail[1] - tail[0])
    print(f"\nimplied safe reset period (drift<{args.drift_budget}): {reset_period} steps")
    print(f"drift appears to be {'SATURATING (consistent with a geometric bound)' if saturating else 'NOT clearly saturating — check for divergence'}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({
                "config": vars(args), "episodes": n_done,
                "recursive_drift_by_k": drift_mean,
                "teacher_forced_by_k": tf_mean,
                "reset_period": reset_period, "saturating": bool(saturating),
            }, fh, indent=2)
        print(f"saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("recursive drift eval", parents=[get_args_parser()])
    main(parser.parse_args())
