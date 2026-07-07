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


def _norm_mse(pred: torch.Tensor, target: torch.Tensor, scale: float) -> float:
    """MSE(pred, target) divided by a supplied, per-episode STABLE scale
    (mean single-step change energy). Using a fixed scale — instead of the
    step-k-specific ‖target − reset‖² — avoids the artifact where episodes
    start with the robot stationary (‖s_1 − s_0‖² ≈ 0), which made the
    normalized drift blow up near k=1 and was not comparable to the
    training recon. This matches training's single-step normalization."""
    return (pred - target).pow(2).mean().item() / (scale + 1e-8)


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

    # accumulate across episodes, for each deployment strategy
    H = args.horizon
    recursive_by_k = [[] for _ in range(H + 1)]   # feed own prediction (compounds)
    anchored_by_k = [[] for _ in range(H + 1)]    # feed the real last reset s_0 (no compounding)
    tf_by_k = [[] for _ in range(H + 1)]          # feed real previous state (gap-1 reference)

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

                # stable per-episode scale: mean single-step change energy
                scale = float(np.mean([
                    (real_states[j] - real_states[j - 1]).pow(2).mean().item()
                    for j in range(1, H + 1)
                ]))

                recursive_state = real_states[0].clone()  # start from a real reset
                for lst in (recursive_by_k, anchored_by_k, tf_by_k):
                    lst[0].append(0.0)
                for k in range(1, H + 1):
                    # recursive: feed the model's own previous prediction (compounds)
                    recursive_state = updater(recursive_state, cheap_feats[k])
                    recursive_by_k[k].append(_norm_mse(recursive_state, real_states[k], scale))
                    # anchored: feed the real last reset s_0 (gap-k, no compounding)
                    anch_pred = updater(real_states[0], cheap_feats[k])
                    anchored_by_k[k].append(_norm_mse(anch_pred, real_states[k], scale))
                    # teacher-forced: feed the real previous state (gap-1 reference)
                    tf_pred = updater(real_states[k - 1], cheap_feats[k])
                    tf_by_k[k].append(_norm_mse(tf_pred, real_states[k], scale))
                n_done += 1

    def summarize(rows):
        return [float(np.mean(r)) if r else float("nan") for r in rows]

    recursive_mean = summarize(recursive_by_k)
    anchored_mean = summarize(anchored_by_k)
    tf_mean = summarize(tf_by_k)

    def reset_period(series):
        for k in range(1, H + 1):
            if series[k] > args.drift_budget:
                return k - 1
        return H

    rp_recursive = reset_period(recursive_mean)
    rp_anchored = reset_period(anchored_mean)

    print(f"\nepisodes evaluated: {n_done}  (drift normalized by mean single-step change energy)")
    print(f"{'k':>4} {'recursive':>11} {'anchored':>11} {'teacher_forced':>15}")
    for k in range(H + 1):
        print(f"{k:>4} {recursive_mean[k]:>11.4f} {anchored_mean[k]:>11.4f} {tf_mean[k]:>15.4f}")

    print(f"\nimplied safe reset period (drift<{args.drift_budget}):")
    print(f"  recursive deployment : {rp_recursive} steps")
    print(f"  anchored  deployment : {rp_anchored} steps")
    print("\n>>> 'anchored' feeds the real last-reset state every step (no error "
          "compounding by construction); it is the recommended deployment and the "
          "one to compare against full re-encode. 'recursive' feeds the model's own "
          "prediction and is expected to be worse. 'teacher_forced' (gap-1, real "
          "state) is the best-case floor.")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({
                "config": vars(args), "episodes": n_done,
                "recursive_drift_by_k": recursive_mean,
                "anchored_drift_by_k": anchored_mean,
                "teacher_forced_by_k": tf_mean,
                "reset_period_recursive": rp_recursive,
                "reset_period_anchored": rp_anchored,
            }, fh, indent=2)
        print(f"saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("recursive drift eval", parents=[get_args_parser()])
    main(parser.parse_args())
