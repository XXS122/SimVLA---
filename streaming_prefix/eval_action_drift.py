"""
Phase C, diagnostic 3: ACTION-space drift (the metric that actually
matters), not feature-space drift.

eval_drift measures L2 error in the fused VLM features. But those features
are only a means to an end — what matters is whether the (frozen, trained)
action head produces the SAME actions from the approximate features as
from the real ones. A large feature error may or may not change actions,
depending on how the action head uses the features. This script measures
the action difference directly, so it can rule the streaming approach in
or out without setting up the LIBERO simulator.

For each control step k it compares actions generated from:
  - real   : real full re-encode (ground truth)
  - anchored: updater(real reset s_0, cheap_k)   (recommended deployment)
  - recursive: updater(own previous prediction, cheap_k)

The SAME flow-matching noise is used for real vs approximate so the
difference isolates the feature approximation (not sampling noise). Actions
are un-normalized to physical units; drift is reported both as a fraction
of the real action magnitude and split into translation / rotation /
gripper.

Requires a TRAINED SmolVLMVLA action head (e.g. the released SimVLA
checkpoint or a baseline you trained) — an untrained head produces
meaningless actions.

Usage:
    python -m streaming_prefix.eval_action_drift \
        --checkpoint <trained SmolVLMVLA dir> \
        --updater_ckpt <...>/updater_final.pt \
        --meta_path runs/latent_action_ws/metas/libero_train.json \
        --norm_stats_path runs/latent_action_ws/norm_stats/libero_norm.json \
        --horizon 12 --num_episodes 30 --euler_steps 10 \
        --output logs/action_drift.json
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
from models.modeling_smolvlm_vla import SmolVLMVLA
from streaming_prefix.models import VLMPrefixTeacher, load_updater


def get_args_parser():
    p = argparse.ArgumentParser("action-space drift eval", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="trained SmolVLMVLA checkpoint dir (action head + VLM)")
    p.add_argument("--updater_ckpt", type=str, required=True)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--norm_stats_path", type=str, default=None)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--num_episodes", type=int, default=30)
    p.add_argument("--euler_steps", type=int, default=10)
    p.add_argument("--action_budget", type=float, default=0.15,
                   help="fractional action-drift threshold for the reset period")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default=None)
    return p


def _proprio_from_demo(demo, k: int) -> np.ndarray:
    """Reproduce libero_hdf5's 8-dim proprio [ee_pos(3), axis_angle(3),
    gripper(2)] at frame k, matching the training/inference convention."""
    from datasets.domain_handler.libero_hdf5 import euler_to_axisangle
    ee_pos = np.asarray(demo["obs/ee_pos"][k], dtype=np.float32)
    ee_ori = euler_to_axisangle(np.asarray(demo["obs/ee_ori"][k], dtype=np.float32)[None])[0]
    grip = np.asarray(demo["obs/gripper_states"][k], dtype=np.float32)
    return np.concatenate([ee_pos, ee_ori, grip]).astype(np.float32)


@torch.no_grad()
def _actions_from_features(model, vlm_features, proprio, noise, steps):
    """Euler-integrate the flow head from a FIXED noise tensor, so two calls
    differ only in vlm_features. Returns un-normalized action chunk."""
    x = noise.clone()
    dt = -1.0 / steps
    t = 1.0
    while t > -dt / 2:
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = model.transformer(vlm_features=vlm_features, action_with_noise=x, proprio=proprio, t=tt)
        x = x + dt * v
        t = t + dt
    return model.action_space.postprocess(x)


def _rel(err_sq: float, ref_sq: float) -> float:
    return float(np.sqrt(err_sq / (ref_sq + 1e-8)))


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    if args.norm_stats_path:
        model.action_space.load_norm_stats(args.norm_stats_path)
    model.action_space.to(device)

    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_model_path, dtype=torch.float32, trust_remote_code=True).to(device)
    teacher = VLMPrefixTeacher(vlm).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.smolvlm_model_path, trust_remote_code=True)
    updater = load_updater(args.updater_ckpt).to(device).eval()

    H = args.horizon
    Dact = model.action_space.dim_action
    Tn = model.num_actions
    # slices for interpretable per-group reporting (libero_joint 7-dim action)
    groups = {"translation": slice(0, 3), "rotation": slice(3, 6), "gripper": slice(6, 7)}

    anch_by_k = [[] for _ in range(H + 1)]
    rec_by_k = [[] for _ in range(H + 1)]
    grp_anch = {g: [[] for _ in range(H + 1)] for g in groups}

    meta = load_meta(args.meta_path)
    items = list(meta["datalist"])
    rng.shuffle(items)

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

                real_states, cheap_feats = [], []
                for k in range(H + 1):
                    fk = gpu_preprocess(_demo_frames_raw(demo, np.array([k])), args.image_size, device)
                    ck, ak = teacher.cheap_forward(fk, mask, input_ids)
                    real_states.append(teacher.expensive_forward(ck, ak).float())
                    cheap_feats.append(ck.float())

                recursive_state = real_states[0].clone()
                for k in range(1, H + 1):
                    proprio = torch.tensor(_proprio_from_demo(demo, k), device=device).unsqueeze(0)
                    proprio = proprio.to(next(model.parameters()).dtype)
                    noise = torch.randn(1, Tn, Dact, device=device, generator=gen,
                                        dtype=next(model.parameters()).dtype)

                    feat_real = real_states[k].to(next(model.parameters()).dtype)
                    a_real = _actions_from_features(model, feat_real, proprio, noise, args.euler_steps)

                    anch_feat = updater(real_states[0], cheap_feats[k]).to(feat_real.dtype)
                    a_anch = _actions_from_features(model, anch_feat, proprio, noise, args.euler_steps)

                    recursive_state = updater(recursive_state, cheap_feats[k])
                    a_rec = _actions_from_features(model, recursive_state.to(feat_real.dtype),
                                                   proprio, noise, args.euler_steps)

                    ref_sq = a_real.pow(2).mean().item()
                    anch_by_k[k].append(_rel((a_anch - a_real).pow(2).mean().item(), ref_sq))
                    rec_by_k[k].append(_rel((a_rec - a_real).pow(2).mean().item(), ref_sq))
                    for g, sl in groups.items():
                        rsq = a_real[..., sl].pow(2).mean().item()
                        grp_anch[g][k].append(_rel((a_anch[..., sl] - a_real[..., sl]).pow(2).mean().item(), rsq))
                n_done += 1

    def summ(rows):
        return [float(np.mean(r)) if r else float("nan") for r in rows]

    anch_mean, rec_mean = summ(anch_by_k), summ(rec_by_k)
    grp_mean = {g: summ(grp_anch[g]) for g in groups}

    def reset_period(series):
        for k in range(1, H + 1):
            if series[k] > args.action_budget:
                return k - 1
        return H

    print(f"\nepisodes: {n_done}   action drift = RMS(a_approx - a_real) / RMS(a_real)")
    print(f"{'k':>4} {'anchored':>10} {'recursive':>10} {'anch_trans':>11} {'anch_rot':>9} {'anch_grip':>10}")
    for k in range(1, H + 1):
        print(f"{k:>4} {anch_mean[k]:>10.4f} {rec_mean[k]:>10.4f} "
              f"{grp_mean['translation'][k]:>11.4f} {grp_mean['rotation'][k]:>9.4f} "
              f"{grp_mean['gripper'][k]:>10.4f}")

    rp = reset_period(anch_mean)
    print(f"\nanchored reset period (action drift < {args.action_budget}): {rp} steps")
    print(">>> This is the metric that matters: if anchored action drift stays "
          "small (<~0.15) for a useful number of steps, the feature-space drift "
          "does NOT translate into bad actions and the streaming approach is "
          "viable — proceed to closed-loop LIBERO. If it blows up immediately, "
          "feature fidelity matters and the target should move to action-level "
          "distillation (train the updater through the frozen action head).")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"config": vars(args), "episodes": n_done,
                       "anchored_action_drift_by_k": anch_mean,
                       "recursive_action_drift_by_k": rec_mean,
                       "anchored_by_group": grp_mean,
                       "anchored_reset_period": rp}, fh, indent=2)
        print(f"saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("action-space drift eval", parents=[get_args_parser()])
    main(parser.parse_args())
