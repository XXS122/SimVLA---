#!/usr/bin/env python3
"""
Offline H1 diagnostic (NO robosuite / NO EGL / NO rollout).

Question it answers: does the trained ChunkBoundaryHead produce *informative*
boundary scores at inference -- i.e. do they rise at genuinely decisive moments
(high ground-truth action change-rate / gripper toggles)? Boundary-driven
adaptive re-planning can only help if this holds. If the boundary is ~constant or
uncorrelated, the head must be retrained on an ABSOLUTE change-rate target.

It reads raw LIBERO demo HDF5 directly (no simulator), runs
SmolVLMVLA.generate_actions(..., return_boundary=True) on sampled timesteps, and
compares the predicted boundary profile against the GT future-action change rate
and gripper toggles. Runs in a couple of minutes on one GPU.

Run in the `simvla` env:
    python analysis/offline_boundary_check.py \
        --checkpoint /path/to/ckpt-200000 \
        --norm_stats ./norm_stats/libero_norm.json \
        --smolvlm_model "$SIMVLA_SMOLVLM_MODEL" \
        --hdf5 /path/to/libero_object/<some_task>_demo.hdf5 \
        --num_demos 3 --stride 5
(--hdf5 may also be a directory; the first *.hdf5 in it is used.)
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from datasets.domain_handler.libero_hdf5 import euler_to_axisangle

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_task_from_filename(path: str) -> str:
    base = os.path.basename(path)
    task = re.sub(r"_demo\.hdf5$", "", base)
    m = re.search(r"SCENE\d+_", task)
    if m:
        task = task[m.end():]
    return task.replace("_", " ")


def main():
    ap = argparse.ArgumentParser(description="Offline H1 diagnostic for the boundary head")
    ap.add_argument("--checkpoint", required=True, help="SimVLA checkpoint dir (with the boundary head)")
    ap.add_argument("--norm_stats", required=True)
    ap.add_argument("--smolvlm_model",
                    default=os.environ.get("SIMVLA_SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct"))
    ap.add_argument("--hdf5", required=True, help="a LIBERO demo .hdf5 file (or a dir; first .hdf5 is used)")
    ap.add_argument("--num_demos", type=int, default=3)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10, help="flow-matching integration steps")
    ap.add_argument("--window", type=int, default=2, help="+-steps counted as 'near' a gripper toggle")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve the HDF5 path (file or directory) and the language instruction.
    h5path = args.hdf5
    if os.path.isdir(h5path):
        cands = sorted(Path(h5path).glob("*.hdf5"))
        if not cands:
            print(f"No .hdf5 found under {h5path}")
            return
        h5path = str(cands[0])
    instruction = parse_task_from_filename(h5path)
    print(f"HDF5       : {h5path}")
    print(f"instruction: {instruction!r}")

    # Load model / processor / norm-stats (same as the policy server).
    print("loading model ...")
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    model.action_space.load_norm_stats(args.norm_stats)
    if getattr(model.transformer, "chunk_boundary_head", None) is None:
        print("WARNING: this checkpoint has NO boundary head (use_adaptive_chunking was false); "
              "boundary will be all-zeros and H1 cannot be judged.")
    T = model.num_actions
    tf = transforms.Compose([
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    input_ids = processor.encode_language([instruction])["input_ids"].to(device)

    def preprocess(img0, img1):
        # Rotate 180 deg (matches the data handler + eval client), resize+normalize,
        # pad to the 3-view layout the model expects.
        i0 = tf(Image.fromarray(np.ascontiguousarray(img0[::-1, ::-1])))
        i1 = tf(Image.fromarray(np.ascontiguousarray(img1[::-1, ::-1])))
        image_input = torch.stack([i0, i1, torch.zeros_like(i0)], 0).unsqueeze(0).to(device)
        image_mask = torch.tensor([[True, True, False]], device=device)
        return image_input, image_mask

    boundaries, gt_rates, near_flags, per_chunk_corr = [], [], [], []

    with h5py.File(h5path, "r") as f:
        if "data" not in f:
            print("no 'data' group in HDF5")
            return
        for dk in list(f["data"].keys())[:args.num_demos]:
            d = f["data"][dk]
            actions = np.asarray(d["actions"], dtype=np.float32)        # [N,7]
            agv = np.asarray(d["obs/agentview_rgb"])                    # [N,H,W,3] uint8
            wrist = np.asarray(d["obs/eye_in_hand_rgb"])
            ee_pos = np.asarray(d["obs/ee_pos"], dtype=np.float32)
            ee_eul = np.asarray(d["obs/ee_ori"], dtype=np.float32)
            grip = np.asarray(d["obs/gripper_states"], dtype=np.float32)
            N = min(len(actions), len(agv), len(wrist), len(ee_pos), len(ee_eul), len(grip))
            if N < T + 2:
                continue
            axis = euler_to_axisangle(ee_eul[:N])
            proprio_all = np.concatenate([ee_pos[:N], axis, grip[:N]], -1).astype(np.float32)  # [N,8]
            # GT change rate is measured in the normalized action space (== training signal).
            a_norm = model.action_space.normalize_action(torch.tensor(actions[:N])).cpu().numpy()

            for t in range(0, N - (T + 1), args.stride):
                image_input, image_mask = preprocess(agv[t], wrist[t])
                proprio = torch.tensor(proprio_all[t:t + 1], device=device)
                with torch.no_grad():
                    _, b = model.generate_actions(
                        input_ids=input_ids, image_input=image_input,
                        image_mask=image_mask, proprio=proprio,
                        steps=args.steps, return_boundary=True,
                    )
                b = b[0].float().cpu().numpy()                          # [T] predicted boundary
                gt = np.linalg.norm(a_norm[t + 1:t + 1 + T] - a_norm[t:t + T], axis=-1)  # [T] GT change rate
                gcmd = actions[t:t + T + 1, 6]                          # GT gripper command over the chunk
                toggles = [k for k in range(1, len(gcmd)) if (gcmd[k] > 0) != (gcmd[k - 1] > 0)]
                for k in range(T):
                    boundaries.append(float(b[k]))
                    gt_rates.append(float(gt[k]))
                    near_flags.append(any(abs(k - tg) <= args.window for tg in toggles))
                if b.std() > 1e-6 and gt.std() > 1e-6:
                    per_chunk_corr.append(float(np.corrcoef(b, gt)[0, 1]))

    boundaries = np.asarray(boundaries)
    gt_rates = np.asarray(gt_rates)
    near = np.asarray(near_flags, dtype=bool)
    if boundaries.size == 0:
        print("no samples collected (demos too short for the chunk length?)")
        return

    print("\n==================== H1 diagnostic ====================")
    print(f"samples={boundaries.size}  chunks={len(per_chunk_corr)}")
    print(f"boundary: mean={boundaries.mean():.3f} std={boundaries.std():.3f} "
          f"min={boundaries.min():.3f} max={boundaries.max():.3f}")

    gcorr = (float(np.corrcoef(boundaries, gt_rates)[0, 1])
             if boundaries.std() > 1e-6 and gt_rates.std() > 1e-6 else float("nan"))
    pcc = float(np.nanmean(per_chunk_corr)) if per_chunk_corr else float("nan")
    print(f"corr(boundary, GT change-rate): global={gcorr:.3f}  per-chunk-mean={pcc:.3f}")

    bn = bf = None
    if near.any() and (~near).any():
        bn, bf = float(boundaries[near].mean()), float(boundaries[~near].mean())
        print(f"boundary NEAR gripper toggle={bn:.3f}  FAR={bf:.3f}  ratio={bn / max(bf, 1e-6):.2f}")
    else:
        print("(not enough gripper toggles to compute near/far)")

    print("\n[VERDICT]", end=" ")
    if boundaries.std() < 0.05:
        print("boundary is ~CONSTANT -> useless for adaptive timing. H1 FAILS; "
              "retrain the head on an ABSOLUTE change-rate target.")
    elif (not np.isnan(gcorr) and gcorr > 0.3) or (bn is not None and bf is not None and bn > 1.3 * bf):
        print("boundary TRACKS decisiveness -> H1 SUPPORTED. Adaptive re-planning can work: "
              "raise --max_horizon and sweep beta on the success-vs-compute Pareto.")
    else:
        print("WEAK/no correlation -> H1 WEAK. Most likely the per-chunk-normalized training "
              "target; retrain the boundary head on an absolute change-rate before relying on it.")


if __name__ == "__main__":
    main()
