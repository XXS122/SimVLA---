"""
Phase B: distill PrefixStateUpdater against the real (frozen) SmolVLM
fused text-model forward.

Per pair (frame_t, frame_{t+stride}) from the same episode (fixed
instruction):
  cached_state   = teacher.expensive_forward(*teacher.cheap_forward(frame_t))    [real, t]
  target         = teacher.expensive_forward(*teacher.cheap_forward(frame_tk))   [real, t+stride -- distillation target]
  combined_tk, _ = teacher.cheap_forward(frame_tk)                               [cheap features at t+stride]
  pred           = updater(cached_state, combined_tk)
  loss           = normalized MSE(pred, target) -- see distillation_loss

This teacher-forces the "cached state" input during training (always the
REAL previous output, not the updater's own previous prediction). At
inference the updater is applied recursively on its own outputs between
periodic resets -- Phase C must check whether recursive drift compounds
faster than this training loss implies (this is exactly what the
geometric drift bound in the paper's theory is meant to address).

Usage:
    python -m streaming_prefix.train_student \
        --meta_path <libero_train.json> --output_dir <ws>/streaming_prefix \
        --iters 30000 --batch_size 64 --stride 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from latent_action import config as C
from latent_action.data import gpu_preprocess
from streaming_prefix.data import InstructionSequenceDataset
from streaming_prefix.models import (
    PrefixStateUpdater, VLMPrefixTeacher, masked_norm_mse, save_updater,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args_parser():
    p = argparse.ArgumentParser("PrefixStateUpdater distillation", add_help=False)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    # data
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--stride", type=int, default=1,
                   help="frame gap between consecutive control steps")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    # training strategy
    p.add_argument("--train_mode", type=str, default="anchored",
                   choices=["anchored", "rollout"],
                   help="anchored: predict s_k from the REAL reset s_0 at gaps "
                        "1..rollout_len (no exposure bias, matches anchored "
                        "deployment; recommended). rollout: recursive + scheduled "
                        "sampling (legacy).")
    # recursive-rollout / scheduled-sampling (used only when --train_mode rollout)
    p.add_argument("--rollout_len", type=int, default=8,
                   help="window length: gaps 1..rollout_len (anchored) or "
                        "unroll steps (rollout); 1 == single-step training")
    p.add_argument("--max_ss_prob", type=float, default=0.9,
                   help="max probability of feeding the model's own prediction "
                        "(scheduled sampling); ramped 0 -> this over --ss_ramp_frac")
    p.add_argument("--ss_ramp_frac", type=float, default=0.5,
                   help="fraction of training over which ss_prob ramps to max")
    # model
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    # optimization
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, default=None)
    return p


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", C.WANDB_PROJECT),
            name=args.run_name or f"prefix-updater-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    print(f"loading frozen teacher VLM: {args.smolvlm_model_path}")
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_model_path, dtype=torch.float32, trust_remote_code=True,
    ).to(device)
    teacher = VLMPrefixTeacher(vlm).to(device)
    hidden_size = vlm.config.text_config.hidden_size
    print(f"teacher hidden_size={hidden_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.smolvlm_model_path, trust_remote_code=True)

    updater = PrefixStateUpdater(
        hidden_size=hidden_size, depth=args.depth, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    n_updater = sum(p.numel() for p in updater.parameters())
    n_teacher_lm = sum(p.numel() for p in vlm.model.text_model.parameters())
    print(f"[PrefixStateUpdater] params: {n_updater/1e6:.1f}M "
          f"(teacher text_model: {n_teacher_lm/1e6:.1f}M -> "
          f"{100*n_updater/n_teacher_lm:.1f}% of the cost it replaces)")

    dataset = InstructionSequenceDataset(
        args.meta_path, tokenizer=tokenizer, rollout_len=args.rollout_len,
        stride=args.stride, seq_len=args.seq_len, training=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    optim = AdamW(updater.parameters(), lr=args.learning_rate,
                  weight_decay=args.weight_decay, betas=(0.9, 0.95))

    step, t0 = 0, time.time()
    updater.train()
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ss_ramp_steps = max(1, int(args.ss_ramp_frac * args.iters))

    for batch in loader:
        # frames: [B, K+1, V, H, W, 3] uint8
        frames = batch["frames"].to(device, non_blocking=True)
        B, Kp1, V = frames.shape[:3]
        K = Kp1 - 1
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        mask = torch.ones(B, V, dtype=torch.bool, device=device)

        lr = cosine_lr(step, args.warmup_steps, args.iters, args.learning_rate)
        for g in optim.param_groups:
            g["lr"] = lr
        # scheduled sampling probability: 0 (teacher forcing) -> max_ss_prob
        ss_prob = args.max_ss_prob * min(1.0, step / ss_ramp_steps)

        # Real teacher states + cheap features for every step in the window (frozen).
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            real_states, cheap_feats = [], []
            for k in range(Kp1):
                fk = gpu_preprocess(frames[:, k], args.image_size, device)
                ck, ak = teacher.cheap_forward(fk, mask, input_ids)
                real_states.append(teacher.expensive_forward(ck, ak).float())
                cheap_feats.append(ck.float())
            attn_full = torch.ones(B, real_states[0].shape[1], device=device)

        step_losses = []
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            if args.train_mode == "anchored":
                # Anchored-to-reset: predict s_k from the REAL reset s_0 (never
                # a prediction) at every gap k. No recursion => no exposure bias
                # by construction; matches the anchored deployment strategy. Each
                # gap is normalized by its own change-from-reset energy so 1.0 =
                # "predict no change from the anchor".
                anchor = real_states[0]
                for k in range(1, Kp1):
                    pred = updater(anchor, cheap_feats[k])
                    denom = (real_states[k] - anchor).pow(2).mean().detach()
                    step_losses.append(masked_norm_mse(pred, real_states[k], denom, attn_full))
            else:
                # Recursive rollout with scheduled sampling (legacy; found to
                # only weakly reduce inference drift -- see PROGRESS.md).
                fed_state = real_states[0]
                for k in range(1, Kp1):
                    pred = updater(fed_state, cheap_feats[k])
                    denom = (real_states[k] - real_states[k - 1]).pow(2).mean().detach()
                    step_losses.append(masked_norm_mse(pred, real_states[k], denom, attn_full))
                    use_pred = torch.rand(()) < ss_prob
                    fed_state = pred.detach() if use_pred else real_states[k]
            loss = torch.stack(step_losses).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(updater.parameters(), 1.0)
        optim.step()

        if step % args.log_interval == 0:
            dt = (time.time() - t0) / max(1, args.log_interval)
            t0 = time.time()
            logs = {
                "recon": loss.item(),
                "recon_step1": step_losses[0].item(),
                "recon_stepK": step_losses[-1].item(),
                "ss_prob": ss_prob, "lr": lr,
            }
            print(f"[{step}/{args.iters}] " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()) + f" ({dt:.2f}s/it)")
            if use_wandb:
                wandb.log(logs, step=step)

        step += 1
        if step % args.save_interval == 0 or step >= args.iters:
            ckpt = out / (f"updater_{step}.pt" if step < args.iters else "updater_final.pt")
            save_updater(updater, ckpt, extra={"step": step, "args": vars(args)})
            with open(out / "updater_config.json", "w") as f:
                json.dump({**updater.config_dict(), "stride": args.stride, "image_size": args.image_size}, f, indent=2)
            print(f"saved {ckpt}")
        if step >= args.iters:
            break

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PrefixStateUpdater distillation", parents=[get_args_parser()])
    main(parser.parse_args())
