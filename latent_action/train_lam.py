"""
Stage 1: train the Latent Action Model on action-free LIBERO video.

Loss = forward reconstruction (MSE in SigLIP/connector feature space)
     + var/cov identifiability regularizer (whitening up to rotation)
     + ego-motion invariance (shared-nuisance augmentation)
     [+ VQ commitment when --use_vq, for the discrete ablation]

Ground-truth actions are never used here.

Usage (normally via `python -m latent_action.run train-lam`):
    python -m latent_action.train_lam --meta_path <libero_train.json> \
        --output_dir <ws>/lam --iters 50000 --batch_size 96
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

# Avoid "DataLoader worker killed by Bus error" in containers with a small
# /dev/shm: exchange tensors through the file system instead of shared memory.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from latent_action import config as C
from latent_action.data import FramePairDataset, gpu_preprocess
from latent_action.models import (
    FrozenVisionBackbone,
    LatentActionModel,
    save_lam,
    shared_ego_augment,
    variance_covariance_reg,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args_parser():
    p = argparse.ArgumentParser("LAM training", add_help=False)
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    # data
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=8)
    # model
    p.add_argument("--z_dim", type=int, default=C.Z_DIM)
    p.add_argument("--lam_dim", type=int, default=512)
    p.add_argument("--enc_depth", type=int, default=4)
    p.add_argument("--dec_depth", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--use_vq", action="store_true", default=False,
                   help="discrete VQ bottleneck (LAPA/UniVLA-style ablation)")
    p.add_argument("--vq_codes", type=int, default=256)
    # loss coefficients
    p.add_argument("--var_coef", type=float, default=1.0)
    p.add_argument("--cov_coef", type=float, default=0.1)
    p.add_argument("--inv_coef", type=float, default=1.0)
    p.add_argument("--vq_coef", type=float, default=0.25)
    p.add_argument("--aug_prob", type=float, default=0.5)
    # optimization
    p.add_argument("--iters", type=int, default=50000)
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
            name=args.run_name or f"lam-{time.strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )

    backbone = FrozenVisionBackbone(args.smolvlm_model_path).to(device)
    model = LatentActionModel(
        vlm_dim=backbone.out_dim,
        dim=args.lam_dim,
        z_dim=args.z_dim,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        num_heads=args.num_heads,
        use_vq=args.use_vq,
        vq_codes=args.vq_codes,
    ).to(device)
    print(f"[LAM] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"vlm_dim={backbone.out_dim}, z_dim={args.z_dim}, use_vq={args.use_vq}")

    dataset = FramePairDataset(
        args.meta_path, image_size=args.image_size,
        stride=args.stride, training=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    optim = AdamW(model.parameters(), lr=args.learning_rate,
                  weight_decay=args.weight_decay, betas=(0.9, 0.95))

    step, t0 = 0, time.time()
    model.train()
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    for batch in loader:
        # workers ship raw uint8 frames; resize/normalize on GPU
        frames_t = gpu_preprocess(batch["frames_t"], args.image_size, device)
        frames_tk = gpu_preprocess(batch["frames_tk"], args.image_size, device)

        lr = cosine_lr(step, args.warmup_steps, args.iters, args.learning_rate)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            feats_t = backbone(frames_t)
            feats_tk = backbone(frames_tk)
            outputs = model(feats_t.float(), feats_tk.float())
            z = outputs["z"]

            reg = variance_covariance_reg(z)
            loss = (
                outputs["recon_loss"]
                + args.var_coef * reg["var_loss"]
                + args.cov_coef * reg["cov_loss"]
                + args.vq_coef * outputs["vq_loss"]
            )

            inv_loss = torch.zeros((), device=device)
            if args.inv_coef > 0 and torch.rand(()) < args.aug_prob:
                aug_t, aug_tk = shared_ego_augment(frames_t, frames_tk)
                feats_at = backbone(aug_t)
                feats_atk = backbone(aug_tk)
                z_aug = model.encode(feats_at.float(), feats_atk.float())
                if model.vq is not None:
                    z_aug, _, _ = model.vq(z_aug)
                inv_loss = torch.nn.functional.mse_loss(z_aug, z.detach())
                loss = loss + args.inv_coef * inv_loss

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.log_interval == 0:
            dt = (time.time() - t0) / max(1, args.log_interval)
            t0 = time.time()
            logs = {
                "loss": loss.item(),
                "recon": outputs["recon_loss"].item(),
                "var": reg["var_loss"].item(),
                "cov": reg["cov_loss"].item(),
                "inv": float(inv_loss),
                "vq": float(outputs["vq_loss"]),
                "lr": lr,
            }
            print(f"[{step}/{args.iters}] " +
                  " ".join(f"{k}={v:.4f}" for k, v in logs.items()) +
                  f" ({dt:.2f}s/it)")
            if use_wandb:
                wandb.log(logs, step=step)

        step += 1
        if step % args.save_interval == 0 or step >= args.iters:
            ckpt = out / (f"lam_{step}.pt" if step < args.iters else "lam_final.pt")
            save_lam(model, ckpt, extra={"step": step, "args": vars(args)})
            with open(out / "lam_config.json", "w") as f:
                json.dump({**model.config_dict(), "stride": args.stride,
                           "image_size": args.image_size}, f, indent=2)
            print(f"saved {ckpt}")
        if step >= args.iters:
            break

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LAM training", parents=[get_args_parser()])
    main(parser.parse_args())
