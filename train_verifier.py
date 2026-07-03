"""
Action Energy Verifier Training (Task-2, A800 / YYK site)

Trains a lightweight energy model E(o, a) with InfoNCE:
  - positive  = the dataset action chunk (normalized)
  - negatives = (a) proposals from the one-step MeanFlow generator
                    (the checkpoint shipped from the A100 site),
                (b) Gaussian-perturbed ground truth,
                (c) in-batch shuffled ground truth.

Generator proposals that land within --min_neg_dist (mean L2 in normalized
action space) of the ground truth are masked out as false negatives.

Only the frozen SFT policy's VLM runs for features; the generator's action
head is reused on those same features, so a single 80GB GPU handles
everything comfortably.

Usage:
    python train_verifier.py \
        --policy_ckpt   ./runs/simvla_libero_small/ckpt-200000 \
        --generator_ckpt ./runs/meanflow_small/ckpt-50000 \
        --train_metas_path $LIBERO_DATASETS/libero_train.json \
        --norm_stats_path ./norm_stats/libero_norm.json \
        --output_dir ./runs/verifier_small
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.action_hub import build_action_space
from models.verifier import ActionEnergyVerifier, ActionEnergyVerifierConfig, info_nce_loss

from train_smolvlm import get_logger

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args_parser():
    parser = argparse.ArgumentParser("Verifier training", add_help=False)

    # I/O
    parser.add_argument("--policy_ckpt", type=str, required=True,
                        help="SFT SimVLA checkpoint (frozen; provides VLM features)")
    parser.add_argument("--generator_ckpt", type=str, default=None,
                        help="One-step MeanFlow checkpoint (frozen; provides negatives). "
                             "Optional: without it only perturb/shuffle negatives are used.")
    parser.add_argument("--output_dir", type=str, default="./runs/verifier")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)

    # Verifier architecture
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_pool_tokens", type=int, default=8)

    # Negatives
    parser.add_argument("--num_neg_gen", type=int, default=8,
                        help="Generator proposals per sample")
    parser.add_argument("--gen_steps", type=int, default=1,
                        help="Integration steps for generator proposals")
    parser.add_argument("--num_neg_perturb", type=int, default=4)
    parser.add_argument("--perturb_std_min", type=float, default=0.1)
    parser.add_argument("--perturb_std_max", type=float, default=0.5)
    parser.add_argument("--num_neg_shuffle", type=int, default=4)
    parser.add_argument("--min_neg_dist", type=float, default=0.1,
                        help="Mask generator negatives closer than this mean-L2 "
                             "(normalized action space) to the ground truth")
    parser.add_argument("--temperature", type=float, default=0.1)

    # Optimizer / schedule
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None)

    return parser


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def cosine_lr(step, warmup, total, base_lr, min_ratio):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT") or args.wandb_project
    use_wandb = WANDB_AVAILABLE and wandb_api_key and wandb_project

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")

    accelerator = Accelerator(log_with=log_with, project_dir=output_dir)
    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": f"verifier-{time.strftime('%Y%m%d-%H%M%S')}"}},
        )
    else:
        accelerator.init_trackers("Verifier-Training", config=vars(args))

    logger = get_logger("train_verifier", output_dir=output_dir, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    device = accelerator.device
    logger.info(f"Args: {args}")

    # ---------------- frozen policy (features + normalization) ----------------
    logger.info(f"Loading frozen policy from {args.policy_ckpt}")
    policy = SmolVLMVLA.from_pretrained(args.policy_ckpt)
    if args.norm_stats_path:
        policy.action_space = build_action_space(
            policy.action_mode, norm_stats_path=args.norm_stats_path
        )
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    # ---------------- frozen generator head (negatives) ----------------
    generator = None
    if args.generator_ckpt:
        logger.info(f"Loading one-step generator from {args.generator_ckpt}")
        generator = SmolVLMVLA.from_pretrained(args.generator_ckpt)
        if args.norm_stats_path:
            generator.action_space = build_action_space(
                generator.action_mode, norm_stats_path=args.norm_stats_path
            )
        # The generator was distilled with a frozen VLM identical to the
        # policy backbone, so its own VLM is redundant: drop it and reuse
        # the policy's features when sampling proposals.
        generator.vlm = None
        generator.to(device).eval()
        for p in generator.parameters():
            p.requires_grad_(False)

    # ---------------- verifier ----------------
    vlm_hidden_size = policy.vlm.config.text_config.hidden_size
    v_config = ActionEnergyVerifierConfig(
        vlm_hidden_size=vlm_hidden_size,
        dim_action=policy.action_space.dim_action,
        dim_proprio=getattr(policy.action_space, "dim_proprio", policy.action_space.dim_action),
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        num_pool_tokens=args.num_pool_tokens,
        max_num_actions=policy.num_actions,
    )
    verifier = ActionEnergyVerifier(v_config)
    logger.info(f"Verifier params: {verifier.num_parameters / 1e6:.1f}M")

    processor = SmolVLMVLAProcessor.from_pretrained(policy.config.smolvlm_model_path)

    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=policy.num_actions,
        action_mode=policy.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    optim = AdamW(verifier.parameters(), lr=args.learning_rate,
                  weight_decay=args.weight_decay)
    verifier, optim = accelerator.prepare(verifier, optim)
    verifier.train()

    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start verifier training for {args.iters} iterations")

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        lr = cosine_lr(global_step, args.warmup_steps, args.iters,
                       args.learning_rate, args.min_lr_ratio)
        for g in optim.param_groups:
            g["lr"] = lr

        with torch.no_grad():
            enc = policy.forward_vlm_efficient(
                inputs["image_input"], inputs["image_mask"], inputs["input_ids"]
            )
            vlm_features = enc["vlm_features"]
            proprio_norm, action_norm = policy._normalize_inputs(
                inputs["proprio"], inputs["action"]
            )

            B, T, D = action_norm.shape
            negatives = []
            gen_mask = None

            # (a) generator proposals on the shared features
            if generator is not None and args.num_neg_gen > 0:
                gen_out = generator.sample_action_candidates(
                    input_ids=None, image_input=None, image_mask=None,
                    proprio=inputs["proprio"],
                    num_samples=args.num_neg_gen,
                    steps=args.gen_steps,
                    vlm_features=vlm_features,
                )
                gen_norm = gen_out["actions_norm"]                 # [B, Kg, T, D]
                dist = (gen_norm - action_norm.unsqueeze(1)).pow(2).mean(dim=(2, 3)).sqrt()
                gen_mask = dist >= args.min_neg_dist               # [B, Kg]
                negatives.append(gen_norm)

            # (b) Gaussian-perturbed ground truth
            if args.num_neg_perturb > 0:
                stds = torch.empty(B, args.num_neg_perturb, 1, 1, device=device).uniform_(
                    args.perturb_std_min, args.perturb_std_max
                )
                pert = action_norm.unsqueeze(1) + stds * torch.randn(
                    B, args.num_neg_perturb, T, D, device=device
                )
                negatives.append(pert)

            # (c) in-batch shuffled ground truth
            if args.num_neg_shuffle > 0 and B > 1:
                shuf = torch.stack(
                    [action_norm.roll(shifts=k + 1, dims=0)
                     for k in range(min(args.num_neg_shuffle, B - 1))],
                    dim=1,
                )
                negatives.append(shuf)

            neg = torch.cat(negatives, dim=1)                      # [B, N, T, D]
            N = neg.shape[1]
            neg_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            if gen_mask is not None:
                neg_mask[:, : gen_mask.shape[1]] = gen_mask

        # ---------------- energies + InfoNCE ----------------
        e_pos = verifier(vlm_features, proprio_norm, action_norm)                 # [B]
        e_neg = accelerator.unwrap_model(verifier).score_candidates(
            vlm_features, proprio_norm, neg
        )                                                                          # [B, N]
        out = info_nce_loss(e_pos, e_neg, temperature=args.temperature,
                            neg_mask=neg_mask)

        accelerator.backward(out["loss"])
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(verifier.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {
                "infonce_loss": float(out["loss"].detach()),
                "verifier_acc": float(out["acc"]),
                "energy_margin": float(out["margin"]),
                "num_negatives": float(N),
                "masked_frac": float((~neg_mask).float().mean()),
                "lr": lr,
            }
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['infonce_loss']:.4f} acc={logs['verifier_acc']:.3f} "
                    f"margin={logs['energy_margin']:.3f} lr={lr:.2e} ({dt:.2f}s/it)"
                )

        global_step += 1
        if accelerator.is_main_process and (
            global_step == args.iters or global_step % args.save_interval == 0
        ):
            save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
            accelerator.print(f"💾 Saving verifier to {save_dir}")
            accelerator.unwrap_model(verifier).save_pretrained(save_dir)
            with open(os.path.join(save_dir, "state.json"), "w") as f:
                json.dump({"global_step": global_step,
                           "policy_ckpt": args.policy_ckpt,
                           "generator_ckpt": args.generator_ckpt}, f)

        if global_step >= args.iters:
            break

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Verifier training", parents=[get_args_parser()])
    main(parser.parse_args())
