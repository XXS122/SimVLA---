"""
MeanFlow One-Step Distillation (Task-2, A100 / SAPI site)

Distills a trained SimVLA flow-matching policy into a one-step MeanFlow
generator:

  1. Load the SFT checkpoint (teacher).
  2. Build a student with identical architecture + use_meanflow=True; the
     interval conditioning is zero-initialized, so the student starts
     numerically identical to the teacher.
  3. Freeze the VLM backbone; train only the action head with the MeanFlow
     objective (JVP-based average-velocity identity).
  4. Optionally use the frozen teacher head's predicted velocity as v_t
     (--teacher_velocity) instead of the conditional velocity.

The resulting checkpoint loads with SmolVLMVLA.from_pretrained() and
generates actions in a single forward pass (generate_actions(steps=1)).

Usage:
    python train_meanflow_distill.py \
        --teacher_ckpt ./runs/simvla_libero_small/ckpt-200000 \
        --train_metas_path $LIBERO_DATASETS/libero_train.json \
        --norm_stats_path ./norm_stats/libero_norm.json \
        --output_dir ./runs/meanflow_small
"""

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.transformer_smolvlm import Attention
from models.action_hub import build_action_space

from train_smolvlm import get_logger  # reuse the project logger

try:
    import wandb  # noqa: F401
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args_parser():
    parser = argparse.ArgumentParser("MeanFlow distillation", add_help=False)

    # I/O
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to the trained SimVLA (SFT) checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./runs/meanflow_distill")

    # Data (same conventions as train_smolvlm.py)
    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--action_mode", type=str, default="libero_joint")

    # MeanFlow objective
    parser.add_argument("--meanflow_ratio", type=float, default=0.5,
                        help="Fraction of samples trained with r < t")
    parser.add_argument("--adaptive_p", type=float, default=1.0)
    parser.add_argument("--adaptive_c", type=float, default=1e-3)
    parser.add_argument("--teacher_velocity", action="store_true", default=False,
                        help="Use the frozen teacher head's v(x_t, t) as the "
                             "instantaneous velocity instead of noise - action")

    # Optimizer / schedule
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
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


def disable_fused_attention(module: torch.nn.Module):
    """Fused SDPA kernels do not support forward-mode AD (needed for the
    MeanFlow JVP); force the math attention path in the action head."""
    n = 0
    for m in module.modules():
        if isinstance(m, Attention):
            m.fused_attn = False
            n += 1
    return n


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

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=log_with, project_dir=output_dir,
                              kwargs_handlers=[ddp_kwargs])

    tracker_config = {k: v for k, v in vars(args).items()}
    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"meanflow-{time.strftime('%Y%m%d-%H%M%S')}"}},
        )
    else:
        accelerator.init_trackers("MeanFlow-Distill", config=tracker_config)

    logger = get_logger("meanflow_distill", output_dir=output_dir, accelerator=accelerator)
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # ---------------- teacher ----------------
    logger.info(f"Loading teacher from {args.teacher_ckpt}")
    teacher = SmolVLMVLA.from_pretrained(args.teacher_ckpt)
    if teacher.use_meanflow:
        raise ValueError("Teacher checkpoint is already a MeanFlow model.")

    # ---------------- student ----------------
    student_config = copy.deepcopy(teacher.config)
    student_config.use_meanflow = True
    student_config.meanflow_ratio = args.meanflow_ratio
    student_config.meanflow_adaptive_p = args.adaptive_p
    student_config.meanflow_adaptive_c = args.adaptive_c

    student = SmolVLMVLA(student_config)
    # interval_proj keys are missing from the teacher state dict (zero-init)
    missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
    logger.info(f"Student init from teacher: missing={missing}, unexpected={unexpected}")
    if any("interval_proj" not in k for k in missing):
        raise RuntimeError(f"Unexpected missing keys beyond interval_proj: {missing}")

    if args.norm_stats_path:
        student.action_space = build_action_space(
            student.action_mode, norm_stats_path=args.norm_stats_path
        )

    # Freeze VLM backbone; only the action head learns.
    for p in student.vlm.parameters():
        p.requires_grad_(False)
    student.vlm.eval()
    student.meanflow_detach_vlm = True

    # The MeanFlow JVP needs the math attention path.
    n_attn = disable_fused_attention(student.transformer)
    logger.info(f"Disabled fused attention in {n_attn} student attention modules")

    # Optional frozen teacher velocity field. Bypass nn.Module registration so
    # the teacher head is never serialized into the student checkpoint.
    if args.teacher_velocity:
        teacher_head = teacher.transformer
        for p in teacher_head.parameters():
            p.requires_grad_(False)
        teacher_head.eval()
        object.__setattr__(student, "teacher_transformer", teacher_head)
        logger.info("Using frozen teacher head as the velocity field v_t")
    del teacher

    processor = SmolVLMVLAProcessor.from_pretrained(student_config.smolvlm_model_path)

    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=student.num_actions,
        action_mode=student.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    trainable = [p for p in student.transformer.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Trainable action-head params: {n_trainable / 1e6:.1f}M")
    optim = AdamW(trainable, lr=args.learning_rate,
                  weight_decay=args.weight_decay, betas=tuple(args.betas))

    student, optim = accelerator.prepare(student, optim)
    if args.teacher_velocity:
        unwrapped = accelerator.unwrap_model(student)
        unwrapped.teacher_transformer.to(accelerator.device)

    student.train()
    accelerator.unwrap_model(student).vlm.eval()

    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start MeanFlow distillation for {args.iters} iterations")

    for batch in train_dataloader:
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.to(accelerator.device, non_blocking=True) for k, v in inputs.items()}

        lr = cosine_lr(global_step, args.warmup_steps, args.iters,
                       args.learning_rate, args.min_lr_ratio)
        for g in optim.param_groups:
            g["lr"] = lr

        loss_dict: Dict[str, torch.Tensor] = student(**inputs)
        loss = sum(loss_dict.values())

        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            raw = getattr(accelerator.unwrap_model(student), "_last_meanflow_raw_mse", None)
            if raw is not None:
                logs["meanflow_raw_mse"] = float(raw)
            logs["lr"] = lr
            accelerator.log(logs, step=global_step)
            if accelerator.is_main_process:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs.get('meanflow_loss', 0.0):.4f} "
                    f"raw_mse={logs.get('meanflow_raw_mse', 0.0):.4f} "
                    f"lr={lr:.2e} ({dt:.2f}s/it)"
                )

        global_step += 1
        if accelerator.is_main_process and (
            global_step == args.iters or global_step % args.save_interval == 0
        ):
            save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
            accelerator.print(f"💾 Saving one-step generator to {save_dir}")
            accelerator.unwrap_model(student).save_pretrained(save_dir, safe_serialization=True)
            with open(os.path.join(save_dir, "state.json"), "w") as f:
                json.dump({"global_step": global_step,
                           "teacher_ckpt": args.teacher_ckpt}, f)

        if global_step >= args.iters:
            break

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MeanFlow distillation", parents=[get_args_parser()])
    main(parser.parse_args())
