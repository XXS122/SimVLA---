"""
Flow-DPO: preference optimization for a flow-matching VLA policy.

Adapts the Diffusion-DPO ELBO surrogate to velocity parameterization.
For a matched-state pair (a_w preferred, a_l dispreferred) at state s:

  d_theta(a) = || v_theta(x_t, t | s) - u_t(a) ||^2     (flow residual)
  d_ref(a)   = same with the frozen reference policy
  margin     = (d_theta(a_w) - d_ref(a_w)) - (d_theta(a_l) - d_ref(a_l))
  L_dpo      = -log sigmoid(-beta * margin)

The same t and the same noise are used for both actions (variance
reduction). The frozen reference is the BC checkpoint itself, which
doubles as the KL anchor that prevents drift and mode collapse.

The VLM backbone is FROZEN during DPO: both policies share one VLM
feature pass, and the reference is just a frozen copy of the action
transformer (cheap in memory).

Optional BC term on the winner (bc_coef * d_theta(a_w)) keeps the policy
on the data manifold.

Usage:
    python train_flow_dpo.py \
        --models ./runs/exp_tds/ckpt-200000 \
        --pairs ./evaluation/libero/dpo_data/branches.jsonl \
        --norm_stats_path ./norm_stats/libero_norm.json \
        --output_dir ./runs/flow_dpo \
        --beta 50.0 --bc_coef 1.0 --learning_rate 1e-5 --iters 5000
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.dpo_pairs import DPOPairDataset
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.action_hub import build_action_space


def get_args_parser():
    parser = argparse.ArgumentParser("Flow-DPO training", add_help=False)
    parser.add_argument("--models", type=str, required=True,
                        help="BC checkpoint to start from (also the reference)")
    parser.add_argument("--pairs", type=str, nargs="+", required=True,
                        help="branches.jsonl file(s) from the pair collector "
                             "(pass several to merge multiple collection runs)")
    parser.add_argument("--output_dir", type=str, default="./runs/flow_dpo")
    parser.add_argument("--smolvlm_model_path", type=str,
                        default=os.environ.get("SIMVLA_SMOLVLM_MODEL",
                                               "HuggingFaceTB/SmolVLM-500M-Instruct"))
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument("--action_mode", type=str, default="libero_joint")

    parser.add_argument("--beta", type=float, default=50.0,
                        help="DPO inverse temperature on the residual margin")
    parser.add_argument("--bc_coef", type=float, default=1.0,
                        help="Weight of the winner BC (flow) term")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pairs_per_snapshot", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--min_branch_rate", type=float, default=None,
                        help="Keep only snapshots with branch success rate >= "
                             "this (filters non-pivotal states; round-1 label "
                             "noise came from 7/8-type snapshots)")
    parser.add_argument("--max_branch_rate", type=float, default=None,
                        help="Keep only snapshots with branch success rate <= this")
    parser.add_argument("--arm_filter", type=str, nargs="+", default=None,
                        choices=["spike", "control"],
                        help="Keep only snapshots triggered by these arms "
                             "(e.g. --arm_filter spike to test whether "
                             "uncertainty-triggered branch points carry "
                             "cleaner preference signal than fixed-point ones)")

    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def flow_dpo_terms(
    transformer,
    ref_transformer,
    vlm_features: torch.Tensor,
    proprio_norm: torch.Tensor,
    action_w_norm: torch.Tensor,
    action_l_norm: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    beta: float,
):
    """
    Compute the Flow-DPO loss terms for one batch of matched-state pairs.

    Returns dict with: loss_dpo, loss_bc (winner residual), margin mean,
    implicit_acc (fraction of pairs where the policy already prefers the
    winner relative to the reference).
    """
    t_exp = t.view(-1, 1, 1)

    def residual(model, action_norm):
        x_t = t_exp * noise + (1 - t_exp) * action_norm
        u_t = noise - action_norm
        v = model(vlm_features=vlm_features, action_with_noise=x_t,
                  proprio=proprio_norm, t=t)
        return torch.square(v - u_t).mean(dim=(1, 2))  # [B]

    d_w = residual(transformer, action_w_norm)
    d_l = residual(transformer, action_l_norm)
    with torch.no_grad():
        d_w_ref = residual(ref_transformer, action_w_norm)
        d_l_ref = residual(ref_transformer, action_l_norm)

    margin = (d_w - d_w_ref) - (d_l - d_l_ref)  # want negative
    loss_dpo = -F.logsigmoid(-beta * margin).mean()
    loss_bc = d_w.mean()

    return {
        "loss_dpo": loss_dpo,
        "loss_bc": loss_bc,
        "margin": margin.mean().detach(),
        "implicit_acc": (margin < 0).float().mean().detach(),
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Model: policy + frozen reference ----------------
    print(f"Loading BC checkpoint: {args.models}")
    model = SmolVLMVLA.from_pretrained(
        args.models, smolvlm_model_path=args.smolvlm_model_path
    )
    if args.norm_stats_path:
        model.action_space = build_action_space(
            args.action_mode, norm_stats_path=args.norm_stats_path
        )
    model = model.to(device)

    # Freeze the VLM: DPO trains the action transformer only
    for p in model.vlm.parameters():
        p.requires_grad = False
    model.vlm.eval()

    ref_transformer = copy.deepcopy(model.transformer)
    for p in ref_transformer.parameters():
        p.requires_grad = False
    ref_transformer.eval()

    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    # ---------------- Data ----------------
    dataset = DPOPairDataset(
        records_path=args.pairs,
        image_size=args.image_size,
        training=True,
        max_pairs_per_snapshot=args.max_pairs_per_snapshot,
        min_branch_rate=args.min_branch_rate,
        max_branch_rate=args.max_branch_rate,
        arm_filter=args.arm_filter,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    optim = AdamW(
        [p for p in model.transformer.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---------------- Train ----------------
    model.transformer.train()
    step, t0 = 0, time.time()
    print(f"Start Flow-DPO: beta={args.beta} bc_coef={args.bc_coef} "
          f"lr={args.learning_rate} iters={args.iters}")

    while step < args.iters:
        for batch in loader:
            lang = processor.encode_language(list(batch["language_instruction"]))
            input_ids = lang["input_ids"].to(device)
            image_input = batch["image_input"].to(device)
            image_mask = batch["image_mask"].to(device)
            proprio = batch["proprio"].to(device)
            action_w = batch["action_w"].to(device)
            action_l = batch["action_l"].to(device)

            # Shared frozen VLM features for policy AND reference
            with torch.no_grad():
                enc = model.forward_vlm_efficient(image_input, image_mask, input_ids)
                vlm_features = enc["vlm_features"].detach()

            proprio_norm = model.action_space.normalize_state(proprio)
            a_w = model.action_space.normalize_action(action_w)
            a_l = model.action_space.normalize_action(action_l)

            B = a_w.shape[0]
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((B,)) * 0.999 + 0.001
            noise = torch.randn_like(a_w)

            terms = flow_dpo_terms(
                model.transformer, ref_transformer, vlm_features,
                proprio_norm, a_w, a_l, t, noise, args.beta,
            )
            loss = terms["loss_dpo"] + args.bc_coef * terms["loss_bc"]

            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.transformer.parameters(), args.max_grad_norm
                )
            optim.step()
            optim.zero_grad()

            if step % args.log_interval == 0:
                dt = (time.time() - t0) / max(1, args.log_interval)
                t0 = time.time()
                print(
                    f"[{step}/{args.iters}] "
                    f"dpo={terms['loss_dpo'].item():.4f} "
                    f"bc={terms['loss_bc'].item():.4f} "
                    f"margin={terms['margin'].item():+.5f} "
                    f"implicit_acc={terms['implicit_acc'].item():.3f} "
                    f"({dt:.2f}s/it)",
                    flush=True,
                )

            step += 1
            if step % args.save_interval == 0 or step == args.iters:
                save_dir = output_dir / f"ckpt-dpo-{step}"
                print(f"Saving to {save_dir}")
                model.save_pretrained(save_dir, safe_serialization=True)
                with open(save_dir / "state.json", "w") as f:
                    json.dump({"global_step": step, "beta": args.beta,
                               "bc_coef": args.bc_coef}, f)
            if step >= args.iters:
                break

    print("Flow-DPO training done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flow-DPO", parents=[get_args_parser()])
    main(parser.parse_args())
