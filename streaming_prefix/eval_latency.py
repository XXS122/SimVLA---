"""
Phase C, diagnostic 2: end-to-end per-control-step latency, streaming
prefix vs. full re-encode.

Measures the actual realized speedup of replacing the fused text-model
forward with the trained PrefixStateUpdater, to confirm Phase A's
predicted headroom (VLM prefix = 45.5% of per-step latency at 10 Euler
steps; text-model = 72% of that prefix) is realized in practice.

Two per-step latency configurations, both timed on the real backbone
(weights don't affect timing, but we load the trained updater so the
graph shape is exactly the deployment one):

  full     : cheap_forward + expensive_forward (real fused text model) +
             action head (Euler integration)  -- the baseline
  streaming: cheap_forward + PrefixStateUpdater (replaces text model) +
             action head                        -- our method

A reset step (periodic full re-encode) costs the same as `full`; with a
reset period P, the amortized per-step latency is
  (full + (P-1) * streaming) / P.
Reported across a range of Euler step counts and reset periods, alongside
achievable control frequency (Hz).

Usage:
    python -m streaming_prefix.eval_latency \
        --updater_ckpt <...>/updater_final.pt \
        --smolvlm_model_path $SIMVLA_SMOLVLM_MODEL \
        --euler_steps 1 5 10 --reset_periods 5 10 20 \
        --repeats 30 --output latency_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForImageTextToText

from latent_action import config as C
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.action_hub import build_action_space
from streaming_prefix.models import VLMPrefixTeacher, load_updater


def get_args_parser():
    p = argparse.ArgumentParser("streaming latency eval", add_help=False)
    p.add_argument("--updater_ckpt", type=str, required=True)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    p.add_argument("--action_mode", type=str, default="libero_joint")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--num_actions", type=int, default=10)
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=24)
    p.add_argument("--euler_steps", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--reset_periods", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--output", type=str, default=None)
    return p


def _dtype(name):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _time_fn(fn: Callable[[], None], warmup: int, repeats: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device == "cuda":
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); fn(); times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times)


@torch.no_grad()
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(args.dtype) if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_model_path, dtype=dtype, trust_remote_code=True).to(device)
    teacher = VLMPrefixTeacher(vlm).to(device)
    updater = load_updater(args.updater_ckpt).to(device=device, dtype=dtype).eval()

    # action head from a freshly-built SmolVLMVLA (timing is weight-independent)
    cfg = SmolVLMVLAConfig(
        smolvlm_model_path=args.smolvlm_model_path, hidden_size=args.hidden_size,
        depth=args.depth, num_heads=args.num_heads, action_mode=args.action_mode,
        num_actions=args.num_actions, image_size=args.image_size, num_views=args.num_views)
    # reuse the teacher's already-loaded VLM to avoid a second copy in memory
    vla = SmolVLMVLA.__new__(SmolVLMVLA)
    torch.nn.Module.__init__(vla)
    vla.config = cfg
    vla.num_actions = args.num_actions
    vla.action_space = build_action_space(args.action_mode)
    from models.transformer_smolvlm import SmolVLMActionTransformer
    vla.transformer = SmolVLMActionTransformer(
        hidden_size=cfg.hidden_size, vlm_hidden_size=vlm.config.text_config.hidden_size,
        depth=cfg.depth, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio,
        dim_action=vla.action_space.dim_action,
        dim_propio=getattr(vla.action_space, "dim_proprio", vla.action_space.dim_action),
        dim_time=cfg.dim_time, max_len_seq=cfg.max_len_seq, use_adaln=False,
    ).to(device=device, dtype=dtype)
    vla.transformer.eval()

    B, V, S = 1, args.num_views, args.image_size
    images = torch.randn(B, V, 3, S, S, device=device, dtype=dtype) * 0.1
    mask = torch.ones(B, V, dtype=torch.bool, device=device)
    input_ids = torch.randint(5, 1000, (B, args.seq_len), device=device)
    proprio = torch.randn(B, 8, device=device, dtype=dtype)

    combined, attn = teacher.cheap_forward(images, mask, input_ids)
    cached_state = teacher.expensive_forward(combined, attn)

    def action_head(vlm_features, steps):
        Dact = vla.action_space.dim_action
        x = torch.randn(B, args.num_actions, Dact, device=device, dtype=dtype)
        dt = -1.0 / steps
        t = 1.0
        while t > -dt / 2:
            tt = torch.full((B,), t, device=device, dtype=dtype)
            v = vla.transformer(vlm_features=vlm_features, action_with_noise=x, proprio=proprio, t=tt)
            x = x + dt * v
            t = t + dt

    def full_step(steps):
        c, a = teacher.cheap_forward(images, mask, input_ids)
        feats = teacher.expensive_forward(c, a)
        action_head(feats, steps)

    def streaming_step(steps):
        c, a = teacher.cheap_forward(images, mask, input_ids)
        feats = updater(cached_state.to(dtype), c.to(dtype))
        action_head(feats, steps)

    results = {"config": vars(args), "device": device, "dtype": str(dtype), "by_euler_steps": {}}
    print(f"\n{'steps':>6} {'full_ms':>9} {'stream_ms':>10} {'speedup':>8}   amortized-by-reset-period")
    for steps in args.euler_steps:
        full_ms = _time_fn(lambda: full_step(steps), args.warmup, args.repeats, device)
        stream_ms = _time_fn(lambda: streaming_step(steps), args.warmup, args.repeats, device)
        row = {"full_ms": full_ms, "streaming_ms": stream_ms,
               "per_step_speedup": full_ms / stream_ms, "amortized": {}}
        amort_str = []
        for P in args.reset_periods:
            amort = (full_ms + (P - 1) * stream_ms) / P
            row["amortized"][str(P)] = {"ms": amort, "speedup": full_ms / amort,
                                        "hz": 1000.0 / amort}
            amort_str.append(f"P={P}:{full_ms/amort:.2f}x/{1000.0/amort:.0f}Hz")
        print(f"{steps:>6} {full_ms:>9.2f} {stream_ms:>10.2f} {full_ms/stream_ms:>7.2f}x   " + "  ".join(amort_str))
        results["by_euler_steps"][str(steps)] = row

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsaved -> {out}")
    print("\n>>> per_step_speedup is the full-vs-streaming ratio at a non-reset step; "
          "amortized folds in one full re-encode every reset_period steps (the reset "
          "period Phase C's drift eval says is safe).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("streaming latency eval", parents=[get_args_parser()])
    main(parser.parse_args())
