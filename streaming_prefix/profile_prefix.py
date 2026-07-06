"""
Phase A: profile per-control-step inference latency of SmolVLMVLA, split
into (1) VLM prefix re-encoding (forward_vlm_efficient: vision tower +
connector + fused text-model forward) and (2) the flow-matching action
head's Euler integration loop.

No trained checkpoint needed -- timing depends on the compute graph, not
the weight values -- so this can (and should) run before any streaming/
caching model is built, to establish the ceiling on possible speedup.

Usage:
    python -m streaming_prefix.profile_prefix \
        --smolvlm_model_path $SIMVLA_SMOLVLM_MODEL \
        --euler_steps 1 5 10 20 --repeats 20 --warmup 5 \
        --output profile_report.json

Add --breakdown to also split the VLM forward itself into
vision-tower / connector / fused-text-model timings (informs where the
state-update operator should intervene: only the vision tower, or also
the language model's processing of image tokens).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from latent_action import config as C
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.action_hub import build_action_space


def get_args_parser():
    p = argparse.ArgumentParser("prefix profiling", add_help=False)
    p.add_argument("--smolvlm_model_path", type=str, default=C.SMOLVLM_MODEL)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="optional trained SmolVLMVLA checkpoint dir; if omitted, "
                        "builds a freshly-initialized model of the given size "
                        "(timing is weight-independent)")
    p.add_argument("--action_mode", type=str, default="libero_joint")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--num_actions", type=int, default=10)
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=24, help="tokenized instruction length")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--euler_steps", type=int, nargs="+", default=[1, 5, 10, 20])
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--breakdown", action="store_true", default=False,
                   help="also split the VLM forward into vision/connector/text_model")
    p.add_argument("--output", type=str, default=None)
    return p


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _time_fn(fn: Callable[[], None], warmup: int, repeats: int, device: str) -> dict:
    """CUDA-event (or wall-clock on CPU) timing with proper synchronization."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        if device == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "n": repeats,
    }


def build_model(args, device: str, dtype: torch.dtype) -> SmolVLMVLA:
    if args.checkpoint:
        print(f"loading trained checkpoint: {args.checkpoint}")
        model = SmolVLMVLA.from_pretrained(args.checkpoint)
    else:
        print("no --checkpoint given: building a freshly-initialized model "
              "(fine for timing -- weight values don't affect latency)")
        cfg = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
            action_mode=args.action_mode, num_actions=args.num_actions,
            image_size=args.image_size, num_views=args.num_views,
        )
        model = SmolVLMVLA(cfg)
        model.action_space = build_action_space(args.action_mode)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def make_inputs(args, device: str, dtype: torch.dtype):
    B, V, S = args.batch_size, args.num_views, args.image_size
    return dict(
        input_ids=torch.randint(5, 1000, (B, args.seq_len), device=device),
        image_input=(torch.randn(B, V, 3, S, S, device=device, dtype=dtype) * 0.1),
        image_mask=torch.ones(B, V, dtype=torch.bool, device=device),
        proprio=torch.randn(B, 8, device=device, dtype=dtype),
    )


@torch.no_grad()
def profile_breakdown(model: SmolVLMVLA, inputs: dict, warmup: int, repeats: int, device: str) -> dict:
    """Split forward_vlm_efficient into vision tower / connector / fused
    text-model forward. Mirrors the internal steps of
    models/modeling_smolvlm_vla.py::forward_vlm_efficient for profiling
    granularity only.
    """
    image_input, image_mask, input_ids = inputs["image_input"], inputs["image_mask"], inputs["input_ids"]
    B, V = image_input.shape[:2]
    flat_images = image_input.flatten(0, 1)
    flat_mask = image_mask.view(-1).bool()
    valid_images = flat_images[flat_mask]

    def run_vision():
        model.vlm.model.vision_model(pixel_values=valid_images, output_hidden_states=True, return_dict=True)

    vision_out = model.vlm.model.vision_model(pixel_values=valid_images, return_dict=True).last_hidden_state
    connector = getattr(model.vlm.model, "connector", None) or getattr(model.vlm.model, "multi_modal_projector")

    def run_connector():
        connector(vision_out)

    image_features = connector(vision_out)
    hidden_size = image_features.shape[-1]
    num_patches = image_features.shape[1]
    text_embeds = model.vlm.model.text_model.get_input_embeddings()(input_ids)

    full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
    full_image_features[flat_mask] = image_features
    full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
    valid_per_sample = image_mask.sum(dim=1).int()

    combined_list, max_len = [], 0
    for b in range(B):
        n = valid_per_sample[b].item()
        img = full_image_features[b, :n].reshape(-1, hidden_size)
        combined = torch.cat([img, text_embeds[b]], dim=0)
        combined_list.append(combined)
        max_len = max(max_len, combined.shape[0])
    padded = torch.zeros(B, max_len, hidden_size, device=image_input.device, dtype=image_features.dtype)
    attn = torch.zeros(B, max_len, device=image_input.device, dtype=torch.long)
    for b, c in enumerate(combined_list):
        padded[b, :c.shape[0]] = c
        attn[b, :c.shape[0]] = 1

    def run_text_model():
        model.vlm.model.text_model(inputs_embeds=padded, attention_mask=attn, return_dict=True)

    return {
        "vision_tower_ms": _time_fn(run_vision, warmup, repeats, device),
        "connector_ms": _time_fn(run_connector, warmup, repeats, device),
        "fused_text_model_ms": _time_fn(run_text_model, warmup, repeats, device),
        "prefix_seq_len": max_len,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype(args.dtype) if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}")

    model = build_model(args, device, dtype)
    inputs = make_inputs(args, device, dtype)

    with torch.no_grad():
        def run_vlm():
            model.forward_vlm_efficient(inputs["image_input"], inputs["image_mask"], inputs["input_ids"])

        vlm_timing = _time_fn(run_vlm, args.warmup, args.repeats, device)
        print(f"\nVLM prefix forward (1x per control step): "
              f"{vlm_timing['mean_ms']:.2f} +/- {vlm_timing['std_ms']:.2f} ms")

        breakdown = None
        if args.breakdown:
            breakdown = profile_breakdown(model, inputs, args.warmup, args.repeats, device)
            print(f"  vision tower  : {breakdown['vision_tower_ms']['mean_ms']:.2f} ms")
            print(f"  connector     : {breakdown['connector_ms']['mean_ms']:.2f} ms")
            print(f"  text model    : {breakdown['fused_text_model_ms']['mean_ms']:.2f} ms "
                  f"(prefix seq_len={breakdown['prefix_seq_len']})")

        results = {"config": vars(args), "device": device, "dtype": str(dtype),
                   "vlm_forward_ms": vlm_timing, "breakdown": breakdown, "by_euler_steps": {}}

        print(f"\n{'steps':>6} {'total_ms':>10} {'vlm_ms':>10} {'action_ms':>10} {'vlm_frac':>9} {'hz':>8}")
        for steps in args.euler_steps:
            def run_full():
                model.generate_actions(inputs["input_ids"], inputs["image_input"],
                                       inputs["image_mask"], inputs["proprio"], steps=steps)

            total_timing = _time_fn(run_full, args.warmup, args.repeats, device)
            total_ms = total_timing["mean_ms"]
            vlm_ms = vlm_timing["mean_ms"]
            action_ms = max(0.0, total_ms - vlm_ms)
            vlm_frac = vlm_ms / total_ms if total_ms > 0 else float("nan")
            hz = 1000.0 / total_ms if total_ms > 0 else float("nan")
            print(f"{steps:>6} {total_ms:>10.2f} {vlm_ms:>10.2f} {action_ms:>10.2f} {vlm_frac:>9.1%} {hz:>8.1f}")
            results["by_euler_steps"][str(steps)] = {
                "total": total_timing, "vlm_ms": vlm_ms, "action_ms": action_ms,
                "vlm_fraction": vlm_frac, "achievable_hz": hz,
            }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved -> {out}")

    print("\n>>> interpretation: vlm_frac is the ceiling on speedup from streaming/"
          "caching the prefix alone (action head is untouched by this design). "
          "If vlm_frac is low, this direction has limited headroom regardless of "
          "how good the state-update operator is.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("prefix profiling", parents=[get_args_parser()])
    main(parser.parse_args())
