#!/usr/bin/env python3
"""
reserve_gpu.py — hold spare GPU memory on a shared server so co-tenants
can't schedule work on the GPU you are training on.

It only *holds* memory (a few big idle tensors); it runs no compute
kernels, so it does not steal SM cycles from your own training job. With
(near) zero free memory on the card, another user's process fails to
allocate at launch and cannot contend for the GPU.

Run it in its own tmux pane, on the SAME physical GPU as your job, and
leave enough headroom for your job's PEAK usage (LAM augmentation steps
run the vision backbone twice, ~1.7x a normal step). Press Ctrl-C to
release immediately.

Examples (training runs on physical GPU 6):
    # leave 18 GB free for your own job's peak, grab the rest
    CUDA_VISIBLE_DEVICES=6 python reserve_gpu.py --leave-free 18

    # or grab an exact amount
    CUDA_VISIBLE_DEVICES=6 python reserve_gpu.py --reserve 25

Tips:
    * Set --leave-free a bit ABOVE your job's peak memory (watch
      `nvidia-smi`); if training later OOMs, Ctrl-C and relaunch with a
      larger --leave-free.
    * Be a good cluster citizen: only reserve the GPU you are actually
      using, and release it (Ctrl-C) the moment your run finishes.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


def _gb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def main():
    p = argparse.ArgumentParser(description="Hold spare GPU memory to keep a shared GPU to yourself.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--leave-free", type=float, default=18.0,
                   help="GB to leave free for your own job's peak (default 18)")
    g.add_argument("--reserve", type=float, default=None,
                   help="GB to grab explicitly (overrides --leave-free)")
    p.add_argument("--chunk", type=float, default=0.5,
                   help="allocation block size in GB (default 0.5)")
    p.add_argument("--poll", type=float, default=60.0,
                   help="status print interval in seconds (default 60)")
    p.add_argument("--refill", action="store_true",
                   help="periodically re-grab memory that became free "
                        "(e.g. after a co-tenant exits). Use with care: it "
                        "will also swallow memory your own job later tries to "
                        "grow into, so keep --leave-free above your peak.")
    args = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible — set CUDA_VISIBLE_DEVICES to the GPU you want to hold")
    if torch.cuda.device_count() != 1:
        print(f"warning: {torch.cuda.device_count()} GPUs visible; this holds cuda:0 only. "
              f"Set CUDA_VISIBLE_DEVICES=<one id> to target a specific card.", file=sys.stderr)

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    name = torch.cuda.get_device_name(dev)
    total = torch.cuda.mem_get_info(dev)[1]

    chunk_bytes = max(1, int(args.chunk * (1024 ** 3)))
    blocks: list = []

    def held_bytes() -> int:
        return sum(b.numel() for b in blocks)

    def grab_up_to_target():
        """Allocate blocks until free memory drops to the headroom target."""
        while True:
            free, _ = torch.cuda.mem_get_info(dev)
            if args.reserve is not None:
                remaining = int(args.reserve * (1024 ** 3)) - held_bytes()
            else:
                remaining = int(free - args.leave_free * (1024 ** 3))
            if remaining <= 0:
                break
            this = min(chunk_bytes, remaining)
            try:
                blocks.append(torch.empty(this, dtype=torch.uint8, device=dev))
            except RuntimeError:
                # fragmentation / raced with another allocator — stop for now
                break

    free0, _ = torch.cuda.mem_get_info(dev)
    print(f"GPU: {name}  total={_gb(total):.1f}GB  free={_gb(free0):.1f}GB")
    grab_up_to_target()
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info(dev)
    print(f"holding {_gb(held_bytes()):.1f}GB  ->  free now {_gb(free1):.1f}GB. "
          f"Ctrl-C to release." + ("  [refill on]" if args.refill else ""))

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))

    while not stop["v"]:
        time.sleep(args.poll)
        if args.refill:
            grab_up_to_target()
        free, tot = torch.cuda.mem_get_info(dev)
        print(f"[holding {_gb(held_bytes()):.1f}GB] free={_gb(free):.1f}GB / {_gb(tot):.1f}GB",
              flush=True)

    blocks.clear()
    torch.cuda.empty_cache()
    print("released GPU memory, exiting.")


if __name__ == "__main__":
    main()
