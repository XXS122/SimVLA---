"""
Frame-pair data loading for latent action model training / labeling.

Reads the same LIBERO HDF5 files and metadata JSON as the SimVLA training
pipeline (datasets/domain_handler/libero_hdf5.py) and reproduces its image
preprocessing exactly:
  - 180 degree rotation ([::-1, ::-1]) of agentview and wrist images
  - bicubic resize to image_size x image_size
  - ImageNet mean/std normalization

Ground-truth actions are carried along ONLY for the probe / diagnostics —
the LAM training loss never sees them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def demo_key_of(h5_path: str, demo_name: str) -> str:
    """Stable key for a demo, matching datasets/domain_handler/libero_z.py."""
    p = Path(h5_path)
    demo = demo_name.split("/")[-1]
    return f"{p.parent.name}/{p.stem}::{demo}"


def load_meta(meta_path: str) -> dict:
    with open(meta_path) as f:
        return json.load(f)


def _demo_frames_raw(demo: h5py.Group, indices: np.ndarray) -> torch.Tensor:
    """Return raw uint8 frames [N, V=2, H, W, 3] (agentview, wrist), rotated
    180 degrees like LiberoHDF5Handler. Kept small (uint8, native resolution)
    so DataLoader workers move ~100KB/sample through IPC instead of ~7MB of
    preprocessed float tensors — resize/normalize happens on GPU
    (gpu_preprocess), which also avoids /dev/shm exhaustion in containers.
    """
    agent = demo["obs/agentview_rgb"]
    wrist = demo["obs/eye_in_hand_rgb"]
    out = []
    for t in indices:
        a = np.asarray(agent[t])[::-1, ::-1].copy()
        w = np.asarray(wrist[t])[::-1, ::-1].copy()
        out.append(np.stack([a, w], axis=0))
    return torch.from_numpy(np.stack(out, axis=0))


def gpu_preprocess(frames_u8: torch.Tensor, image_size: int,
                   device: torch.device | str) -> torch.Tensor:
    """uint8 [B, V, H, W, 3] -> normalized float [B, V, 3, S, S] on device.

    Same ops/order as the CPU pipeline in datasets/dataset_smolvlm.py:
    scale to [0,1], bicubic+antialias resize, ImageNet normalization.
    """
    x = frames_u8.to(device, non_blocking=True)
    B, V = x.shape[:2]
    x = x.permute(0, 1, 4, 2, 3).float().div_(255.0).flatten(0, 1)  # [B*V,3,H,W]
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size),
                          mode="bicubic", align_corners=False, antialias=True)
    mean = IMAGE_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGE_STD.to(device=x.device, dtype=x.dtype)
    x = (x - mean) / std
    return x.view(B, V, 3, image_size, image_size)


class FramePairDataset(IterableDataset):
    """
    Yields frame pairs (o_t, o_{t+stride}) for LAM training.

    Sample (raw uint8; run gpu_preprocess on the batch before the encoder):
      {
        "frames_t":  ByteTensor [V=2, H, W, 3],
        "frames_tk": ByteTensor [V=2, H, W, 3],
        "action":    FloatTensor [7]   (a_t, probe/diagnostics only),
      }
    """

    def __init__(
        self,
        meta_path: str,
        image_size: int = 384,
        stride: int = 1,
        training: bool = True,
        pairs_per_demo: int = 8,
    ):
        self.meta = load_meta(meta_path)
        self.image_size = image_size
        self.stride = stride
        self.training = training
        self.pairs_per_demo = pairs_per_demo
        self.items: List[dict] = list(self.meta["datalist"])

    def _iter_file(self, item: dict) -> Iterator[dict]:
        path = item["path"]
        allowed = set(item.get("demos", [])) or None
        try:
            f = h5py.File(path, "r")
        except OSError:
            return
        with f:
            if "data" not in f:
                return
            demo_keys = [k for k in f["data"].keys() if allowed is None or k in allowed]
            if self.training:
                random.shuffle(demo_keys)
            for dk in demo_keys:
                demo = f["data"][dk]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                T = min(len(actions), len(demo["obs/agentview_rgb"]))
                if T <= self.stride:
                    continue
                valid = np.arange(0, T - self.stride)
                if self.training:
                    ts = np.random.choice(valid, size=min(self.pairs_per_demo, len(valid)), replace=False)
                else:
                    ts = valid
                for t in ts:
                    pair = _demo_frames_raw(demo, np.array([t, t + self.stride]))
                    yield {
                        "frames_t": pair[0],
                        "frames_tk": pair[1],
                        "action": torch.from_numpy(actions[t]),
                    }

    def __iter__(self):
        items = list(self.items)
        info = get_worker_info()
        if info is not None:  # shard files across workers
            items = items[info.id::info.num_workers]
        if not self.training:
            for item in items:
                yield from self._iter_file(item)
            return
        while True:
            random.shuffle(items)
            for item in items:
                yield from self._iter_file(item)


def iter_demos(meta: dict) -> Iterable[dict]:
    """Iterate all demos in a meta, yielding handles for sequential labeling.

    Yields {"key", "h5_path", "demo_name", "num_frames", "actions", "frames_fn"}
    where frames_fn(indices) -> [N, 2, 3, S, S] must be called while the
    file is open (i.e. inside the generator loop body).
    """
    for item in meta["datalist"]:
        path = item["path"]
        allowed = set(item.get("demos", [])) or None
        with h5py.File(path, "r") as f:
            if "data" not in f:
                continue
            for dk in sorted(f["data"].keys()):
                if allowed is not None and dk not in allowed:
                    continue
                demo = f["data"][dk]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                T = min(len(actions), len(demo["obs/agentview_rgb"]))
                yield {
                    "key": demo_key_of(path, dk),
                    "h5_path": path,
                    "demo_name": dk,
                    "num_frames": T,
                    "actions": actions[:T],
                    "demo": demo,
                }
