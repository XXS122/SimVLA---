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


def preprocess_frame(img: np.ndarray, image_size: int) -> torch.Tensor:
    """uint8 [H, W, 3] (raw hdf5 orientation) -> float [3, S, S], normalized."""
    img = img[::-1, ::-1].copy()  # same 180-degree rotation as LiberoHDF5Handler
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    if t.shape[-1] != image_size or t.shape[-2] != image_size:
        t = F.interpolate(
            t.unsqueeze(0), size=(image_size, image_size),
            mode="bicubic", align_corners=False, antialias=True,
        ).squeeze(0)
    return (t - IMAGE_MEAN) / IMAGE_STD


def _demo_frames(demo: h5py.Group, indices: np.ndarray, image_size: int) -> torch.Tensor:
    """Return [N, V=2, 3, S, S] frames (agentview, wrist) at given indices."""
    agent = demo["obs/agentview_rgb"]
    wrist = demo["obs/eye_in_hand_rgb"]
    out = []
    for t in indices:
        a = preprocess_frame(np.asarray(agent[t]), image_size)
        w = preprocess_frame(np.asarray(wrist[t]), image_size)
        out.append(torch.stack([a, w], dim=0))
    return torch.stack(out, dim=0)


class FramePairDataset(IterableDataset):
    """
    Yields frame pairs (o_t, o_{t+stride}) for LAM training.

    Sample:
      {
        "frames_t":  FloatTensor [V=2, 3, S, S],
        "frames_tk": FloatTensor [V=2, 3, S, S],
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
                    pair = _demo_frames(demo, np.array([t, t + self.stride]), self.image_size)
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
