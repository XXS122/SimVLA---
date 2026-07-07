"""
Frame-pair data loading for PrefixStateUpdater distillation training.

Action-free, like latent_action/data.py (whose frame-reading utilities
this reuses directly): only images + the episode's language instruction
are needed, since the distillation target is the VLM's own fused-forward
output, not any action label.
"""

from __future__ import annotations

import random
from typing import Iterator, List

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from latent_action.data import _demo_frames_raw, load_meta


class InstructionFramePairDataset(IterableDataset):
    """
    Yields (frames_t, frames_tk, input_ids) for the same episode.

    Sample:
      {
        "frames_t":  ByteTensor [V=2, H, W, 3]  (raw uint8; gpu_preprocess before use)
        "frames_tk": ByteTensor [V=2, H, W, 3]
        "input_ids": LongTensor [seq_len]  (fixed per episode/task)
      }
    """

    def __init__(
        self,
        meta_path: str,
        tokenizer,
        stride: int = 1,
        seq_len: int = 24,
        training: bool = True,
        pairs_per_demo: int = 8,
    ):
        self.meta = load_meta(meta_path)
        self.tokenizer = tokenizer
        self.stride = stride
        self.seq_len = seq_len
        self.training = training
        self.pairs_per_demo = pairs_per_demo
        self.items: List[dict] = list(self.meta["datalist"])

    def _tokenize(self, task: str) -> torch.Tensor:
        ids = self.tokenizer(
            task, return_tensors="pt", padding="max_length",
            max_length=self.seq_len, truncation=True,
        )["input_ids"][0]
        return ids

    def _iter_file(self, item: dict) -> Iterator[dict]:
        path = item["path"]
        task = item.get("task", "")
        input_ids = self._tokenize(task)
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
                T = min(len(demo["obs/agentview_rgb"]), len(demo["obs/eye_in_hand_rgb"]))
                if T <= self.stride:
                    continue
                valid = np.arange(0, T - self.stride)
                if self.training:
                    ts = np.random.choice(valid, size=min(self.pairs_per_demo, len(valid)), replace=False)
                else:
                    ts = valid
                for t in ts:
                    pair = _demo_frames_raw(demo, np.array([t, t + self.stride]))
                    yield {"frames_t": pair[0], "frames_tk": pair[1], "input_ids": input_ids}

    def __iter__(self):
        items = list(self.items)
        info = get_worker_info()
        if info is not None:
            items = items[info.id::info.num_workers]
        if not self.training:
            for item in items:
                yield from self._iter_file(item)
            return
        while True:
            random.shuffle(items)
            for item in items:
                yield from self._iter_file(item)


class InstructionSequenceDataset(IterableDataset):
    """
    Yields CONSECUTIVE frame windows (not random pairs) from one episode,
    for recursive-rollout / scheduled-sampling training that fixes the
    exposure-bias drift observed with teacher-forced training.

    Sample:
      {
        "frames":    ByteTensor [K+1, V=2, H, W, 3]  (raw uint8; gpu_preprocess before use)
        "input_ids": LongTensor [seq_len]
      }
    where K = rollout_len and frames are at t, t+stride, ..., t+K*stride.
    """

    def __init__(
        self,
        meta_path: str,
        tokenizer,
        rollout_len: int = 8,
        stride: int = 1,
        seq_len: int = 24,
        training: bool = True,
        windows_per_demo: int = 4,
    ):
        self.meta = load_meta(meta_path)
        self.tokenizer = tokenizer
        self.rollout_len = rollout_len
        self.stride = stride
        self.seq_len = seq_len
        self.training = training
        self.windows_per_demo = windows_per_demo
        self.items: List[dict] = list(self.meta["datalist"])

    def _tokenize(self, task: str) -> torch.Tensor:
        return self.tokenizer(
            task, return_tensors="pt", padding="max_length",
            max_length=self.seq_len, truncation=True,
        )["input_ids"][0]

    def _iter_file(self, item: dict) -> Iterator[dict]:
        path = item["path"]
        input_ids = self._tokenize(item.get("task", ""))
        allowed = set(item.get("demos", [])) or None
        span = self.rollout_len * self.stride
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
                T = min(len(demo["obs/agentview_rgb"]), len(demo["obs/eye_in_hand_rgb"]))
                if T <= span:
                    continue
                starts = np.arange(0, T - span)
                if self.training:
                    starts = np.random.choice(
                        starts, size=min(self.windows_per_demo, len(starts)), replace=False)
                for t0 in starts:
                    idx = np.arange(t0, t0 + span + 1, self.stride)
                    yield {"frames": _demo_frames_raw(demo, idx), "input_ids": input_ids}

    def __iter__(self):
        items = list(self.items)
        info = get_worker_info()
        if info is not None:
            items = items[info.id::info.num_workers]
        if not self.training:
            for item in items:
                yield from self._iter_file(item)
            return
        while True:
            random.shuffle(items)
            for item in items:
                yield from self._iter_file(item)
