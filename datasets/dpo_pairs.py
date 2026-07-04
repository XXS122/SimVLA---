"""
Matched-state preference pair dataset for Flow-DPO.

Consumes the output of the branch collector
(evaluation/libero/branch_counterfactual.py --save_pairs_dir ...):
  - branches.jsonl : per-snapshot records with branch outcomes and the first
    action chunk each branch sampled at the snapshot state
  - snap_*.npz     : the snapshot observation (agentview image, wrist image,
    8-dim state, prompt)

A preference pair is (winner chunk, loser chunk) sampled at the SAME
simulator state, where the winner comes from a branch continuation that
succeeded and the loser from one that failed. Snapshots where all branches
agree (all succeed / all fail) yield no pairs.

Sample format matches the SmolVLM training pipeline:
  {
    'language_instruction': str,
    'image_input': FloatTensor[V, C, H, W],
    'image_mask': BoolTensor[V],
    'proprio': FloatTensor[8],
    'action_w': FloatTensor[T, 7],   # preferred chunk
    'action_l': FloatTensor[T, 7],   # dispreferred chunk
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_pair_index(records_path: str, max_pairs_per_snapshot: int = 4) -> List[dict]:
    """Scan the collector JSONL and enumerate (snapshot, winner, loser) pairs."""
    pairs = []
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            branch_records = r.get("branch_records")
            snap = r.get("snapshot_file")
            if not branch_records or not snap:
                continue
            winners = [b["first_chunk"] for b in branch_records
                       if b["success"] and b["first_chunk"]]
            losers = [b["first_chunk"] for b in branch_records
                      if not b["success"] and b["first_chunk"]]
            if not winners or not losers:
                continue
            count = 0
            for w in winners:
                for l in losers:
                    pairs.append({
                        "snapshot_file": snap,
                        "action_w": w,
                        "action_l": l,
                        "arm": r.get("arm"),
                        "task_id": r.get("task_id"),
                    })
                    count += 1
                    if count >= max_pairs_per_snapshot:
                        break
                if count >= max_pairs_per_snapshot:
                    break
    return pairs


class DPOPairDataset(Dataset):
    """Map-style dataset over matched-state preference pairs."""

    def __init__(
        self,
        records_path: str,
        image_size: int = 384,
        num_views: int = 3,
        training: bool = True,
        max_pairs_per_snapshot: int = 4,
    ):
        self.pairs = build_pair_index(records_path, max_pairs_per_snapshot)
        if not self.pairs:
            raise ValueError(
                f"No preference pairs found in {records_path}. The collector "
                "must be run with --save_pairs_dir, and snapshots need both "
                "successful and failed branches."
            )
        self.num_views = num_views
        self.records_root = Path(records_path).parent

        transform_list = [
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC,
                              antialias=True),
        ]
        if training:
            transform_list.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.0)
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD, inplace=True),
        ])
        self.image_aug = transforms.Compose(transform_list)

        n_snaps = len({p["snapshot_file"] for p in self.pairs})
        print(f"[DPOPairDataset] {len(self.pairs)} pairs from {n_snaps} snapshots")

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve(self, path: str) -> str:
        p = Path(path)
        if p.exists():
            return str(p)
        # Collector may have logged paths relative to its own cwd
        return str(self.records_root / p.name)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        snap = np.load(self._resolve(pair["snapshot_file"]), allow_pickle=True)

        imgs = [
            self.image_aug(Image.fromarray(snap["image"])),
            self.image_aug(Image.fromarray(snap["wrist_image"])),
        ]
        while len(imgs) < self.num_views:
            imgs.append(torch.zeros_like(imgs[0]))
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True

        return {
            "language_instruction": str(snap["prompt"]),
            "image_input": torch.stack(imgs, dim=0),
            "image_mask": image_mask,
            "proprio": torch.tensor(np.asarray(snap["state"]), dtype=torch.float32),
            "action_w": torch.tensor(pair["action_w"], dtype=torch.float32),
            "action_l": torch.tensor(pair["action_l"], dtype=torch.float32),
        }


__all__ = ["DPOPairDataset", "build_pair_index"]
