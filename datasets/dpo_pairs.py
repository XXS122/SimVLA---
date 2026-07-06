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


def build_pair_index(
    records_paths,
    max_pairs_per_snapshot: int = 4,
    min_branch_rate: float | None = None,
    max_branch_rate: float | None = None,
) -> List[dict]:
    """
    Scan collector JSONL file(s) and enumerate (snapshot, winner, loser) pairs.

    min/max_branch_rate filter snapshots by their branch success rate:
    outcome labels only carry chunk-level credit at PIVOTAL states. At a
    7/8-success snapshot the single failure was almost certainly caused
    later in that branch, not by its first chunk — such pairs are label
    noise (this poisoned round 1). Keeping rates in e.g. [0.2, 0.8]
    restricts training to states where the first action plausibly decides
    the outcome.
    """
    if isinstance(records_paths, (str, Path)):
        records_paths = [records_paths]
    pairs = []
    for records_path in records_paths:
        root = Path(records_path).parent
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
                rate = sum(1 for b in branch_records if b["success"]) / len(branch_records)
                if min_branch_rate is not None and rate < min_branch_rate:
                    continue
                if max_branch_rate is not None and rate > max_branch_rate:
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
                            "records_root": str(root),
                            "action_w": w,
                            "action_l": l,
                            "arm": r.get("arm"),
                            "task_id": r.get("task_id"),
                            "branch_rate": rate,
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
        records_path,
        image_size: int = 384,
        num_views: int = 3,
        training: bool = True,
        max_pairs_per_snapshot: int = 4,
        min_branch_rate: float | None = None,
        max_branch_rate: float | None = None,
    ):
        self.pairs = build_pair_index(
            records_path, max_pairs_per_snapshot,
            min_branch_rate=min_branch_rate, max_branch_rate=max_branch_rate,
        )
        if not self.pairs:
            raise ValueError(
                f"No preference pairs found in {records_path}. The collector "
                "must be run with --save_pairs_dir, snapshots need both "
                "successful and failed branches, and the branch-rate filter "
                "must not exclude everything."
            )
        self.num_views = num_views

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

    @staticmethod
    def _resolve(pair: dict) -> str:
        p = Path(pair["snapshot_file"])
        if p.exists():
            return str(p)
        # Collector may have logged paths relative to its own cwd
        return str(Path(pair["records_root"]) / p.name)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        snap = np.load(self._resolve(pair), allow_pickle=True)

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
