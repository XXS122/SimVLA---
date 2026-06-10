"""
SmolVLM Dataset

Dataset classes specifically designed for SmolVLM-VLA training.
Key differences from the original dataset:
  - Uses 512x512 image resolution (SmolVLM-500M requirement)
  - Handles smaller images by proper upscaling
  - ImageNet normalization compatible with SmolVLM
"""

from __future__ import annotations
from typing import Dict, Iterable, List
import io
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS
from .domain_handler.registry import get_handler_cls


class SmolVLMDataReader(IterableDataset):
    """
    Infinite data reader for SmolVLM-VLA training.
    
    Uses 512x512 image resolution required by SmolVLM-500M.
    Properly handles images that are smaller than 512x512 by upscaling.
    
    Output sample:
      {
        'language_instruction': str,
        'image_input': FloatTensor[V, C, 512, 512],  # 512x512 for SmolVLM-500M
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action],
      }
    """
    
    # SmolVLM specific constants
    IMAGE_SIZE = 384  # Default 384, can be adjusted (384/512)
    
    # ImageNet normalization (same as SmolVLM uses)
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    
    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "galaxea_joint",
        lang_aug: str = None,
        image_size: int = 384,  # Default 384, can be 384 or 512
        tds_weights_csv: str = None,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.image_size = image_size
        self.metas: Dict[str, dict] = {}

        print(f"[SmolVLM Dataset] Image size: {self.image_size}x{self.image_size}")
        print(f"[SmolVLM Dataset] Action mode: {action_mode}")

        # Transition-Density Sampling (off when csv not given -> exact
        # original uniform behavior)
        self.tds_task_pk = None
        if tds_weights_csv:
            from .tds_sampling import load_task_weights
            self.tds_task_pk = load_task_weights(tds_weights_csv)
            print(f"[SmolVLM Dataset] TDS enabled: {len(self.tds_task_pk)} "
                  f"task weights from {tds_weights_csv}")
        
        # Load metadata
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(
                metas_path, suffix=".json", recursive=True, list_dir=False
            )
            root = metas_path
        elif metas_path.endswith('.json'):
            try:
                with open(metas_path, 'r') as f:
                    content = json.load(f)
                if isinstance(content, list):
                    meta_files = content
                    root = ""
                else:
                    meta_files, root = [metas_path], ""
            except Exception:
                meta_files, root = [metas_path], ""
        else:
            meta_files, root = [metas_path], ""
            
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f:
                meta = json.load(f)
            print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
            self.metas[meta["dataset_name"]] = meta

        # Build image augmentation pipeline for 384x384
        self.image_aug = self._build_image_transforms(training)

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        """
        Build image transformation pipeline for SmolVLM.
        
        Handles:
          - Resizing to 512x512 (upscale if smaller, downscale if larger)
          - Color jitter for training augmentation
          - ImageNet normalization
        """
        transform_list = [
            # Resize to 384x384
            # Use BICUBIC for better quality when upscaling small images
            transforms.Resize(
                (self.image_size, self.image_size), 
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
        ]
        
        # Training augmentation
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.0
                )
            )
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        
        return transforms.Compose(transform_list)

    def _finalize_sample(self, sample: dict, meta: dict) -> dict:
        """Shared post-processing: slice abs trajectory into action chunk."""
        idx_for_delta = meta.get("idx_for_delta", [])
        has_proprio = "proprio" in sample
        slice_result = action_slice(sample["abs_trajectory"], idx_for_delta)

        if has_proprio:
            sample["action"] = slice_result["action"]
        else:
            sample.update(slice_result)
        del sample["abs_trajectory"]
        return sample

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        """Iterate over one dataset."""
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training:
            random.shuffle(traj_indices)

        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)

        for traj_idx in traj_indices:
            try:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    action_mode=self.action_mode
                ):
                    yield self._finalize_sample(sample, meta)
            except Exception as e:
                continue

        if self.training:
            yield from self._iter_one_dataset(dataset_name)

    def _iter_one_task(self, dataset_name: str, traj_idx: int) -> Iterable[dict]:
        """Infinite sample stream from a single task file (TDS mode)."""
        meta = self.metas[dataset_name]
        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)
        failures = 0
        while True:
            yielded = False
            try:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    action_mode=self.action_mode
                ):
                    yielded = True
                    yield self._finalize_sample(sample, meta)
            except Exception:
                pass
            if yielded:
                failures = 0
            else:
                failures += 1
                if failures >= 3:
                    print(f"[TDS] task file idx={traj_idx} of {dataset_name} "
                          f"failed {failures} passes in a row, dropping it "
                          f"from the sampling pool")
                    return

    def _iter_tds(self) -> Iterable[dict]:
        """Two-level Transition-Density Sampling over all task files.

        Each yield draws a task file with probability DATA_WEIGHTS-scaled
        p_k, then takes the next sample from that task's persistent stream.
        Note: this keeps one suspended episode generator per task file
        (roughly the decoded arrays of one demo each), so worker RAM grows
        by ~10-20 MB per task file.
        """
        from .tds_sampling import (SamplingMonitor, file_stem,
                                   resolve_stem_weights)

        entries = []  # (dataset_name, traj_idx, stem, dataset_weight)
        for name, meta in self.metas.items():
            dw = DATA_WEIGHTS.get(name, 1.0)
            for i, item in enumerate(meta["datalist"]):
                path = item["path"] if isinstance(item, dict) else item
                entries.append((name, i, file_stem(path), dw))

        pks = resolve_stem_weights([e[2] for e in entries], self.tds_task_pk)
        ws = [dw * pk for (_, _, _, dw), pk in zip(entries, pks)]
        s = sum(ws)
        ws = [w / s for w in ws]
        gens = [iter(self._iter_one_task(name, i)) for name, i, _, _ in entries]

        # Verify the empirical sampling distribution once (worker 0 only)
        worker_info = torch.utils.data.get_worker_info()
        monitor = None
        if worker_info is None or worker_info.id == 0:
            monitor = SamplingMonitor(
                {e[2]: w for e, w in zip(entries, ws)})

        while gens:
            i = random.choices(range(len(gens)), weights=ws, k=1)[0]
            try:
                sample = next(gens[i])
            except StopIteration:
                # Dead task file: remove and renormalize
                del gens[i], ws[i], entries[i]
                if not gens:
                    return
                s = sum(ws)
                ws = [w / s for w in ws]
                continue
            if monitor is not None:
                monitor.record(entries[i][2])
            yield sample

    def __iter__(self):
        """Main iteration."""
        names = list(self.metas.keys())
        if not self.training:
            for n in names:
                yield from self._iter_one_dataset(n)
        elif self.tds_task_pk is not None:
            yield from self._iter_tds()
        else:
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws)
            ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])


class SmolVLMDataReaderWithPadding(SmolVLMDataReader):
    """
    SmolVLM data reader with smart padding for small images.
    
    Instead of just resizing, this version:
    1. If image is smaller than 512x512, pad to maintain aspect ratio
    2. Then resize to 512x512
    
    This can be better for images that are much smaller than 512x512,
    as it avoids extreme upscaling artifacts.
    """
    
    # Padding modes
    PADDING_MODE = "reflect"  # or "constant", "edge"
    
    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        """Build transforms with smart padding."""
        
        class SmartResize:
            """Smart resize that handles small images better."""
            
            def __init__(self, target_size: int, padding_mode: str = "reflect"):
                self.target_size = target_size
                self.padding_mode = padding_mode
                
            def __call__(self, img):
                """
                Resize image to target size with optional padding.
                
                For images much smaller than target:
                - First pad to maintain aspect ratio
                - Then resize
                """
                from PIL import Image
                import numpy as np
                
                w, h = img.size
                
                # If both dimensions are smaller than target/2, use padding approach
                if w < self.target_size // 2 and h < self.target_size // 2:
                    # Pad to target size first
                    result = Image.new('RGB', (self.target_size, self.target_size))
                    
                    # Center the original image
                    paste_x = (self.target_size - w) // 2
                    paste_y = (self.target_size - h) // 2
                    result.paste(img, (paste_x, paste_y))
                    
                    # Apply reflection padding to fill empty space
                    result_np = np.array(result)
                    
                    # Simple reflection: copy border pixels
                    if paste_x > 0:
                        # Reflect left
                        result_np[:, :paste_x] = np.flip(
                            result_np[:, paste_x:paste_x*2], axis=1
                        )[:, :paste_x]
                        # Reflect right
                        result_np[:, paste_x+w:] = np.flip(
                            result_np[:, paste_x+w-paste_x:paste_x+w], axis=1
                        )[:, :self.target_size-paste_x-w]
                    
                    return Image.fromarray(result_np)
                else:
                    # Standard resize for reasonably sized images
                    return img.resize(
                        (self.target_size, self.target_size),
                        Image.BICUBIC
                    )
        
        transform_list = [
            SmartResize(self.image_size, self.PADDING_MODE),
        ]
        
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        
        return transforms.Compose(transform_list)


def create_smolvlm_dataloader(
    batch_size: int,
    metas_path: str,
    num_actions: int,
    training: bool,
    action_mode: str,
    num_workers: int = 4,
    image_size: int = 384,
    use_smart_padding: bool = False,
    tds_weights_csv: str = None,
):
    """
    Create dataloader for SmolVLM-VLA training.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training.
    metas_path : str
        Path to metadata files.
    num_actions : int
        Number of future actions to predict.
    training : bool
        Whether this is for training.
    action_mode : str
        Action mode (e.g., "galaxea_joint", "libero_joint").
    num_workers : int
        Number of data loading workers.
    image_size : int
        Image size (default 384, can be 384 or 512).
    use_smart_padding : bool
        Whether to use smart padding for small images.
    tds_weights_csv : str
        Optional task_difficulty.csv from transition_density_stats.py;
        enables Transition-Density Sampling over task files.

    Returns
    -------
    DataLoader
        PyTorch DataLoader for SmolVLM-VLA training.
    """
    from torch.utils.data import DataLoader
    
    def worker_init_fn(worker_id: int):
        """Worker initialization."""
        base_seed = torch.initial_seed() % (2**32)
        import random
        np.random.seed(base_seed)
        random.seed(base_seed)
        torch.manual_seed(base_seed)
        
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            tf.get_logger().setLevel("ERROR")
        except Exception:
            pass
    
    # Choose dataset class
    if use_smart_padding:
        DatasetClass = SmolVLMDataReaderWithPadding
    else:
        DatasetClass = SmolVLMDataReader
    
    dataset = DatasetClass(
        metas_path=metas_path,
        num_actions=num_actions,
        training=training,
        action_mode=action_mode,
        image_size=image_size,
        tds_weights_csv=tds_weights_csv,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


__all__ = [
    "SmolVLMDataReader",
    "SmolVLMDataReaderWithPadding", 
    "create_smolvlm_dataloader",
]
