"""
LIBERO latent-action (z) data handler for flow-expert pretraining.

Identical to LiberoHDF5Handler (same images, proprio, language) except that
the chunk targets are the per-step latent actions z written by
latent_action/label_z.py instead of the raw 7-dim actions.

The meta JSON (written by label_z.py with --z_meta_out) must contain:
    "dataset_name":  "libero_z"
    "z_labels_path": path to the z-labels .h5
    "z_dim":         latent dimension
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .libero_hdf5 import LiberoHDF5Handler


def demo_key_of(h5_path: str, demo_name: str) -> str:
    """Must match latent_action/data.py::demo_key_of."""
    p = Path(h5_path)
    demo = demo_name.split("/")[-1]
    return f"{p.parent.name}/{p.stem}::{demo}"


class LiberoZHandler(LiberoHDF5Handler):
    dataset_name = "libero_z"

    def __init__(self, meta: dict, num_views: int = 3) -> None:
        super().__init__(meta, num_views)
        self.z_labels_path = meta["z_labels_path"]
        self.z_dim = int(meta.get("z_dim", 0))
        self._zf: h5py.File | None = None  # opened lazily per worker process

    def _z_file(self) -> h5py.File:
        if self._zf is None:
            self._zf = h5py.File(self.z_labels_path, "r")
        return self._zf

    def _load_targets(self, demo: h5py.Group, actions: np.ndarray) -> np.ndarray:
        key = demo_key_of(demo.file.filename, demo.name)
        zf = self._z_file()
        if key not in zf:
            raise KeyError(
                f"z labels missing for '{key}' in {self.z_labels_path}; "
                f"re-run `python -m latent_action.run label`"
            )
        return np.asarray(zf[key], dtype=np.float32)
