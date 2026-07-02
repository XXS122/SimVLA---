"""Environment-variable driven configuration and workspace layout."""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "")
    return v if v else default


# ---- primary environment variables (see paths.env.example) ----
SMOLVLM_MODEL = _env("SIMVLA_SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
LIBERO_DATASETS = _env("LIBERO_DATASETS", "./datasets/metas")
CHECKPOINTS_ROOT = _env("SIMVLA_CHECKPOINTS", "./runs/latent_action_ws")
RESUME_CKPT = _env("SIMVLA_RESUME_CKPT", "")
CUDA_DEVICES = _env("CUDA_DEVICES", "")
NUM_GPUS = int(_env("NUM_GPUS", "1") or "1")
WANDB_PROJECT = _env("WANDB_PROJECT", "simvla")
Z_DIM = int(_env("SIMVLA_Z_DIM", "16") or "16")

DEFAULT_SUBSETS = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


class Workspace:
    """Directory layout for every artifact of the pipeline, rooted at
    $SIMVLA_CHECKPOINTS so the (possibly read-only) dataset dir is never
    written to."""

    def __init__(self, root: str | None = None):
        self.root = Path(root or CHECKPOINTS_ROOT)

    # ---- metadata ----
    @property
    def metas_dir(self) -> Path:
        return self.root / "metas"

    def meta_path(self, split: str = "p100") -> Path:
        if split in ("p100", "full", "100"):
            return self.metas_dir / "libero_train.json"
        return self.metas_dir / f"libero_train_{split}.json"

    @property
    def norm_stats_path(self) -> Path:
        return self.root / "norm_stats" / "libero_norm.json"

    # ---- latent action model ----
    @property
    def lam_dir(self) -> Path:
        return self.root / "lam"

    @property
    def lam_ckpt(self) -> Path:
        return self.lam_dir / "lam_final.pt"

    # ---- z labels ----
    @property
    def z_dir(self) -> Path:
        return self.root / "z_labels"

    def z_labels_path(self, stress: bool = False) -> Path:
        return self.z_dir / ("z_labels_stress.h5" if stress else "z_labels.h5")

    def z_meta_path(self, split: str = "p100") -> Path:
        if split in ("p100", "full", "100"):
            return self.z_dir / "libero_train_z.json"
        return self.z_dir / f"libero_train_z_{split}.json"

    # ---- probe ----
    @property
    def probe_dir(self) -> Path:
        return self.root / "probe"

    def probe_path(self, stress: bool = False) -> Path:
        return self.probe_dir / ("probe_stress.npz" if stress else "probe.npz")

    # ---- training runs ----
    def run_dir(self, name: str) -> Path:
        return self.root / "runs" / name

    def ensure(self) -> "Workspace":
        for d in (self.metas_dir, self.norm_stats_path.parent, self.lam_dir,
                  self.z_dir, self.probe_dir, self.root / "runs"):
            d.mkdir(parents=True, exist_ok=True)
        return self


def latest_checkpoint(run_dir: str | Path) -> str | None:
    """Return the highest-step ckpt-* directory inside a training run dir."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best_step, best = -1, None
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("ckpt-"):
            try:
                step = int(p.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            if step > best_step and (p / "model.safetensors").exists():
                best_step, best = step, p
    return str(best) if best else None
