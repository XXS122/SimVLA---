from __future__ import annotations
from typing import Iterable, Tuple, Dict, Type, Optional
from pathlib import Path
import json
import os
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Normalization Stats
# =============================================================================
class NormStats:
    """Normalization statistics for action normalization."""
    
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        q01: Optional[np.ndarray] = None,
        q99: Optional[np.ndarray] = None,
    ):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.q01 = torch.as_tensor(q01, dtype=torch.float32) if q01 is not None else None
        self.q99 = torch.as_tensor(q99, dtype=torch.float32) if q99 is not None else None
        
    def to(self, device):
        """Move to specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.q01 is not None:
            self.q01 = self.q01.to(device)
        if self.q99 is not None:
            self.q99 = self.q99.to(device)
        return self


def load_norm_stats(path: str) -> Dict[str, NormStats]:
    """
    Load normalization statistics from JSON file.
    
    Supports two formats:
    1. Legacy format (action stats only):
       {"action": {"mean": [...], "std": [...], ...}}
       
    2. Extended format (separate state and actions stats):
       {"norm_stats": {"state": {...}, "actions": {...}}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found: {path}")
        
    with open(path) as f:
        data = json.load(f)
    
    result = {}
    
    # Check if it's extended format (has norm_stats key)
    if "norm_stats" in data:
        data = data["norm_stats"]
    
    for key, stats in data.items():
        if key == "metadata":
            continue
        result[key] = NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
            q01=np.array(stats.get("q01"), dtype=np.float32) if stats.get("q01") else None,
            q99=np.array(stats.get("q99"), dtype=np.float32) if stats.get("q99") else None,
        )
    return result


# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections.
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


# =============================================================================
# LIBERO Action Space
# =============================================================================
@register_action("libero_joint")
class LiberoJointActionSpace(BaseActionSpace):
    """
    LIBERO joint/delta action space.
    
    Data layout:
      - state (proprio): 8-dim [ee_pos(3), ee_ori(3), gripper_states(2)]
      - actions: 7-dim [delta_xyz(3), delta_euler(3), gripper_cmd(1)]
      
    Actions range: [-1, 1] (normalized delta actions)
    
    - Uses MSE loss
    - Optional Z-score or Quantile normalization
    """

    dim_action = 7
    dim_proprio = 8
    gripper_idx = (6,)  # Last dimension is gripper

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
    ):
        super().__init__()
        self.use_quantile_norm = use_quantile_norm
        self.state_norm_stats: Optional[NormStats] = None
        self.action_norm_stats: Optional[NormStats] = None
        
        if norm_stats_path:
            self.load_norm_stats(norm_stats_path)
            
    def load_norm_stats(self, path: str):
        """Load normalization statistics."""
        stats_dict = load_norm_stats(path)
        
        if "state" in stats_dict:
            self.state_norm_stats = stats_dict["state"]
            print(f"[LiberoJointActionSpace] Loaded state norm stats, dim={len(self.state_norm_stats.mean)}")
            
        if "actions" in stats_dict:
            self.action_norm_stats = stats_dict["actions"]
            print(f"[LiberoJointActionSpace] Loaded actions norm stats, dim={len(self.action_norm_stats.mean)}")
            
    def to(self, device):
        """Move to specified device."""
        if self.state_norm_stats is not None:
            self.state_norm_stats.to(device)
        if self.action_norm_stats is not None:
            self.action_norm_stats.to(device)
        return super().to(device)
    
    def _normalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Normalize using specified statistics."""
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        D = x.shape[-1]
        
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return (x - mean) / (std + 1e-6)
    
    def _unnormalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Unnormalize using specified statistics."""
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        D = x.shape[-1]
            
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return x * (std + 1e-6) + mean
    
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state/proprio."""
        if self.state_norm_stats is not None:
            return self._normalize_with_stats(x, self.state_norm_stats)
        return x
    
    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize action."""
        if self.action_norm_stats is not None:
            return self._normalize_with_stats(x, self.action_norm_stats)
        return x
    
    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalize action."""
        if self.action_norm_stats is not None:
            return self._unnormalize_with_stats(x, self.action_norm_stats)
        return x

    def compute_loss(self, pred, target):
        """Full-dimension MSE loss."""
        loss = torch.square(pred - target)
        return {"velocity_loss": torch.mean(loss)}

    def preprocess(self, proprio, action, mode="train"):
        """Normalize proprio and action separately."""
        proprio_norm = self.normalize_state(proprio)
        action_norm = self.normalize_action(action)
        return proprio_norm, action_norm

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Unnormalize action."""
        return self.unnormalize_action(action)


# =============================================================================
# Latent-action spaces (latent_action pipeline)
# =============================================================================
def _default_z_dim() -> int:
    return int(os.environ.get("SIMVLA_Z_DIM", "16") or "16")


@register_action("latent_z")
class LatentZActionSpace(LiberoJointActionSpace):
    """
    Flow-matching pretraining space: the "action" chunk is a latent-action
    chunk z produced by the LAM (latent_action/label_z.py).

    z is whitened by the LAM's variance/covariance regularizer, so action
    normalization is the identity. Proprio is still normalized with the
    LIBERO state stats so it matches the downstream fine-tuning stage.
    """

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
        z_dim: Optional[int] = None,
    ):
        super().__init__(norm_stats_path=norm_stats_path,
                         use_quantile_norm=use_quantile_norm)
        self.dim_action = int(z_dim or _default_z_dim())
        self.dim_proprio = 8
        self.gripper_idx = ()

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return action


@register_action("libero_z_adapter")
class LiberoZAdapterActionSpace(LiberoJointActionSpace):
    """
    Downstream fine-tuning space: the flow expert keeps operating in the
    pretrained z-space; real 7-dim actions are mapped in/out through a
    frozen affine adapter initialized from the probe fit
    (latent_action/probe.py), i.e. the "identifiable up to affine" claim
    made executable:

        encode  : z = A @ a_norm + b          (training targets)
        decode  : a_norm = pinv(A) @ (z - b)  (inference postprocess)

    A, b are nn.Parameters (saved/restored with the checkpoint) but frozen
    by default; --train_adapter unfreezes them for the ablation.
    """

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
        z_dim: Optional[int] = None,
        probe_path: Optional[str] = None,
        trainable: bool = False,
    ):
        super().__init__(norm_stats_path=norm_stats_path,
                         use_quantile_norm=use_quantile_norm)
        zd = int(z_dim or _default_z_dim())
        self.z_dim = zd
        self.dim_action = zd          # the flow head generates in z-space
        self.dim_proprio = 8
        self.raw_action_dim = 7

        A = torch.empty(zd, self.raw_action_dim)
        nn.init.orthogonal_(A)
        self.A = nn.Parameter(A, requires_grad=trainable)
        self.b = nn.Parameter(torch.zeros(zd), requires_grad=trainable)

        if probe_path:
            self.load_probe(probe_path)

    def load_probe(self, path: str):
        probe = np.load(path)
        A = torch.as_tensor(probe["A"], dtype=torch.float32)   # [z_dim, 7]
        b = torch.as_tensor(probe["b"], dtype=torch.float32)   # [z_dim]
        if A.shape != self.A.shape:
            raise ValueError(
                f"probe A shape {tuple(A.shape)} != adapter {tuple(self.A.shape)}; "
                f"check SIMVLA_Z_DIM consistency across stages"
            )
        with torch.no_grad():
            self.A.copy_(A)
            self.b.copy_(b)
        print(f"[LiberoZAdapter] loaded affine adapter from {path} "
              f"(probe mean R^2 z->a: {float(np.mean(probe['r2_za'])):.4f})")

    # a raw [.., 7] -> z [.., z_dim]
    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        a_norm = super().normalize_action(x)
        A = self.A.to(device=a_norm.device, dtype=a_norm.dtype)
        b = self.b.to(device=a_norm.device, dtype=a_norm.dtype)
        return a_norm @ A.t() + b

    # z [.., z_dim] -> a raw [.., 7]
    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        A = self.A.to(device=action.device, dtype=torch.float32)
        b = self.b.to(device=action.device, dtype=torch.float32)
        z = action.to(torch.float32) - b
        a_norm = z @ torch.linalg.pinv(A).t()
        return super().unnormalize_action(a_norm).to(action.dtype)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "LiberoJointActionSpace",
    "LatentZActionSpace",
    "LiberoZAdapterActionSpace",
    "ACTION_REGISTRY",
    "NormStats",
    "load_norm_stats",
]
