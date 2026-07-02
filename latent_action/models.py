"""
Latent Action Model (LAM) for identifiable continuous latent-action pretraining.

Components
----------
FrozenVisionBackbone : SmolVLM's SigLIP vision tower + connector, frozen.
LatentActionModel    : inverse-dynamics encoder (frame pair -> z) and
                       forward decoder (frame_t tokens + z -> frame_{t+k} tokens).
VectorQuantizerEMA   : optional discrete bottleneck for the VQ ablation
                       (same-scale stand-in for LAPA/UniVLA-style tokens).

Identifiability-oriented regularizers (used by train_lam.py):
  variance_covariance_reg : VICReg-style whitening — keeps per-dim variance
                            above 1 and decorrelates dimensions, pinning the
                            latent space up to rotation.
  shared_ego_augment      : applies the SAME random affine (translation /
                            scale / rotation) to both frames of a pair;
                            invariance of z to this shared nuisance separates
                            camera ego-motion from true actions.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_smolvlm import TransformerBlock
from transformers import AutoModelForImageTextToText


# ============================================================
# Frozen vision backbone
# ============================================================
class FrozenVisionBackbone(nn.Module):
    """SigLIP vision encoder + connector from SmolVLM, frozen.

    forward: [B, V, 3, H, W] -> tokens [B, V*P, D] where D is the LM hidden
    size (same feature space the SimVLA flow head is conditioned on).
    """

    def __init__(self, smolvlm_model_path: str):
        super().__init__()
        vlm = AutoModelForImageTextToText.from_pretrained(
            smolvlm_model_path, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.vision_model = vlm.model.vision_model
        if hasattr(vlm.model, "connector"):
            self.connector = vlm.model.connector
        elif hasattr(vlm.model, "multi_modal_projector"):
            self.connector = vlm.model.multi_modal_projector
        else:
            raise AttributeError("SmolVLM model has no connector/multi_modal_projector")
        self.out_dim = vlm.config.text_config.hidden_size
        del vlm
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # stay in eval mode permanently
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, V = images.shape[:2]
        flat = images.flatten(0, 1)
        feats = self.vision_model(pixel_values=flat).last_hidden_state
        feats = self.connector(feats)  # [B*V, P, D]
        P, D = feats.shape[1], feats.shape[2]
        return feats.view(B, V * P, D)


# ============================================================
# Optional VQ bottleneck (discrete ablation)
# ============================================================
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes, self.dim, self.decay, self.eps = num_codes, dim, decay, eps
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embed.t()
            + self.embed.pow(2).sum(1)[None, :]
        )
        idx = dist.argmin(dim=1)
        z_q = self.embed[idx]
        if self.training:
            onehot = F.one_hot(idx, self.num_codes).type(z.dtype)
            self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
            self.embed_avg.mul_(self.decay).add_(onehot.t() @ z, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
            self.embed.copy_(self.embed_avg / cluster.unsqueeze(1))
        commit = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()  # straight-through
        return z_q, commit, idx


# ============================================================
# Latent Action Model
# ============================================================
class LatentActionModel(nn.Module):
    def __init__(
        self,
        vlm_dim: int = 960,
        dim: int = 512,
        z_dim: int = 16,
        enc_depth: int = 4,
        dec_depth: int = 4,
        num_heads: int = 8,
        max_tokens: int = 512,
        use_vq: bool = False,
        vq_codes: int = 256,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.use_vq = use_vq

        self.proj_in = nn.Linear(vlm_dim, dim)
        self.time_emb = nn.Parameter(torch.zeros(2, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.time_emb, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.query, std=0.02)

        self.encoder = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(enc_depth)]
        )
        self.to_z = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, z_dim))

        self.vq = VectorQuantizerEMA(vq_codes, z_dim) if use_vq else None

        self.z_in = nn.Linear(z_dim, dim)
        self.decoder = nn.ModuleList(
            [TransformerBlock(dim, num_heads) for _ in range(dec_depth)]
        )
        self.dec_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, vlm_dim))

    # ---- inverse dynamics: (o_t, o_{t+k}) -> z ----
    def encode(self, feats_t: torch.Tensor, feats_tk: torch.Tensor) -> torch.Tensor:
        B, P, _ = feats_t.shape
        x_t = self.proj_in(feats_t) + self.time_emb[0] + self.pos_emb[:, :P]
        x_tk = self.proj_in(feats_tk) + self.time_emb[1] + self.pos_emb[:, :P]
        q = self.query.expand(B, 1, -1)
        x = torch.cat([x_t, x_tk, q], dim=1)
        for blk in self.encoder:
            x = blk(x)
        return self.to_z(x[:, -1])

    # ---- forward model: (o_t, z) -> o_{t+k} tokens ----
    def decode(self, feats_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, P, _ = feats_t.shape
        x = self.proj_in(feats_t) + self.pos_emb[:, :P]
        zt = self.z_in(z).unsqueeze(1)
        x = torch.cat([zt, x], dim=1)
        for blk in self.decoder:
            x = blk(x)
        return self.dec_out(x[:, 1:])

    def forward(
        self, feats_t: torch.Tensor, feats_tk: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z = self.encode(feats_t, feats_tk)
        vq_loss = feats_t.new_zeros(())
        if self.vq is not None:
            z, vq_loss, _ = self.vq(z)
        pred = self.decode(feats_t, z)
        recon = F.mse_loss(pred, feats_tk)
        return {"z": z, "pred": pred, "recon_loss": recon, "vq_loss": vq_loss}

    def config_dict(self) -> dict:
        return {
            "vlm_dim": self.proj_in.in_features,
            "dim": self.proj_in.out_features,
            "z_dim": self.z_dim,
            "enc_depth": len(self.encoder),
            "dec_depth": len(self.decoder),
            "num_heads": self.encoder[0].attn.num_heads,
            "max_tokens": self.pos_emb.shape[1],
            "use_vq": self.use_vq,
            "vq_codes": self.vq.num_codes if self.vq is not None else 256,
        }


# ============================================================
# Identifiability regularizers
# ============================================================
def variance_covariance_reg(z: torch.Tensor, gamma: float = 1.0) -> Dict[str, torch.Tensor]:
    """VICReg-style variance hinge + covariance decorrelation on a z batch."""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(gamma - std).mean()
    B, D = z.shape
    cov = (z.t() @ z) / max(1, B - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / D
    return {"var_loss": var_loss, "cov_loss": cov_loss}


def shared_ego_augment(
    frames_t: torch.Tensor,
    frames_tk: torch.Tensor,
    max_translate: float = 0.08,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    max_rot_deg: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the SAME random affine warp to both frames of each pair
    (simulated camera ego-motion). Different samples get different warps;
    all views of a sample share one warp.

    frames: [B, V, 3, H, W]
    """
    B, V = frames_t.shape[:2]
    device, dtype = frames_t.device, frames_t.dtype

    angle = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * math.radians(max_rot_deg)
    scale = scale_range[0] + torch.rand(B, device=device, dtype=dtype) * (scale_range[1] - scale_range[0])
    tx = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_translate
    ty = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_translate

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty

    def warp(frames: torch.Tensor) -> torch.Tensor:
        Bv = frames.flatten(0, 1)  # [B*V, 3, H, W]
        th = theta.repeat_interleave(V, dim=0)
        grid = F.affine_grid(th, list(Bv.shape), align_corners=False)
        out = F.grid_sample(Bv, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        return out.view_as(frames)

    return warp(frames_t), warp(frames_tk)


# ============================================================
# Checkpoint helpers
# ============================================================
def save_lam(model: LatentActionModel, path, extra: Optional[dict] = None):
    payload = {"config": model.config_dict(), "state_dict": model.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_lam(path, map_location="cpu") -> LatentActionModel:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = LatentActionModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model
