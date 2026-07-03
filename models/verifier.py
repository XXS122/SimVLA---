"""
Action Energy Verifier

A lightweight energy model E(o, a) over action chunks, sharing the frozen
SmolVLM features of the policy. Used for best-of-K test-time scaling:
sample K one-step candidates from the MeanFlow generator, pick argmin E.

Design:
  - VLM features are compressed with M learnable attention-pooling queries
    (so scoring K candidates only replicates M pooled tokens, not the full
    VLM sequence).
  - Sequence = [ENERGY token, proprio token, action tokens (T), pooled VLM
    tokens (M)] -> small pre-LN transformer -> MLP head on the ENERGY token.
  - Operates in the *normalized* action space (same space the flow head is
    trained in).

Trained with InfoNCE: the dataset action is the positive; negatives come
from the generator's own proposal distribution (plus perturbed/shuffled
ground truth). See train_verifier.py.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_smolvlm import TransformerBlock, basic_init


@dataclass
class ActionEnergyVerifierConfig:
    vlm_hidden_size: int = 576
    dim_action: int = 7
    dim_proprio: int = 8
    hidden_size: int = 384
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_pool_tokens: int = 8
    max_num_actions: int = 32

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "verifier_config.json"), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, save_dir: str) -> "ActionEnergyVerifierConfig":
        with open(os.path.join(save_dir, "verifier_config.json")) as f:
            return cls(**json.load(f))


class ActionEnergyVerifier(nn.Module):
    def __init__(self, config: ActionEnergyVerifierConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # VLM feature compression: M learnable queries cross-attend once.
        self.pool_queries = nn.Parameter(torch.zeros(1, config.num_pool_tokens, H))
        self.vlm_proj = nn.Linear(config.vlm_hidden_size, H)
        self.pool_attn = nn.MultiheadAttention(H, config.num_heads, batch_first=True)

        # Token embeddings
        self.action_proj = nn.Linear(config.dim_action, H)
        self.proprio_proj = nn.Linear(config.dim_proprio, H)
        self.energy_token = nn.Parameter(torch.zeros(1, 1, H))

        max_seq = 1 + 1 + config.max_num_actions + config.num_pool_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, H))

        self.blocks = nn.ModuleList(
            [TransformerBlock(H, config.num_heads, mlp_ratio=config.mlp_ratio)
             for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(H)
        self.energy_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(approximate="tanh"),
            nn.Linear(H, 1),
        )

        self.apply(basic_init)
        nn.init.normal_(self.pool_queries, std=0.02)
        nn.init.normal_(self.energy_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def pool_vlm_features(self, vlm_features: torch.Tensor) -> torch.Tensor:
        """Compress [B, T_vlm, D_vlm] -> [B, M, H] once per observation."""
        feats = self.vlm_proj(vlm_features)
        q = self.pool_queries.expand(feats.shape[0], -1, -1)
        pooled, _ = self.pool_attn(q, feats, feats, need_weights=False)
        return pooled

    def energy_from_pooled(
        self,
        pooled_vlm: torch.Tensor,     # [N, M, H]
        proprio: torch.Tensor,        # [N, dim_proprio] (normalized)
        actions_norm: torch.Tensor,   # [N, T, dim_action] (normalized)
    ) -> torch.Tensor:
        """Scalar energy per sample, lower = better."""
        N, T = actions_norm.shape[0], actions_norm.shape[1]
        x = torch.cat(
            [
                self.energy_token.expand(N, -1, -1),
                self.proprio_proj(proprio).unsqueeze(1),
                self.action_proj(actions_norm),
                pooled_vlm,
            ],
            dim=1,
        )
        x = x + self.pos_emb[:, : x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        return self.energy_head(self.norm(x[:, 0])).squeeze(-1)  # [N]

    def forward(
        self,
        vlm_features: torch.Tensor,   # [B, T_vlm, D_vlm]
        proprio: torch.Tensor,        # [B, dim_proprio] (normalized)
        actions_norm: torch.Tensor,   # [B, T, dim_action] (normalized)
    ) -> torch.Tensor:
        pooled = self.pool_vlm_features(vlm_features)
        return self.energy_from_pooled(pooled, proprio, actions_norm)

    def score_candidates(
        self,
        vlm_features: torch.Tensor,   # [B, T_vlm, D_vlm]
        proprio: torch.Tensor,        # [B, dim_proprio] (normalized)
        candidates_norm: torch.Tensor,  # [B, K, T, dim_action] (normalized)
    ) -> torch.Tensor:
        """Energies [B, K]; the VLM pooling runs once and is broadcast to K."""
        B, K = candidates_norm.shape[0], candidates_norm.shape[1]
        pooled = self.pool_vlm_features(vlm_features)               # [B, M, H]
        pooled_k = pooled.repeat_interleave(K, dim=0)               # [B*K, M, H]
        proprio_k = proprio.repeat_interleave(K, dim=0)             # [B*K, dp]
        flat = candidates_norm.reshape(B * K, *candidates_norm.shape[2:])
        return self.energy_from_pooled(pooled_k, proprio_k, flat).view(B, K)

    # ------------------------------ persistence ------------------------------
    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, "verifier.pt"))

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location="cpu") -> "ActionEnergyVerifier":
        config = ActionEnergyVerifierConfig.load(save_dir)
        model = cls(config)
        state = torch.load(
            os.path.join(save_dir, "verifier.pt"),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        return model


def info_nce_loss(
    energy_pos: torch.Tensor,      # [B]
    energy_neg: torch.Tensor,      # [B, N]
    temperature: float = 0.1,
    neg_mask: torch.Tensor | None = None,  # [B, N] True = keep as negative
) -> dict:
    """
    InfoNCE over energies: logits = -E / tau, positive at index 0.

    neg_mask lets the caller drop false negatives (e.g. generator samples that
    landed too close to the ground-truth action).
    """
    logits_neg = -energy_neg / temperature
    if neg_mask is not None:
        logits_neg = logits_neg.masked_fill(~neg_mask, float("-inf"))
    logits = torch.cat([(-energy_pos / temperature).unsqueeze(1), logits_neg], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == 0).float().mean()
        if neg_mask is not None:
            denom = neg_mask.float().sum(dim=1).clamp(min=1.0)
            neg_mean = (energy_neg * neg_mask.float()).sum(dim=1) / denom
        else:
            neg_mean = energy_neg.mean(dim=1)
        margin = (neg_mean - energy_pos).mean()
    return {"loss": loss, "acc": acc, "margin": margin}


__all__ = [
    "ActionEnergyVerifier",
    "ActionEnergyVerifierConfig",
    "info_nce_loss",
]
