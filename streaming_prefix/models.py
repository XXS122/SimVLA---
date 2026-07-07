"""
Phase B: learned state-update operator that replaces SmolVLM's fused
text-model forward (the ~72% of VLM-prefix cost per profile_prefix.py)
with a cheap recurrent update, given the previous step's real output and
this step's cheaply-recomputed vision+connector+text-embedding features.

Split of forward_vlm_efficient (models/modeling_smolvlm_vla.py):
  cheap  (~23% of VLM cost): vision_model + connector + text embedding
                            lookup + padding -> `combined_embeds`
  expensive (~72%):          text_model(inputs_embeds=combined_embeds) ->
                            `vlm_features` (what the action transformer
                            actually conditions on)

VLMPrefixTeacher exposes both halves so the cheap half can be recomputed
every step (fresh, correct) while PrefixStateUpdater learns to
approximate the expensive half's *change* from the cached previous
output -- avoiding the LAM's copy-shortcut mistake (Innovation 4) by
predicting a delta with a normalized, interpretable loss scale: 1.0 ==
"predicts zero change", must drop clearly below 1.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_smolvlm import TransformerBlock


class VLMPrefixTeacher(nn.Module):
    """Wraps a frozen SmolVLM (AutoModelForImageTextToText, same class
    SmolVLMVLA loads) to expose the cheap/expensive split of
    forward_vlm_efficient separately. No gradients flow through this
    module; it is the ground-truth signal for distillation.
    """

    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm
        for p in self.vlm.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def cheap_forward(
        self, image_input: torch.Tensor, image_mask: torch.Tensor, input_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vision tower + connector + text embeddings + padding. Cheap
        (~23% of VLM cost); safe to recompute every control step.

        Returns (combined_embeds [B, L, D], attention_mask [B, L]).
        """
        if image_input.dim() == 6:
            image_input = image_input[:, :, 0] if image_input.size(2) > 1 else image_input.squeeze(2)
        B, V = image_input.shape[:2]
        device = image_input.device

        flat_images = image_input.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]

        vision_out = self.vlm.model.vision_model(pixel_values=valid_images, return_dict=True).last_hidden_state
        connector = getattr(self.vlm.model, "connector", None) or getattr(self.vlm.model, "multi_modal_projector")
        image_features = connector(vision_out)
        hidden_size, num_patches = image_features.shape[-1], image_features.shape[1]

        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)

        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        valid_per_sample = image_mask.sum(dim=1).int()

        combined_list, max_len = [], 0
        for b in range(B):
            n = valid_per_sample[b].item()
            img = full_image_features[b, :n].reshape(-1, hidden_size)
            combined = torch.cat([img, text_embeds[b]], dim=0)
            combined_list.append(combined)
            max_len = max(max_len, combined.shape[0])

        padded = torch.zeros(B, max_len, hidden_size, device=device, dtype=image_features.dtype)
        attn = torch.zeros(B, max_len, device=device, dtype=torch.long)
        for b, c in enumerate(combined_list):
            padded[b, :c.shape[0]] = c
            attn[b, :c.shape[0]] = 1
        return padded, attn

    @torch.no_grad()
    def expensive_forward(self, combined_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The fused text-model forward (~72% of VLM cost) -- what the
        state-update operator is trained to approximate without running.
        """
        out = self.vlm.model.text_model(inputs_embeds=combined_embeds, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state

    @torch.no_grad()
    def forward(
        self, image_input: torch.Tensor, image_mask: torch.Tensor, input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Full real forward (cheap + expensive) -- ground truth vlm_features."""
        combined, attn = self.cheap_forward(image_input, image_mask, input_ids)
        return self.expensive_forward(combined, attn)


class PrefixStateUpdater(nn.Module):
    """Predicts vlm_features_t from (cached vlm_features_{t-1}, this
    step's cheap combined_embeds_t), replacing the expensive text-model
    forward at non-reset steps.

    Trained via self-distillation: normalized MSE between the predicted
    delta and the real delta (real_vlm_features_t - cached_state), so the
    loss has an absolute scale (1.0 == predicts zero change / useless,
    same convention as latent_action's LAM delta-prediction fix).
    """

    def __init__(self, hidden_size: int, depth: int = 4, num_heads: int = 8,
                mlp_ratio: float = 4.0, max_len_seq: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.new_proj = nn.Linear(hidden_size, hidden_size)
        self.cache_proj = nn.Linear(hidden_size, hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # start as identity (delta=0) at init

    def forward(self, cached_state: torch.Tensor, combined_embeds_t: torch.Tensor) -> torch.Tensor:
        """cached_state, combined_embeds_t: [B, L, D] (same L: fixed camera
        views + fixed instruction length within an episode). Returns the
        predicted vlm_features_t (cached_state + predicted delta).
        """
        L = combined_embeds_t.shape[1]
        x = self.new_proj(combined_embeds_t) + self.cache_proj(cached_state) + self.pos_emb[:, :L]
        for blk in self.blocks:
            x = blk(x)
        delta_hat = self.out_proj(self.norm(x))
        return cached_state + delta_hat

    def config_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "depth": len(self.blocks),
            "num_heads": self.blocks[0].attn.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "max_len_seq": self.pos_emb.shape[1],
        }


def distillation_loss(pred: torch.Tensor, target: torch.Tensor, cached_state: torch.Tensor,
                      attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Normalized delta-reconstruction loss, masked to valid (non-pad)
    tokens. 1.0 == the updater predicts zero change (useless); healthy
    training must push this clearly below 1 (same convention as the LAM
    fix in latent_action/models.py).
    """
    mask = attention_mask.unsqueeze(-1).to(pred.dtype)  # [B, L, 1]
    target_delta = target - cached_state
    delta_energy = (target_delta.pow(2) * mask).sum() / (mask.sum() * target.shape[-1] + 1e-8)
    sq_err = ((pred - target).pow(2) * mask).sum() / (mask.sum() * target.shape[-1] + 1e-8)
    recon = sq_err / (delta_energy.detach() + 1e-8)
    return {"recon_loss": recon, "delta_energy": delta_energy.detach()}


def masked_norm_mse(pred: torch.Tensor, target: torch.Tensor, denom_energy: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE(pred, target) normalized by a supplied energy scalar.

    Used by rollout training so every step shares ONE consistent
    denominator (the real single-step change energy), keeping the 1.0 ==
    "predicts no change" convention stable even when the model is fed its
    own (erroneous) previous prediction under scheduled sampling.
    """
    mask = attention_mask.unsqueeze(-1).to(pred.dtype)
    sq_err = ((pred - target).pow(2) * mask).sum() / (mask.sum() * target.shape[-1] + 1e-8)
    return sq_err / (denom_energy + 1e-8)


def save_updater(model: PrefixStateUpdater, path, extra: dict | None = None):
    payload = {"config": model.config_dict(), "state_dict": model.state_dict()}
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_updater(path, map_location="cpu") -> PrefixStateUpdater:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = PrefixStateUpdater(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model
