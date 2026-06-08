"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone.

Key differences from FlorenceVLA:
  - Uses SmolVLM-500M-Instruct (efficient 500M parameter model)
  - 512x512 image input (SmolVLM-500M uses 512x512 patches)
  - All views processed together by SmolVLM, no aux_visual_inputs
  - Unified VLM output for multi-view inputs
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • SmolVLM-500M-Instruct backbone (vision-language)
      • SmolVLMActionTransformer (flow matching action head)
      • Action space (pre/post-processing + loss)
      
    Key differences from FlorenceVLA:
      • All camera views are input to VLM together (no aux_visual_inputs)
      • 512x512 image resolution (SmolVLM-500M uses 512x512 patches)
      • Efficient 500M parameter model
    """
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views
        
        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # SmolVLM backbone
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_model_path,
            torch_dtype=torch.float32,  # Use float32 for training stability
            trust_remote_code=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )
        
        # Get SmolVLM hidden size from model config
        # SmolVLM-500M has hidden_size from text_config
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # DiT/AdaLN mode setting
        self.use_adaln = getattr(config, 'use_adaln', False)

        # Adaptive action chunking
        self.use_adaptive_chunking = getattr(config, 'use_adaptive_chunking', False)
        self.chunk_loss_weight = getattr(config, 'chunk_loss_weight', 0.1)

        # Flow matching action head (SmolVLM version - no aux_visual)
        self.transformer = SmolVLMActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_adaln=self.use_adaln,
            use_adaptive_chunking=self.use_adaptive_chunking,
        )
        
        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= SmolVLM encoder =============================
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - multi-view images
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        language_instruction: list[str] | None = None,  # Optional text prompts
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-view images via SmolVLM2.
        
        All views are processed together by SmolVLM, producing unified features.
        No aux_visual_inputs needed - everything goes through VLM.

        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
            
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # Prepare images for SmolVLM - flatten views and filter by mask
        # SmolVLM can handle multiple images as part of multi-image inference
        batch_features = []
        
        for b in range(B):
            # Get valid images for this sample
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]  # [num_valid, C, H, W]
            
            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")
            
            # Convert to PIL images for SmolVLM processor
            pil_images = []
            for img_tensor in valid_images:
                # Denormalize and convert to PIL
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Assuming normalized with ImageNet stats, denormalize
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Build message for SmolVLM with multiple images
            content = []
            for i, img in enumerate(pil_images):
                content.append({"type": "image", "image": img})
            
            # Add text prompt if provided
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})
            
            messages = [{"role": "user", "content": content}]
            
            # Process with SmolVLM
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Get encoder outputs (hidden states) instead of generating text
            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Use the last hidden state as features
            # Shape: [1, seq_len, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))  # [seq_len, hidden_size]
        
        # Pad to same length and stack
        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]
        
        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat
        
        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - Already preprocessed
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - Pre-tokenized text
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient VLM forward for training - uses FULL VLM to fuse vision and language.
        
        Key improvement: Uses complete VLM forward (vision encoder + language model)
        to get features that fuse visual and linguistic information, rather than
        just using the vision encoder alone.
        
        Pipeline:
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return
        
        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # ========== Step 1: Get vision features ==========
        # Flatten images: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        
        # Get valid images
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]
        
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        
        # Encode images through SmolVLM's vision encoder (SigLIP)
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get image features and project to LM space
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]
        
        # Project to language model space using the connector/projector
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)
        
        # ========== Step 2: Get text embeddings ==========
        # Idefics3 (SmolVLM) uses 'text_model' instead of 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # ========== Step 3: Build combined sequence per sample ==========
        # For each sample, concatenate: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        
        # Reconstruct image features with batch structure
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        
        # Count valid views per sample for proper concatenation
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]
        
        batch_inputs_embeds = []
        max_seq_len = 0
        
        for b in range(B):
            # Get valid image features for this sample
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]
            
            # Get text embeddings for this sample
            sample_text_embeds = text_embeds[b]  # [L, D]
            
            # Concatenate: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])
        
        # ========== Step 4: Pad and stack ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1
        
        # ========== Step 5: Forward through text model (Idefics3/SmolVLM) ==========
        # This fuses visual and linguistic information through the full transformer
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use the last hidden state as VLM features
        # This now contains fused vision-language representations
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]
        
        return {"vlm_features": vlm_features}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor,     # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V]
        proprio: torch.Tensor,              # [B, dim_proprio]
        action: torch.Tensor,               # [B, T=num_actions, D=dim_action]
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training.
        
        1) Time sampling: t ~ Beta(1.5, 1) * 0.999 + 0.001
        2) Interpolation: x_t = t * noise + (1-t) * actions
        3) Target: velocity u_t = noise - actions
        4) Model predicts v_t, compute MSE(v_t, u_t)
        """
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        device = input_ids.device
        
        # Beta(1.5, 1) time sampling
        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device), 
            torch.tensor(1.0, device=device)
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001

        # Normalize action and proprio
        if hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action
            
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio
        
        # Flow Matching
        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        if self.use_adaptive_chunking:
            # Compute per-step change-rate weights from ground-truth actions.
            # delta[b, t] = ||a[t+1] - a[t]|| measures how abruptly the
            # trajectory changes at each timestep.  High-change moments
            # (contact, direction reversal) get larger weights so the model is
            # penalised more for errors there; smooth free-space motion gets
            # smaller weights.
            # Shape progression: [B,T,D] -> [B,T-1] -> [B,T]
            T = action_norm.shape[1]
            if T >= 2:
                delta = action_norm[:, 1:] - action_norm[:, :-1]      # [B, T-1, D]
                change_rate = delta.norm(dim=-1)                       # [B, T-1]
                # Prepend the first step's rate so length == T
                change_rate = torch.cat([change_rate[:, :1], change_rate], dim=1)  # [B, T]
            else:
                # Chunk of length 1: no notion of change rate, fall back to flat.
                change_rate = torch.zeros_like(action_norm[..., 0])    # [B, T]
            # Normalise per sample to [0, 1] (normalized_rate), then shift to
            # [0.5, 1.5] so the minimum weight is 0.5 (never fully ignored)
            # and the max is 1.5.
            rate_max = change_rate.amax(dim=1, keepdim=True).clamp(min=1e-6)
            normalized_rate = change_rate / rate_max                   # [B, T] in [0,1]
            step_weights = 0.5 + normalized_rate                       # [B, T] in [0.5,1.5]

            v_t, boundary_logits = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                t=t,
                proprio=proprio_norm,
                return_boundary=True,
            )

            # Change-rate-weighted velocity loss
            sq_err = torch.square(v_t - u_t)                           # [B, T, D]
            velocity_loss = (step_weights.unsqueeze(-1) * sq_err).mean()

            # Boundary head regresses the (detached) normalised change rate so
            # gradient only flows through the head, not the target.
            boundary_loss = torch.nn.functional.mse_loss(
                boundary_logits, normalized_rate.detach()
            )
            return {
                "velocity_loss": velocity_loss,
                "boundary_loss": self.chunk_loss_weight * boundary_loss,
            }

        # ----- Default path: uniform flow-matching loss -----
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )
        velocity_loss = torch.mean(torch.square(v_t - u_t))
        return {"velocity_loss": velocity_loss}

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Flow Matching inference (Euler integration).

        1) Initialize x_t = noise (t=1)
        2) Loop t from 1 to 0:
           - Model predicts velocity v_t
           - Euler update: x_t = x_t + dt * v_t
        3) Final x_0 ≈ target action
        """
        self.eval()
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)
        return self.generate_actions_from_enc(enc, proprio, steps=steps)

    @torch.no_grad()
    def generate_actions_from_enc(
        self,
        enc: Dict[str, torch.Tensor],
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """Euler flow matching from pre-computed VLM features."""
        self.eval()
        B = enc["vlm_features"].shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        steps = max(1, int(steps))
        dt = -1.0 / steps
        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t
            t = t + dt

        return self.action_space.postprocess(x_t)

    @torch.no_grad()
    def forward_vlm_with_cache(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        cache: dict = None,
        alpha: float = 1.0,
        beta: float = 0.9,
        warmup_steps: int = 3,
        fixed_threshold: float = None,
    ):
        """
        Adaptive Temporal Token Cache (ATTC) forward.

        Per-view caching: for each camera view, computes patch-level pixel
        differences against the previous frame.  An EMA-tracked adaptive
        threshold τ = EMA_mean + α·EMA_std decides per-view:
          - diff > τ  →  re-encode with SigLIP  (scene changed)
          - diff ≤ τ  →  reuse cached features  (cache hit)

        The adaptive threshold rises automatically during fast arm motion /
        contact events and falls during slow free-space travel, trading off
        speed vs. accuracy without manual tuning.

        Args:
            cache:           None on episode start, else dict returned by previous call
            alpha:           threshold multiplier (higher → more caching)
            beta:            EMA decay (~0.9 = 10-frame window)
            warmup_steps:    encode fully for first N steps to warm up EMA stats
            fixed_threshold: if not None, use this CONSTANT threshold instead of
                             the adaptive τ (ablation baseline, e.g. VLA-Cache).
                             EMA stats are still tracked for logging but ignored.

        Returns:
            enc:          {"vlm_features": [B, T, D]}
            updated_cache: pass to next call; reset to None on episode start
            stats:        {"cache_hits": int, "total_views": int, "hit_rate": float}
        """
        if pixel_values.dim() == 6:
            pixel_values = pixel_values.squeeze(2) if pixel_values.size(2) == 1 \
                else pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        try:
            patch_size = self.vlm.model.vision_model.config.patch_size
        except AttributeError:
            patch_size = 14
        h_p = max(1, H // patch_size)
        w_p = max(1, W // patch_size)

        # Initialise cache on first call
        if cache is None:
            cache = {
                "step_count": 0,
                "patch_features": None,      # [B*V, num_patches, lm_hidden] on CPU
                "prev_pixel_values": None,   # [B*V, C, H, W] on CPU float32
                "ema_mean": np.zeros((B, V), dtype=np.float32),
                # Start ema_std at zero so EMA converges from actual data before
                # caching begins — ema_std=0.1 would make threshold ~0.1 which is
                # far above typical ImageNet-normalised frame diffs (~0.05), causing
                # near-total caching and catastrophic failure.
                "ema_std":  np.zeros((B, V), dtype=np.float32),
                "cache_hits": 0,
                "total_views": 0,
            }
        cache = {k: v for k, v in cache.items()}  # shallow copy

        step = cache["step_count"]
        flat_images = pixel_values.flatten(0, 1)   # [B*V, C, H, W]
        flat_mask   = image_mask.view(-1).bool()   # [B*V]
        valid_indices = [i for i in range(B * V) if flat_mask[i]]

        # ---- Decide which views need re-encoding --------------------------------
        # EMA is updated for EVERY step (including warmup) so that, by the time we
        # start making cache decisions (step >= warmup_steps), the statistics
        # reflect the actual per-view frame-diff distribution.
        view_needs_encode = {}  # bv_idx -> bool
        for bv_idx in valid_indices:
            b_idx = bv_idx // V
            v_idx = bv_idx  % V

            if cache["prev_pixel_values"] is None:
                # Very first step: no previous frame to compare against
                view_needs_encode[bv_idx] = True
                continue

            curr = flat_images[bv_idx]                                   # [C,H,W]
            prev = cache["prev_pixel_values"][bv_idx].to(device=device, dtype=dtype)
            frame_mean = (curr - prev).abs().mean().item()

            # Always update EMA (even during warmup) for calibration
            em = float(cache["ema_mean"][b_idx, v_idx])
            es = float(cache["ema_std"][b_idx,  v_idx])
            new_em = beta * em + (1 - beta) * frame_mean
            new_es = beta * es + (1 - beta) * abs(frame_mean - em)
            cache["ema_mean"][b_idx, v_idx] = new_em
            cache["ema_std"][b_idx,  v_idx] = new_es

            # During warmup: always encode (EMA is being calibrated)
            if step < warmup_steps:
                view_needs_encode[bv_idx] = True
                continue

            if fixed_threshold is not None:
                # Ablation baseline: constant threshold (VLA-Cache style)
                threshold = fixed_threshold
            else:
                # Adaptive threshold τ = EMA_mean + α·EMA_std
                threshold = new_em + alpha * new_es
            view_needs_encode[bv_idx] = frame_mean > threshold

        encode_indices = [i for i in valid_indices if     view_needs_encode[i]]
        cached_indices = [i for i in valid_indices if not view_needs_encode[i]]
        num_valid      = len(valid_indices)
        cache_hits     = len(cached_indices)
        cache["cache_hits"]   += cache_hits
        cache["total_views"]  += num_valid

        # ---- Encode views that changed ------------------------------------------
        if encode_indices:
            imgs_to_encode = flat_images[encode_indices]
            vis_out = self.vlm.model.vision_model(
                pixel_values=imgs_to_encode,
                output_hidden_states=True,
                return_dict=True,
            )
            new_feats = vis_out.last_hidden_state   # [n_enc, num_patches, vis_hid]
            if hasattr(self.vlm.model, 'connector'):
                new_feats = self.vlm.model.connector(new_feats)
            elif hasattr(self.vlm.model, 'multi_modal_projector'):
                new_feats = self.vlm.model.multi_modal_projector(new_feats)
        else:
            new_feats = None

        # ---- Determine num_patches / hidden_size --------------------------------
        if new_feats is not None:
            num_patches = new_feats.shape[1]
            hidden_size = new_feats.shape[2]
        else:
            num_patches = cache["patch_features"].shape[1]
            hidden_size = cache["patch_features"].shape[2]

        # ---- Assemble image_features [num_valid, num_patches, hidden_size] ------
        flat_to_rank = {bv: r for r, bv in enumerate(valid_indices)}
        image_features = torch.zeros(
            num_valid, num_patches, hidden_size, device=device, dtype=dtype
        )
        for enc_rank, bv_idx in enumerate(encode_indices):
            image_features[flat_to_rank[bv_idx]] = new_feats[enc_rank]
        for bv_idx in cached_indices:
            image_features[flat_to_rank[bv_idx]] = \
                cache["patch_features"][bv_idx].to(device=device, dtype=dtype)

        # ---- Update feature cache & prev pixels ---------------------------------
        if cache["patch_features"] is None:
            cache["patch_features"] = torch.zeros(
                B * V, num_patches, hidden_size, dtype=torch.float32
            )
        for enc_rank, bv_idx in enumerate(encode_indices):
            cache["patch_features"][bv_idx] = new_feats[enc_rank].cpu().float()
        cache["prev_pixel_values"] = flat_images.cpu().float()
        cache["step_count"] = step + 1

        # ---- Continue with text model (same as forward_vlm_efficient) -----------
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)

        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)

        valid_per_sample = image_mask.sum(dim=1).int()
        batch_inputs_embeds = []
        max_seq_len = 0
        for b in range(B):
            n_valid = valid_per_sample[b].item()
            img_feats = full_image_features[b, :n_valid].reshape(-1, hidden_size)
            combined = torch.cat([img_feats, text_embeds[b]], dim=0)
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])

        padded = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attn_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        for b, emb in enumerate(batch_inputs_embeds):
            padded[b, :emb.shape[0]] = emb
            attn_mask[b, :emb.shape[0]] = 1

        lm_out = self.vlm.model.text_model(
            inputs_embeds=padded,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        vlm_features = lm_out.last_hidden_state

        enc = {"vlm_features": vlm_features}
        hit_rate = cache_hits / num_valid if num_valid > 0 else 0.0
        stats = {
            "cache_hits": cache_hits,
            "total_views": num_valid,
            "hit_rate": hit_rate,
            "encode_count": len(encode_indices),
        }
        return enc, cache, stats

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """Build FastAPI app for SmolVLM-VLA inference."""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode images
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                        
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Process inputs
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio tensor
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                # Align to model device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                # Inference
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
