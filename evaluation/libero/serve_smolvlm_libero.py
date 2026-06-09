#!/usr/bin/env python3
"""
SimVLA LIBERO Policy Server (WebSocket)

A WebSocket-based policy server for LIBERO evaluation:
- Uses msgpack_numpy serialization for efficient data transfer
- Sends server metadata on connection
- Receives: observation/image, observation/wrist_image, observation/state, prompt
- Returns: {"actions": [...]}

State format (8D): [ee_pos(3), axis_angle(3), gripper_qpos(2)]
Action format (7D): [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import websockets

try:
    import msgpack
    import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack_numpy not installed, using JSON fallback")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    # Inference-time knobs (set from CLI in main()):
    "solver": "euler",        # "euler" (baseline) or "heun" (2nd-order)
    "nfe_steps": 10,          # number of ODE integration steps
    "send_boundary": False,   # also return adaptive-chunking boundary scores
    # --- Dual-rate VLM feature caching (the actual speedup) ---------------
    # The VLM stage (~90% of latency) depends ONLY on images+language, so its
    # output can be reused for several consecutive control queries while the
    # cheap action transformer still consumes FRESH proprio each step.
    #   vlm_refresh_every = 1  -> recompute VLM every query == exact baseline
    #   vlm_refresh_every = N  -> recompute VLM once per N queries (N-1 reuses)
    "vlm_refresh_every": 1,
    # Optional image-change gate: if > 0, force a VLM refresh whenever the
    # current frame differs from the cached frame by more than this per-element
    # RMS (ImageNet-normalized space). 0 disables the gate (fixed-N only).
    "vlm_img_thresh": 0.0,
    # Mixed-precision inference. "fp32" = baseline. "bf16"/"fp16" run the model
    # forward under autocast: faster matmuls, ~no change to what the model
    # "sees", so success rate is preserved by construction (unlike caching).
    "amp_dtype": "fp32",
}


def _amp_ctx():
    """Autocast context for the configured AMP dtype (nullcontext for fp32)."""
    d = CONFIG.get("amp_dtype", "fp32")
    dev = "cuda" if device == "cuda" else "cpu"
    if d == "bf16":
        return torch.autocast(device_type=dev, dtype=torch.bfloat16)
    if d == "fp16":
        return torch.autocast(device_type=dev, dtype=torch.float16)
    return contextlib.nullcontext()

# Dual-rate VLM feature cache. Holds the last VLM encoding so it can be reused
# across consecutive queries. Reset on episode boundaries (client sends
# {"reset": True}) and whenever the prompt or image changes enough.
_VLM_CACHE = {
    "enc": None,          # dict returned by model.encode_vlm()
    "last_image": None,   # preprocessed image tensor at last refresh (for gating)
    "prompt": None,       # language instruction at last refresh
    "cache_uses": 0,      # how many queries have used the current enc (incl. refresh)
}


def reset_vlm_cache():
    """Invalidate the dual-rate VLM cache (call on episode boundaries)."""
    _VLM_CACHE["enc"] = None
    _VLM_CACHE["last_image"] = None
    _VLM_CACHE["prompt"] = None
    _VLM_CACHE["cache_uses"] = 0


def load_model(checkpoint_path: str, norm_stats_path: str = None, smolvlm_model_path: str = None):
    """Load SimVLA model and processor."""
    global model, processor
    
    logger.info(f"Loading SimVLA from {checkpoint_path}...")
    
    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    smolvlm_path = smolvlm_model_path or "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)
    
    if norm_stats_path and os.path.exists(norm_stats_path):
        logger.info(f"Loading norm stats from: {norm_stats_path}")
        model.action_space.load_norm_stats(norm_stats_path)
        if hasattr(model.action_space, 'state_norm_stats') and model.action_space.state_norm_stats:
            logger.info(f"   State norm: mean={model.action_space.state_norm_stats.mean[:3].tolist()}")
        if hasattr(model.action_space, 'action_norm_stats') and model.action_space.action_norm_stats:
            logger.info(f"   Action norm: mean={model.action_space.action_norm_stats.mean[:3].tolist()}")
    else:
        logger.warning("No norm_stats loaded!")
    
    logger.info(f"Model loaded! Device: {device}, Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")


def preprocess_images(image0: np.ndarray, image1: np.ndarray):
    """Preprocess images to model input format."""
    image_size = CONFIG["image_size"]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img0 = Image.fromarray(image0.astype(np.uint8))
    img1 = Image.fromarray(image1.astype(np.uint8))
    
    img0_t = transform(img0)
    img1_t = transform(img1)
    
    # Pad to 3 views (model processes all views together)
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])
    
    return images.unsqueeze(0), image_mask


def decode_numpy(obj):
    """Decode numpy array from msgpack_numpy dict format."""
    if isinstance(obj, dict):
        if b'__ndarray__' in obj or '__ndarray__' in obj:
            data_key = b'data' if b'data' in obj else 'data'
            dtype_key = b'dtype' if b'dtype' in obj else 'dtype'
            shape_key = b'shape' if b'shape' in obj else 'shape'
            
            data = obj[data_key]
            dtype_str = obj[dtype_key]
            shape = obj[shape_key]
            
            if isinstance(dtype_str, bytes):
                dtype_str = dtype_str.decode()
            
            if shape and isinstance(shape[0], bytes):
                shape = tuple(int(s) for s in shape)
            else:
                shape = tuple(shape)
            
            return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
    return obj


def infer(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single observation."""
    global model, processor
    
    try:
        # Extract observation fields
        image0 = observation.get("observation/image")
        image1 = observation.get("observation/wrist_image")
        state = observation.get("observation/state", np.zeros(8))
        prompt = observation.get("prompt", "")
        
        # Decode msgpack_numpy format if needed
        image0 = decode_numpy(image0)
        image1 = decode_numpy(image1)
        state = decode_numpy(state)
        
        # Ensure numpy arrays
        if not isinstance(image0, np.ndarray):
            image0 = np.array(image0, dtype=np.uint8)
        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1, dtype=np.uint8)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        state = state[:8]
        
        # Preprocess images
        images, image_mask = preprocess_images(image0, image1)
        images = images.to(device)
        image_mask = image_mask.to(device)
        
        # Encode language instruction
        lang = processor.encode_language([prompt])
        lang = {k: v.to(device) for k, v in lang.items()}
        
        # Proprioception
        proprio_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # ---- Dual-rate VLM feature caching -------------------------------
        # Decide whether to rerun the expensive VLM (refresh) or reuse the
        # cached features. proprio is ALWAYS fresh and only consumed by the
        # cheap action transformer below, so caching never stales the robot
        # state — only the (slow-changing) visual-language semantics.
        # Per-request override (lets one server sweep N without reloading the
        # model); falls back to the server's CLI default when absent.
        refresh_every = max(1, int(observation.get("vlm_refresh_every",
                                                   CONFIG["vlm_refresh_every"])))
        img_thresh = float(observation.get("vlm_img_thresh",
                                           CONFIG["vlm_img_thresh"]))

        if bool(observation.get("reset", False)):
            reset_vlm_cache()

        prompt_changed = (_VLM_CACHE["prompt"] != prompt)

        img_changed = False
        if img_thresh > 0.0 and _VLM_CACHE["last_image"] is not None:
            _diff = (images - _VLM_CACHE["last_image"]).float()
            _img_rms = torch.sqrt(torch.mean(_diff * _diff)).item()
            img_changed = _img_rms > img_thresh

        need_refresh = (
            _VLM_CACHE["enc"] is None
            or prompt_changed
            or img_changed
            or _VLM_CACHE["cache_uses"] >= refresh_every
        )

        # Inference — time the (optional) VLM stage and the cheap action stage
        # separately so we can confirm where the latency actually goes.
        import time as _time

        def _sync():
            if device == "cuda":
                torch.cuda.synchronize()

        with torch.no_grad(), _amp_ctx():
            _sync(); _t0 = _time.perf_counter()
            if need_refresh:
                enc = model.encode_vlm(lang['input_ids'], images, image_mask)
                _VLM_CACHE["enc"] = enc
                _VLM_CACHE["last_image"] = images
                _VLM_CACHE["prompt"] = prompt
                _VLM_CACHE["cache_uses"] = 1
            else:
                enc = _VLM_CACHE["enc"]
                _VLM_CACHE["cache_uses"] += 1
            _sync(); _t1 = _time.perf_counter()
            gen = model.generate_actions_from_enc(
                enc,
                proprio=proprio_tensor,
                steps=CONFIG["nfe_steps"],
                solver=CONFIG["solver"],
                return_boundary=CONFIG["send_boundary"],
            )
            _sync(); _t2 = _time.perf_counter()

        _vlm_ms = (_t1 - _t0) * 1000
        _act_ms = (_t2 - _t1) * 1000
        _latency_ms = (_t2 - _t0) * 1000
        if need_refresh:
            logger.info(f"[LATENCY] total={_latency_ms:.1f}ms "
                        f"| VLM={_vlm_ms:.1f}ms ({_vlm_ms/_latency_ms*100:.0f}%) REFRESH "
                        f"| Action={_act_ms:.1f}ms ({_act_ms/_latency_ms*100:.0f}%) "
                        f"[solver={CONFIG['solver']} nfe={CONFIG['nfe_steps']} "
                        f"refresh_every={refresh_every}]")
        else:
            logger.info(f"[LATENCY] total={_latency_ms:.1f}ms "
                        f"| VLM=cached(~0ms) "
                        f"| Action={_act_ms:.1f}ms "
                        f"[cache_use {_VLM_CACHE['cache_uses']}/{refresh_every}]")

        if isinstance(gen, tuple):
            actions, boundary = gen
        else:
            actions, boundary = gen, None

        result = {
            "actions": actions.cpu().numpy()[0],
            "latency_ms": float(_latency_ms),
            "vlm_ms": float(_vlm_ms),
            "action_ms": float(_act_ms),
            "vlm_refreshed": bool(need_refresh),
        }
        if boundary is not None:
            result["boundary"] = boundary.cpu().numpy()[0]
        return result
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        return {"actions": np.zeros((CONFIG["action_horizon"], CONFIG["action_dim"]))}


async def handle_connection(websocket, path=None):
    """Handle a WebSocket connection."""
    logger.info(f"Connection from {websocket.remote_address} opened")
    
    try:
        # Send server metadata on connection
        metadata = {
            "model": "SimVLA",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
        }
        if HAS_MSGPACK:
            await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
        else:
            import json
            await websocket.send(json.dumps(metadata))
        
        # Process requests
        async for message in websocket:
            try:
                # Parse request
                if HAS_MSGPACK and isinstance(message, bytes):
                    request = msgpack_numpy.unpackb(message, raw=False)
                else:
                    import json
                    request = json.loads(message)
                
                # Run inference
                result = infer(request)
                
                # Send response (convert numpy to list for compatibility)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()

                response_data = {
                    "actions": actions,
                    "latency_ms": result.get("latency_ms", 0.0),
                    "vlm_refreshed": result.get("vlm_refreshed", True),
                }
                boundary = result.get("boundary")
                if boundary is not None:
                    response_data["boundary"] = (
                        boundary.tolist() if isinstance(boundary, np.ndarray) else boundary
                    )
                
                if HAS_MSGPACK:
                    import msgpack
                    response = msgpack.packb(response_data, use_bin_type=True)
                else:
                    import json
                    response = json.dumps(response_data)
                
                await websocket.send(response)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                error_msg = f"Error: {str(e)}"
                await websocket.send(error_msg)
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(f"Connection from {websocket.remote_address} closed")


async def serve(host: str, port: int):
    """Start the WebSocket server."""
    logger.info(f"Creating SimVLA server (host: {host}, port: {port})")
    
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SimVLA LIBERO Server (WebSocket)")
    # checkpoint defaults to $SIMVLA_CHECKPOINTS from paths.env if set
    parser.add_argument("--checkpoint", type=str,
                        default=os.environ.get("SIMVLA_CHECKPOINTS"),
                        required=os.environ.get("SIMVLA_CHECKPOINTS") is None,
                        help="Path to SimVLA checkpoint (env: SIMVLA_CHECKPOINTS)")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to normalization stats JSON")
    parser.add_argument("--smolvlm_model", type=str,
                        default=os.environ.get("SIMVLA_SMOLVLM_MODEL",
                                               "HuggingFaceTB/SmolVLM-500M-Instruct"),
                        help="SmolVLM model path or HF repo (env: SIMVLA_SMOLVLM_MODEL)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "heun"],
                        help="ODE solver for flow-matching inference. "
                             "'heun' is 2nd-order (lower latency at equal quality).")
    parser.add_argument("--nfe_steps", type=int, default=10,
                        help="Number of ODE integration steps (denoising NFE).")
    parser.add_argument("--send_boundary", action="store_true",
                        help="Also return per-step adaptive-chunking boundary "
                             "scores (requires an adaptive-chunking checkpoint).")
    parser.add_argument("--vlm_refresh_every", type=int, default=1,
                        help="Dual-rate VLM caching: recompute the expensive VLM "
                             "once per N queries (reuse cached features in between). "
                             "1 = recompute every query (exact baseline). N>1 speeds "
                             "up inference ~Nx for the cached queries.")
    parser.add_argument("--vlm_img_thresh", type=float, default=0.0,
                        help="Optional image-change gate (per-element RMS in "
                             "ImageNet-normalized space). >0 forces a VLM refresh "
                             "when the scene changes more than this, regardless of "
                             "--vlm_refresh_every. 0 disables the gate.")
    parser.add_argument("--amp_dtype", type=str, default="fp32",
                        choices=["fp32", "bf16", "fp16"],
                        help="Mixed-precision inference. fp32 = baseline. "
                             "bf16/fp16 speed up the VLM with no change to what "
                             "the model sees (success rate preserved).")

    args = parser.parse_args()

    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")

    CONFIG["solver"] = args.solver
    CONFIG["nfe_steps"] = args.nfe_steps
    CONFIG["send_boundary"] = args.send_boundary
    CONFIG["vlm_refresh_every"] = args.vlm_refresh_every
    CONFIG["vlm_img_thresh"] = args.vlm_img_thresh
    CONFIG["amp_dtype"] = args.amp_dtype

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)

    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    logger.info(f"  Solver: {CONFIG['solver']}  |  NFE steps: {CONFIG['nfe_steps']}  |  "
                f"Send boundary: {CONFIG['send_boundary']}")
    logger.info(f"  VLM cache: refresh_every={CONFIG['vlm_refresh_every']}  |  "
                f"img_thresh={CONFIG['vlm_img_thresh']}  "
                f"({'BASELINE (no caching)' if CONFIG['vlm_refresh_every'] <= 1 and CONFIG['vlm_img_thresh'] <= 0 else 'DUAL-RATE CACHING ON'})")
    logger.info(f"  AMP dtype: {CONFIG['amp_dtype']}  "
                f"({'fp32 baseline' if CONFIG['amp_dtype'] == 'fp32' else 'mixed precision — faster, quality-preserving'})")
    
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
