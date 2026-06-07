#!/usr/bin/env python3
"""
SimVLA LIBERO Policy Server (WebSocket)

A WebSocket-based policy server for LIBERO evaluation:
- Uses msgpack_numpy serialization for efficient data transfer
- Sends server metadata on connection
- Receives: observation/image, observation/wrist_image, observation/state, prompt
- Optional {"reset": true} field resets the temporal visual feature cache
- Returns: {"actions": [...]}

State format (8D): [ee_pos(3), axis_angle(3), gripper_qpos(2)]
Action format (7D): [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]
"""

import argparse
import asyncio
import logging
import os
import sys
import time
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

# Global model state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    # Cache settings (populated in main())
    "use_cache": False,
    "cache_alpha": 1.0,
    "cache_beta": 0.9,
    "cache_warmup": 3,
    "cache_log_interval": 20,
    # Latency tracking (populated at runtime)
    "_latency_warmup": 5,       # skip first N steps (GPU warm-up)
    "_latency_samples": [],     # per-step latencies (ms)
}


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
    if CONFIG["use_cache"]:
        logger.info(
            f"  ATTC cache ENABLED: alpha={CONFIG['cache_alpha']}, "
            f"beta={CONFIG['cache_beta']}, warmup={CONFIG['cache_warmup']}"
        )
    else:
        logger.info("  ATTC cache DISABLED (pass --use_cache to enable)")


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


def infer(observation: Dict[str, Any], cache: dict = None):
    """
    Run inference on a single observation.

    Args:
        observation: dict with keys observation/image, observation/wrist_image,
                     observation/state, prompt.  An optional "reset" key (bool)
                     resets the visual feature cache before encoding.
        cache:       per-connection ATTC cache dict (None = disabled or first step)

    Returns:
        (result_dict, updated_cache)
        result_dict has key "actions".
    """
    global model, processor

    t_start = time.perf_counter()

    try:
        # Extract observation fields
        image0 = observation.get("observation/image")
        image1 = observation.get("observation/wrist_image")
        state  = observation.get("observation/state", np.zeros(8))
        prompt = observation.get("prompt", "")
        reset  = bool(observation.get("reset", False))

        # Decode msgpack_numpy format if needed
        image0 = decode_numpy(image0)
        image1 = decode_numpy(image1)
        state  = decode_numpy(state)

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
        images     = images.to(device)
        image_mask = image_mask.to(device)

        # Encode language instruction
        lang = processor.encode_language([prompt])
        lang = {k: v.to(device) for k, v in lang.items()}

        # Proprioception
        proprio_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            if CONFIG["use_cache"]:
                if reset:
                    cache = None   # hard reset on episode boundary

                enc, cache, stats = model.forward_vlm_with_cache(
                    pixel_values=images,
                    image_mask=image_mask,
                    input_ids=lang['input_ids'],
                    cache=cache,
                    alpha=CONFIG["cache_alpha"],
                    beta=CONFIG["cache_beta"],
                    warmup_steps=CONFIG["cache_warmup"],
                )
                actions = model.generate_actions_from_enc(
                    enc, proprio_tensor, steps=CONFIG["action_horizon"]
                )

                # Periodic cache stats logging
                total = cache.get("total_views", 0)
                if total > 0 and total % CONFIG["cache_log_interval"] == 0:
                    cumulative_hr = cache["cache_hits"] / total
                    logger.info(
                        f"[ATTC] cumulative hit rate: {cumulative_hr:.1%}  "
                        f"(this step: {stats['hit_rate']:.1%}, "
                        f"re-encoded {stats['encode_count']}/{stats['total_views']} views)"
                    )
            else:
                actions = model.generate_actions(
                    input_ids=lang['input_ids'],
                    image_input=images,
                    image_mask=image_mask,
                    proprio=proprio_tensor,
                    steps=CONFIG["action_horizon"],
                )
                cache = None

        actions = actions.cpu().numpy()[0]

        # Track latency (skip warm-up steps)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        samples = CONFIG["_latency_samples"]
        if len(samples) >= CONFIG["_latency_warmup"]:
            samples.append(elapsed_ms)
            if len(samples) % 20 == 0:
                avg = sum(samples) / len(samples)
                p50 = sorted(samples)[len(samples) // 2]
                p95 = sorted(samples)[int(len(samples) * 0.95)]
                logger.info(
                    f"[Latency] n={len(samples)}  "
                    f"avg={avg:.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms"
                )
        else:
            samples.append(elapsed_ms)  # count toward warm-up

        return {"actions": actions}, cache

    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        return {"actions": np.zeros((CONFIG["action_horizon"], CONFIG["action_dim"]))}, cache


async def handle_connection(websocket, path=None):
    """Handle a WebSocket connection with per-connection ATTC cache."""
    logger.info(f"Connection from {websocket.remote_address} opened")

    # Each connection gets its own cache (supports concurrent multi-suite eval)
    vision_cache = None

    try:
        # Send server metadata on connection
        metadata = {
            "model": "SimVLA",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
            "use_cache": CONFIG["use_cache"],
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

                # Run inference (cache maintained across steps within connection)
                result, vision_cache = infer(request, cache=vision_cache)

                # Send response (convert numpy to list for compatibility)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()

                response_data = {"actions": actions}

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
                await websocket.send(f"Error: {str(e)}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if CONFIG["use_cache"] and vision_cache is not None:
            total = vision_cache.get("total_views", 0)
            hits  = vision_cache.get("cache_hits", 0)
            if total > 0:
                logger.info(
                    f"Connection closed — ATTC session stats: "
                    f"hit rate={hits/total:.1%} ({hits}/{total} views cached)"
                )
        # Print final latency summary for this connection
        samples = CONFIG["_latency_samples"]
        valid = samples[CONFIG["_latency_warmup"]:]
        if valid:
            avg = sum(valid) / len(valid)
            p50 = sorted(valid)[len(valid) // 2]
            p95 = sorted(valid)[int(len(valid) * 0.95)]
            logger.info(
                f"[Latency summary] n={len(valid)}  "
                f"avg={avg:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms"
            )
        logger.info(f"Connection from {websocket.remote_address} closed")


async def serve(host: str, port: int):
    """Start the WebSocket server."""
    logger.info(f"Creating SimVLA server (host: {host}, port: {port})")

    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SimVLA LIBERO Server (WebSocket)")
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
    # ATTC cache arguments
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="Enable Adaptive Temporal Token Cache (ATTC)")
    parser.add_argument("--cache_alpha", type=float, default=1.0,
                        help="ATTC threshold = EMA_mean + alpha*EMA_std (default: 1.0)")
    parser.add_argument("--cache_beta", type=float, default=0.9,
                        help="ATTC EMA decay factor (default: 0.9 ≈ 10-frame window)")
    parser.add_argument("--cache_warmup", type=int, default=3,
                        help="ATTC warmup steps before caching activates (default: 3)")

    args = parser.parse_args()

    CONFIG["use_cache"]      = args.use_cache
    CONFIG["cache_alpha"]    = args.cache_alpha
    CONFIG["cache_beta"]     = args.cache_beta
    CONFIG["cache_warmup"]   = args.cache_warmup

    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)

    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")

    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
