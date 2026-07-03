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
    # Test-time scaling defaults (overridable per request via "tts/*" keys)
    "ode_steps": 10,
    "num_samples": 1,
    "selector": "consensus",
    "adaptive_threshold": None,
}

# Diagnostics logging (enabled via --diag_log)
DIAG_LOG_PATH: Optional[str] = None
_diag_call_idx = 0


def log_diagnostics(observation: Dict[str, Any], diag: Dict[str, Any], actions: np.ndarray):
    """Append one JSONL record of per-call test-time-scaling diagnostics."""
    global _diag_call_idx
    if DIAG_LOG_PATH is None:
        return
    import json
    import time as _time

    def _scalar(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return float(v.reshape(-1)[0].item())
        return float(v)

    record = {
        "call_idx": _diag_call_idx,
        "time": _time.time(),
        "prompt": observation.get("prompt", ""),
        "num_samples": diag["num_samples"],
        "num_active_samples": diag["num_active_samples"],
        "steps": diag["steps"],
        "selector": diag["selector"],
        "transformer_forwards": diag["transformer_forwards"],
        "selected_index": int(diag["selected_index"].reshape(-1)[0].item()),
        "uncertainty_v1": _scalar(diag.get("uncertainty_v1")),
        "uncertainty_x0hat": _scalar(diag.get("uncertainty_x0hat")),
        "uncertainty_x0": _scalar(diag.get("uncertainty_x0")),
        # Gripper command channel of the selected chunk (for event alignment)
        "gripper_cmd": [float(a) for a in actions[:, -1]],
    }
    # Optional client-side metadata for offline joins (task/episode/step)
    for key in ("meta/task_suite", "meta/task_id", "meta/episode", "meta/env_step"):
        if key in observation:
            v = observation[key]
            record[key.split("/", 1)[1]] = v.item() if hasattr(v, "item") else v

    _diag_call_idx += 1
    with open(DIAG_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_model(checkpoint_path: str, norm_stats_path: str = None, smolvlm_model_path: str = None):
    """Load SimVLA model and processor."""
    global model, processor
    
    logger.info(f"Loading SimVLA from {checkpoint_path}...")

    smolvlm_path = (
        smolvlm_model_path
        or os.environ.get("SIMVLA_SMOLVLM_MODEL")
        or "HuggingFaceTB/SmolVLM-500M-Instruct"
    )
    # Override the backbone path stored in the checkpoint config so a
    # checkpoint trained on another machine still loads locally
    model = SmolVLMVLA.from_pretrained(checkpoint_path, smolvlm_model_path=smolvlm_path)
    model = model.to(device)
    model.eval()

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
        
        # Test-time scaling parameters: server defaults + per-request overrides
        def _req(key, default, cast):
            v = observation.get(f"tts/{key}", default)
            if v is None:
                return None
            return cast(v.item() if hasattr(v, "item") else v)

        ode_steps = _req("ode_steps", CONFIG["ode_steps"], int)
        num_samples = _req("num_samples", CONFIG["num_samples"], int)
        selector = _req("selector", CONFIG["selector"], str)
        adaptive_threshold = _req("adaptive_threshold", CONFIG["adaptive_threshold"], float)

        # Inference
        with torch.no_grad():
            actions, diag = model.generate_actions_tts(
                input_ids=lang['input_ids'],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio_tensor,
                steps=ode_steps,
                num_samples=num_samples,
                selector=selector,
                adaptive_threshold=adaptive_threshold,
                return_diagnostics=True,
            )

        actions = actions.float().cpu().numpy()[0]
        log_diagnostics(observation, diag, actions)

        return {"actions": actions}
        
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
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SimVLA checkpoint")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to normalization stats JSON")
    parser.add_argument("--smolvlm_model", type=str,
                        default=os.environ.get("SIMVLA_SMOLVLM_MODEL",
                                               "HuggingFaceTB/SmolVLM-500M-Instruct"),
                        help="SmolVLM model path or HF repo (default: $SIMVLA_SMOLVLM_MODEL)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # Test-time scaling defaults (client can override per request via tts/* keys)
    parser.add_argument("--ode_steps", type=int, default=10,
                        help="Euler integration steps for flow matching")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Candidate action chunks sampled per state (best-of-N)")
    parser.add_argument("--selector", type=str, default="consensus",
                        choices=["consensus", "smoothness", "first"],
                        help="Training-free candidate selector")
    parser.add_argument("--adaptive_threshold", type=float, default=None,
                        help="Collapse to 1 candidate when uncertainty_v1 is below this")
    parser.add_argument("--diag_log", type=str, default=None,
                        help="Path to JSONL file for per-call TTS diagnostics")

    args = parser.parse_args()

    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")

    CONFIG["ode_steps"] = args.ode_steps
    CONFIG["num_samples"] = args.num_samples
    CONFIG["selector"] = args.selector
    CONFIG["adaptive_threshold"] = args.adaptive_threshold

    global DIAG_LOG_PATH
    DIAG_LOG_PATH = args.diag_log

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)

    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    logger.info(f"  TTS defaults: steps={CONFIG['ode_steps']}, N={CONFIG['num_samples']}, "
                f"selector={CONFIG['selector']}, adaptive={CONFIG['adaptive_threshold']}")
    if DIAG_LOG_PATH:
        logger.info(f"  Diagnostics log: {DIAG_LOG_PATH}")
    
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
