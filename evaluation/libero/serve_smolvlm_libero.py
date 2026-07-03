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

Inference modes (Task-2):
  flow     multi-step integration (--flow_steps, default 10). Works for both
           flow-matching and MeanFlow checkpoints.
  onestep  single-step generation (MeanFlow checkpoints).
  bok      best-of-K: sample --num_samples one-step candidates, select the
           argmin-energy one with the ActionEnergyVerifier (--verifier).

Per-request latency is appended to --latency_log (JSONL) when given.
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
from models.verifier import ActionEnergyVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
verifier: Optional[ActionEnergyVerifier] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
    "mode": "flow",        # flow | onestep | bok
    "flow_steps": 10,      # integration steps for mode=flow
    "num_samples": 8,      # K for mode=bok
    "latency_log": None,   # JSONL path for per-request latency
}


def load_model(checkpoint_path: str, norm_stats_path: str = None, smolvlm_model_path: str = None,
               verifier_path: str = None):
    """Load SimVLA model, processor and (optionally) the energy verifier."""
    global model, processor, verifier

    logger.info(f"Loading SimVLA from {checkpoint_path}...")

    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    smolvlm_path = smolvlm_model_path or "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)

    if verifier_path:
        logger.info(f"Loading energy verifier from {verifier_path}...")
        verifier = ActionEnergyVerifier.from_pretrained(verifier_path)
        verifier = verifier.to(device)
        verifier.eval()
        logger.info(f"Verifier params: {verifier.num_parameters / 1e6:.1f}M")
    
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


def _log_latency(latency_ms: float):
    """Append per-request latency to the JSONL log (if configured)."""
    path = CONFIG.get("latency_log")
    if not path:
        return
    try:
        import json as _json
        with open(path, "a") as f:
            f.write(_json.dumps({
                "latency_ms": latency_ms,
                "mode": CONFIG["mode"],
                "num_samples": CONFIG["num_samples"],
                "flow_steps": CONFIG["flow_steps"],
            }) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write latency log: {e}")


def _generate(lang, images, image_mask, proprio_tensor):
    """Dispatch action generation according to CONFIG['mode']."""
    mode = CONFIG["mode"]

    if mode == "flow":
        actions = model.generate_actions(
            input_ids=lang['input_ids'],
            image_input=images,
            image_mask=image_mask,
            proprio=proprio_tensor,
            steps=CONFIG["flow_steps"],
        )
        return actions

    if mode == "onestep":
        actions = model.generate_actions(
            input_ids=lang['input_ids'],
            image_input=images,
            image_mask=image_mask,
            proprio=proprio_tensor,
            steps=1,
        )
        return actions

    if mode == "bok":
        if verifier is None:
            raise RuntimeError("mode=bok requires --verifier")
        # VLM runs once; candidates and verifier share the features.
        enc = model.forward_vlm_efficient(images, image_mask, lang['input_ids'])
        cands = model.sample_action_candidates(
            input_ids=None, image_input=None, image_mask=None,
            proprio=proprio_tensor,
            num_samples=CONFIG["num_samples"],
            steps=1,
            vlm_features=enc["vlm_features"],
        )
        if hasattr(model.action_space, 'normalize_state'):
            proprio_norm = model.action_space.normalize_state(proprio_tensor)
        else:
            proprio_norm = proprio_tensor
        energies = verifier.score_candidates(
            enc["vlm_features"], proprio_norm, cands["actions_norm"]
        )  # [B, K]
        best = energies.argmin(dim=1)  # [B]
        idx = best.view(-1, 1, 1, 1).expand(-1, 1, *cands["actions"].shape[2:])
        return cands["actions"].gather(1, idx).squeeze(1)

    raise ValueError(f"Unknown inference mode: {mode}")


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

        # Inference (timed)
        import time as _time
        t_start = _time.perf_counter()
        with torch.no_grad():
            actions = _generate(lang, images, image_mask, proprio_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (_time.perf_counter() - t_start) * 1000.0
        _log_latency(latency_ms)

        actions = actions.cpu().numpy()[0]

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
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM model path or HuggingFace repo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", type=str, default="flow",
                        choices=["flow", "onestep", "bok"],
                        help="Inference mode (see module docstring)")
    parser.add_argument("--flow_steps", type=int, default=10,
                        help="Integration steps for mode=flow")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="K candidates for mode=bok")
    parser.add_argument("--verifier", type=str, default=None,
                        help="ActionEnergyVerifier checkpoint dir (required for mode=bok)")
    parser.add_argument("--latency_log", type=str, default=None,
                        help="JSONL file to append per-request latency")

    args = parser.parse_args()

    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")

    CONFIG["mode"] = args.mode
    CONFIG["flow_steps"] = args.flow_steps
    CONFIG["num_samples"] = args.num_samples
    CONFIG["latency_log"] = args.latency_log
    if args.mode == "bok" and not args.verifier:
        parser.error("--mode bok requires --verifier")

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model,
               verifier_path=args.verifier)

    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    logger.info(f"  Mode: {CONFIG['mode']} (flow_steps={CONFIG['flow_steps']}, "
                f"K={CONFIG['num_samples']})")

    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
