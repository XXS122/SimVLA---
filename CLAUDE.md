# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimVLA is a Vision-Language-Action (VLA) baseline for robotic manipulation, built on top of HuggingFace's SmolVLM-500M-Instruct vision-language model with a diffusion/flow-matching style action transformer head.

## Installation

```bash
conda create -n simvla python=3.11
conda activate simvla
pip install torch torchvision
pip install transformers>=4.57.0
pip install flash-attn==2.5.6 --no-build-isolation
pip install -r requirements.txt
```

## Key Commands

### Data Preparation (LIBERO)
```bash
# Create training metadata
python create_libero_meta.py \
  --data_dir ./datasets/metas \
  --subsets libero_10 libero_goal libero_object libero_spatial \
  --output ./datasets/metas/libero_train.json

# Compute normalization statistics
python compute_libero_norm_stats.py \
  --data_dir ./datasets/metas \
  --subsets libero_10 libero_goal libero_object libero_spatial \
  --output ./norm_stats/libero_norm.json
```

### Training
```bash
# Small model (768 hidden, 12 layers, 12 heads, 384x384 images, 4 GPUs)
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large model (1024 hidden, 24 layers, 16 heads)
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Direct training script invocation
python train_smolvlm.py --help
```

### Evaluation (LIBERO)
```bash
# Start inference server
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint YuankaiLuo/SimVLA-LIBERO \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8102

# Run evaluation
bash evaluation/libero/run_eval_all.sh [port] [num_episodes] [run_name] [seeds]
```

## Architecture

### Model Stack (`models/`)

**`modeling_smolvlm_vla.py`** — `SmolVLMVLA` (main model class)
- Wraps HuggingFace `SmolVLMForConditionalGeneration` as the vision-language backbone
- Attaches a `SmolVLMActionTransformer` action head
- Extracts VLM hidden states as conditioning for the action head
- `SmolVLMVLAConfig` controls both the VLM backbone and the action transformer dimensions

**`transformer_smolvlm.py`** — `SmolVLMActionTransformer`
- DiT-style (Diffusion Transformer) action decoder with timestep and conditioning embeddings
- Components: `TransformerBlock`, `DiTBlock`, `FinalLayer`, `Attention`, `Mlp`
- Used for flow-matching / diffusion-based action prediction

**`action_hub.py`** — Action space registry
- `BaseActionSpace` abstract class; subclasses define observation/action dimensions
- `LiberoJointActionSpace` handles the 7-DoF arm + gripper used in LIBERO
- Register custom action spaces via `ActionSpaceRegistry`

**`processing_smolvlm_vla.py`** — `SmolVLMVLAProcessor`
- Extends SmolVLM's tokenizer/processor with action/proprio normalization and denormalization
- Normalization stats are loaded from a JSON file (e.g., `norm_stats/libero_norm.json`)

**`configuration_smolvlm_vla.py`** — `SmolVLMVLAConfig`
- HuggingFace-compatible config; controls `action_hidden_size`, `action_depth`, `action_num_heads`, `action_dim`, `proprio_dim`

### Dataset Stack (`datasets/`)

**`dataset_smolvlm.py`** — `SmolVLMDataReader` / `SmolVLMDataReaderWithPadding`
- `IterableDataset` implementations that stream episodes from HDF5 files
- Multi-view image support (default 3 views); configurable image size (384×384 or 512×512)
- ImageNet normalization applied to images

**`domain_handler/`** — Plugin system for different dataset formats
- `base.py` defines the `BaseDomainHandler` interface
- `registry.py` provides handler lookup
- `libero_hdf5.py` implements loading from LIBERO HDF5 files

### Training (`train_smolvlm.py`)
- Uses HuggingFace `Accelerate` for multi-GPU distributed training
- Optional WandB integration
- Supports checkpoint resumption via `--resume_from_checkpoint`

## Data Flow

1. HDF5 episode files → `libero_hdf5.py` → raw observations/actions
2. Raw data → `SmolVLMDataReader` → tokenized inputs + normalized actions
3. Tokenized batch → `SmolVLMVLA.forward()` → VLM hidden states → action transformer → predicted actions
4. Actions denormalized by `SmolVLMVLAProcessor` before sending to the robot

## HuggingFace Compatibility

All model classes follow standard HuggingFace patterns (`PreTrainedModel`, `PretrainedConfig`, `ProcessorMixin`), so the model can be loaded with `from_pretrained` and pushed to the Hub with `push_to_hub`.
