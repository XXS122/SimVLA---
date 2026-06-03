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

### Local Machine Configuration (`paths.env`)

Machine-specific paths, GPU config, and secrets live in a **git-ignored** `paths.env`
(copy from `paths.env.example`). `source paths.env` before training/eval; the scripts
read these with fallbacks to the previous hard-coded defaults:

```bash
cp paths.env.example paths.env   # then fill in real values
source paths.env
```

| Variable | Used for |
|----------|----------|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM backbone path (train scripts, serve, `train_smolvlm.py` default) |
| `LIBERO_DATASETS` | LIBERO data root (`--data_dir` for metadata/norm-stats) |
| `SIMVLA_CHECKPOINTS` | Default training output dir / serve `--checkpoint` |
| `SIMVLA_RESUME_CKPT` | Default resume checkpoint |
| `CUDA_DEVICES` / `NUM_GPUS` | GPU ids + `accelerate --num_processes` (single-GPU capable) |
| `WANDB_API_KEY` / `WANDB_PROJECT` | WandB tracking |

`paths.env` is in `.gitignore` (it holds `WANDB_API_KEY`) — never commit it.

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
source paths.env   # loads paths, GPU config, WANDB_API_KEY

# Small model (768 hidden, 12 layers, 12 heads, 384x384 images)
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large model (1024 hidden, 24 layers, 16 heads)
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Enable Adaptive Action Chunking (change-rate-weighted loss + boundary head)
USE_ADAPTIVE_CHUNKING=true CHUNK_LOSS_WEIGHT=0.1 bash train_smolvlm_small.sh

# Direct training script invocation
python train_smolvlm.py --help
```

Training logs are saved for later review:
- Full console output → `OUTPUT_DIR/train_console_<timestamp>.log` (tee'd by the shell script)
- Structured logger → `OUTPUT_DIR/train_smolvlm_<timestamp>.log` (per-run, not clobbered)
- WandB: auto-enabled when `WANDB_API_KEY` is set; the log prints whether it is active

### 评估（LIBERO）

#### 第一步 — 启动推理服务器（单独开一个终端）

```bash
source paths.env

# checkpoint 默认读取 $SIMVLA_CHECKPOINTS，也可以手动指定
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8102
```

等到日志出现 `SimVLA server listening on 0.0.0.0:8102` 后再运行评估。

#### 第二步 — 运行评估（另开一个终端）

**全量评估（4 个任务集并行，需要 4 块 GPU）：**
```bash
cd evaluation/libero
bash run_eval_all.sh <端口> <每任务回合数> <结果前缀> "<gpu0> <gpu1> <gpu2> <gpu3>"

# 示例：
bash run_eval_all.sh 8102 10 eval_step200k "4 5 6 7"
```

**单独评估某个任务集（例如只评 libero_goal，1 块 GPU）：**
```bash
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 \
  --port 8102 \
  --client_type websocket \
  --task_suite libero_goal \
  --num_trials 20 \
  --video_out ./eval_videos
```

#### 第三步 — 查看结果

```bash
# 查看各任务集成功率
grep -E "Success Rate|Average" eval_step200k_*.txt

# 实时监控评估进度
tail -f eval_step200k_goal.txt
```

结果文件保存在 `eval_simvla_<端口>/` 目录下，文件名格式：
`<结果前缀>_spatial.txt` / `_object.txt` / `_goal.txt` / `_10.txt`

#### 注意事项
- `--num_trials 10` 速度快但噪声大；正式对比实验建议用 `20–50`
- 评估期间推理服务器必须保持运行
- `run_eval_all.sh` 4 个任务集并行运行，每个任务集占用一块 GPU

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
- Optional `ChunkBoundaryHead` (when `use_adaptive_chunking=True`): predicts a per-step
  boundary score from action features; `forward(..., return_boundary=True)` returns
  `(velocity, boundary_logits)`

**Adaptive Action Chunking** (opt-in training innovation, off by default)
- Config flags: `use_adaptive_chunking`, `chunk_loss_weight` (also `--use_adaptive_chunking`
  / `--chunk_loss_weight` CLI; `USE_ADAPTIVE_CHUNKING` / `CHUNK_LOSS_WEIGHT` shell env)
- In `SmolVLMVLA.forward` the flow-matching loss is weighted per-step by the ground-truth
  action change rate (`||a[t+1]-a[t]||` → weights in `[0.5, 1.5]`), concentrating learning
  on contact / direction-reversal moments; the boundary head regresses the (detached)
  normalized change rate as an auxiliary loss
- Setting the flag off recovers the exact original uniform-MSE objective (backward compatible)
- Design notes / experiments: `docs/method_section_draft.md`, `docs/experiment_design.md`

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
