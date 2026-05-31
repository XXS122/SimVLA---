# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SimVLA is a Vision-Language-Action (VLA) policy for robot manipulation. It pairs a
**SmolVLM-500M-Instruct** vision-language backbone with a **flow-matching action
transformer** head. The only fully wired-up dataset/benchmark is **LIBERO**.

## Environment

Two separate conda environments are used (they are incompatible — different PyTorch/CUDA):

- `simvla` (python 3.10): training, the norm-stats/meta scripts, and the policy server.
  Requires `transformers>=4.57.0` and `flash-attn==2.5.6`. See `readme.md` for the full
  `pip install` line (`requirements.txt` is intentionally minimal and not sufficient on its own).
- `libero` (python 3.8.13): only for running evaluation rollouts. Needs the
  [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) installed plus
  `openpi_client`. See `evaluation/libero/README.md`.

There is no test suite, linter config, or packaging (`setup.py`/`pyproject.toml`) in this repo.

## Common Commands

End-to-end LIBERO workflow (run from repo root in the `simvla` env). The two `.sh` scripts
auto-run steps 1–2 if their output files are missing, so usually you just run the script.

```bash
# 1. Build training metadata (scans HDF5 files -> JSON)
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json

# 2. Compute action/state normalization stats
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json

# 3. Train (multi-GPU via accelerate). Args: BATCH_SIZE LEARNING_COEF OUTPUT_DIR [RESUME_CKPT]
bash train_smolvlm_small.sh        # hidden=768,  depth=12, heads=12
bash train_smolvlm_large.sh        # hidden=1024, depth=24, heads=16

# Train directly (single config) — see train_smolvlm.py for all flags
accelerate launch --num_processes=4 --mixed_precision bf16 train_smolvlm.py \
    --train_metas_path ./datasets/metas/libero_train.json \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --action_mode libero_joint --num_actions 10 --output_dir ./runs/exp
```

Serving + evaluation:

```bash
# Start policy server (simvla env). --checkpoint is an HF repo or a local ckpt-XXXX dir.
cd evaluation/libero
python serve_smolvlm_libero.py --checkpoint <repo_or_ckpt_dir> \
    --norm_stats ../../norm_stats/libero_norm.json --port 8102

# Run rollouts (libero env, separate terminal). Args: PORT NUM_TRIALS OUT_PREFIX "GPUs"
bash run_eval_all.sh 8102 50 "eval_simvla" "0 1 2 3"
```

## Architecture

### Model (`models/`)

`SmolVLMVLA` (`modeling_smolvlm_vla.py`) is a HuggingFace `PreTrainedModel` composed of three parts:

1. **VLM backbone** (`self.vlm`): SmolVLM-500M-Instruct, loaded in float32. Frozen for the
   first `freeze_steps`, then trained at `learning_rate * learning_coef`.
2. **Action head** (`self.transformer`): `SmolVLMActionTransformer` (`transformer_smolvlm.py`),
   a flow-matching velocity predictor.
3. **Action space** (`self.action_space`): handles normalization + loss, built from a registry
   (`action_hub.py`).

**Flow matching** is the core training objective (`forward`): sample `t ~ Beta(1.5,1)`, interpolate
`x_t = t·noise + (1-t)·action`, and regress the model output toward the velocity `u_t = noise - action`
with MSE. Inference (`generate_actions`) is Euler integration from `t=1` to `t=0`, then
`action_space.postprocess` un-normalizes.

**VLM feature extraction has two methods — know which one runs.** Training and inference both call
`forward_vlm_efficient`, which manually runs `vision_model → connector → concat with text embeddings →
text_model` (the Idefics3 LM) to get *fused* vision-language features. The other method, `forward_vlm`,
uses the SmolVLM chat template and is **not** on the train/inference path.

**Two action-transformer modes** (toggle with `--use_adaln`, baked into the checkpoint and not
changeable on resume):
- **Concat mode** (default, `use_adaln=False`): projected VLM features are concatenated to the action
  token sequence; action tokens carry `[action, proprio, time]`.
- **AdaLN/DiT mode** (`use_adaln=True`): time + mean-pooled VLM + proprio are fused into one condition
  vector injected via Adaptive LayerNorm (`DiTBlock`/`FinalLayer`).

### Training loop (`train_smolvlm.py`)

- Uses 🤗 `accelerate` (bf16, DDP with `find_unused_parameters=True` because the VLM is partially
  frozen early).
- Optimizer (`build_optimizer`) has **three param groups** — `vlm`, `transformer_core`, `action_heads` —
  with independent LR schedules (`update_group_lrs`): action heads train from step 0; VLM + core are
  frozen (lr=0) until `freeze_steps`. Cosine decay is off by default.
- Checkpoints are written to `OUTPUT_DIR/ckpt-{step}/` containing `model.safetensors` + `state.json`
  (`state.json` holds `global_step` for `--resume`).

### Data pipeline (`datasets/`)

- `create_smolvlm_dataloader` → `SmolVLMDataReader`, an **infinite `IterableDataset`** (re-iterates
  forever in training; the training loop stops by step count, not epoch).
- Per-dataset decoding is delegated to **domain handlers** looked up by name in
  `domain_handler/registry.py`. The handler is selected by the meta file's `dataset_name`
  (`"libero_hdf5"` for LIBERO).
- `LiberoHDF5Handler` reads LIBERO HDF5 directly. Important transforms it applies:
  proprio orientation is converted **euler → axis-angle**, and both camera images are **rotated 180°**.
- Sampling across multiple datasets is weighted by `domain_config.DATA_WEIGHTS`.

### Action spaces (`models/action_hub.py`)

Registry pattern via `@register_action(name)` + `build_action_space(name)`. **Only `libero_joint` is
registered** — 7-dim action `[Δxyz, Δeuler, gripper]`, 8-dim proprio `[ee_pos, axis_angle, gripper(2)]`.
Z-score normalization by default; quantile (`q01/q99`) normalization is available but off.

### Processor (`models/processing_smolvlm_vla.py`)

`SmolVLMVLAProcessor` does *not* subclass `ProcessorMixin` (deliberately, to avoid tokenizer type
checks). `encode_image` is a fast torch-based path; `encode_image_legacy` uses the HF image processor
and is kept only for compatibility checks.

### Serving (`evaluation/libero/`)

Two independent inference interfaces exist:
- `serve_smolvlm_libero.py` — a **WebSocket** server (msgpack_numpy) used for LIBERO eval. This is the
  one the eval scripts talk to.
- `SmolVLMVLA.run()` — a built-in **FastAPI `/act`** endpoint (separate, JSON-based), not used by the
  LIBERO eval flow.

## Gotchas

- **`action_mode` default is `galaxea_joint`** in `train_smolvlm.py` and the config, but that space is
  **not registered** — always pass `--action_mode libero_joint` (the `.sh` scripts already do).
- **Image size is 384**, not 512. Many docstrings/comments say "512x512 (SmolVLM requirement)" but every
  default and the training scripts use `image_size=384`. Trust the code, not the comments.
- `image_size`, `use_adaln`, and the action-transformer dims are fixed at checkpoint creation; resuming
  warns and ignores attempts to change `use_adaln`.
- Both LIBERO camera views are rotated 180° during data loading (`libero_hdf5.py`) **and** the eval
  client rotates images 180° — keep these consistent if you touch one.

## Git

Develop on branch `claude/gallant-wright-bV8c5`. Do not push to other branches without explicit
permission, and do not open PRs unless asked.
