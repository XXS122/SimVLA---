"""
latent_action: Identifiable continuous latent-action flow pretraining for SimVLA.

Pipeline stages (driven by ``python -m latent_action.run <stage>``):

  meta        build LIBERO training metadata from $LIBERO_DATASETS
  norm-stats  compute action/state normalization statistics
  splits      build low-data splits (1% / 10%) at demo level
  baseline    train the standard SimVLA baseline (libero_joint) on a split
  train-lam   train the latent action model (inverse dynamics + forward decoder)
  label       write per-frame latent actions z for every demo (z-labels)
  probe       affine probe z <-> action, report R^2, save adapter init
  pretrain    flow-matching pretraining of the action expert on z-chunks
  finetune    downstream fine-tune with affine adapter (from pretrain or scratch)
  serve       launch the LIBERO evaluation policy server

All stages read paths from environment variables (see paths.env.example):
  SIMVLA_SMOLVLM_MODEL, LIBERO_DATASETS, SIMVLA_CHECKPOINTS,
  SIMVLA_RESUME_CKPT, WANDB_API_KEY, WANDB_PROJECT,
  CUDA_DEVICES, NUM_GPUS, SIMVLA_Z_DIM
"""
