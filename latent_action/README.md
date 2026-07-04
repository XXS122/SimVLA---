# Identifiable Continuous Latent-Action Flow Pretraining

Research pipeline built on SimVLA. Scientific claim under test: latent
actions extracted from action-free video by an inverse-dynamics encoder
with (i) variance/covariance whitening and (ii) shared ego-motion
invariance regularization are **identifiable up to an affine transform**
of the true action space — so a probe-initialized frozen affine adapter
suffices to transfer a flow-matching action expert pretrained purely on
latent actions to real robot control, with the gains concentrated in the
low-data regime.

## Setup

```bash
cp paths.env.example paths.env   # edit paths, never commit paths.env
source paths.env
```

Everything below writes into `$SIMVLA_CHECKPOINTS`:

```
$SIMVLA_CHECKPOINTS/
  metas/            libero_train.json, libero_train_p1.json, libero_train_p10.json
  norm_stats/       libero_norm.json
  lam/              lam_final.pt, lam_config.json
  z_labels/         z_labels.h5, z_labels_stress.h5, libero_train_z.json
  probe/            probe.npz (+ .json report), probe_stress.npz
  runs/             baseline_* / pretrain_flow / finetune_*
```

## Pipeline

### Stage 0 — metadata, norm stats, baseline control lines

```bash
./run_pipeline.sh meta                      # scan $LIBERO_DATASETS
./run_pipeline.sh norm-stats
./run_pipeline.sh splits                    # p1 (1%) and p10 (10%) demo-level splits

./run_pipeline.sh baseline --split p100     # 200k iters (default)
./run_pipeline.sh baseline --split p10      # 60k
./run_pipeline.sh baseline --split p1       # 20k
```

These three baselines are the "random init" line of the main figure
(success rate vs. #demos).

### Stage 1 — latent action model + go/no-go gate

```bash
./run_pipeline.sh train-lam                       # ~2-3 days on one A100
./run_pipeline.sh label                           # z labels + pretraining meta
./run_pipeline.sh probe                           # PASS iff mean R^2 >= 0.6
```

**Early health check (first hour of train-lam):** `recon` is normalized —
1.0 means the decoder predicts zero change and z is useless; it must drop
clearly below 1 (e.g. ≤0.7). If it hugs 1.0 after ~1h, kill the run and
escalate (larger `--stride`, larger `--lam_dim/--enc_depth`) instead of
waiting for the probe.

**Stop rule:** if `probe` reports mean R² < 0.6 after reasonable tuning of
`--var_coef/--cov_coef/--inv_coef`, the identifiability premise fails —
stop here and pivot.

Theory-predicted falsification test (ego-motion stress):

```bash
./run_pipeline.sh label --ego_aug                 # stress z labels
./run_pipeline.sh probe --stress                  # R^2 should stay high with
                                                  # inv_coef>0, collapse with 0
```

Discrete (VQ) ablation of the LAM — the same-scale stand-in for
LAPA/UniVLA-style quantized latents:

```bash
./run_pipeline.sh train-lam --use_vq --output_dir $SIMVLA_CHECKPOINTS/lam_vq
```

### Stage 2 — flow-expert pretraining + downstream fine-tuning

```bash
./run_pipeline.sh pretrain                        # action_mode=latent_z, 100k iters

# pretrained vs from-scratch, three data regimes (main figure)
./run_pipeline.sh finetune --split p1
./run_pipeline.sh finetune --split p1  --from_scratch
./run_pipeline.sh finetune --split p10
./run_pipeline.sh finetune --split p10 --from_scratch
./run_pipeline.sh finetune --split p100
./run_pipeline.sh finetune --split p100 --from_scratch
```

`finetune` runs `action_mode=libero_z_adapter`: the flow expert keeps
generating in z-space; a **frozen** affine adapter (initialized from
`probe.npz`) maps actions in/out. `--train_adapter` unfreezes it
(ablation). Seeds: append `--seed 1` etc. (forwarded to train_smolvlm.py).

### Evaluation

```bash
./run_pipeline.sh serve --ckpt $SIMVLA_CHECKPOINTS/runs/finetune_p1_pretrained/ckpt-20000
# then run the LIBERO client from evaluation/libero/ (see its README)
```

## Multi-GPU / resume / wandb

- `NUM_GPUS>1` in the environment switches training stages to
  `accelerate launch --num_processes $NUM_GPUS`.
- `SIMVLA_RESUME_CKPT=<ckpt dir>` resumes training stages from that
  checkpoint (`--models <dir> --resume` is added automatically).
- wandb logging activates whenever `WANDB_API_KEY` is non-empty; project
  name comes from `WANDB_PROJECT`.

## Paper artifact map

| Figure/Table | Command producing the numbers |
|---|---|
| Main fig: success vs #demos (喇叭口) | 6 × `finetune` runs + `baseline` runs, evaluated via `serve` |
| Probe R² table (per action dim) | `probe` → `probe/probe.json` |
| Ego-motion stress (theory falsification) | `label --ego_aug` + `probe --stress`, LAM trained with/without `--inv_coef 0` |
| Continuous vs discrete latents | `train-lam --use_vq` + relabel + re-probe + re-pretrain |
| Convergence speed | wandb curves of `finetune_*` runs (steps to reach fixed success) |
| Adapter ablation | `finetune --train_adapter` vs default frozen |
