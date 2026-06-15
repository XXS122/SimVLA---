# TDS / Transition-Aware Training — Controlled-Study Run Plan

Operational SOP for the uniform-vs-TDS controlled study on a low training
budget, written for **two single-GPU (A100) machines running in parallel**.

---

## 0. Why this plan (the one thing to remember)

The existing 340k checkpoint saturates LIBERO at ~98% — at that budget there
is no headroom to measure any improvement, and that checkpoint additionally
had adaptive chunking ON, so it is **not** a clean baseline. We therefore
re-train from scratch at a **low budget (100k steps)** where success rate is
in the 70–85% range and differences are measurable. The controlled study
varies **one knob at a time** against an otherwise identical recipe.

> The single most important discipline: **every run shares the same batch
> size, iters, seed, learning rate, and 4-suite meta. Only the innovation
> flag changes.** Otherwise the comparison is not controlled.

---

## 1. The run matrix (4 runs, all identical except the flags)

| Run | adaptive chunking | TDS | Purpose |
|-----|:-:|:-:|---------|
| `exp_uniform` | off | off | **B0 control** — the baseline everything is compared to |
| `exp_tds`     | off | **on** | proves **TDS** (task-level sampling) helps |
| `exp_crw`     | **on** | off | proves **CRW + boundary head** (step-level) helps |
| `exp_all`     | **on** | **on** | combined headline number + **synergy** experiment (Q5) |

`exp_all` is required for the paper's "all methods" row and for the synergy
claim (TDS oversamples transitions → boundary head learns better → bigger BGE
gain), but it is **Phase 2** — only worth the compute once `exp_tds` shows a
positive effect over `exp_uniform`.

---

## 2. Two-machine parallel schedule

Because the TDS weight csv is computed offline from demonstrations (no trained
model needed), `exp_tds` does **not** have to wait for `exp_uniform`. Both
phases parallelize across the two machines:

```
Phase 1 (run in parallel):   Machine A = exp_uniform     Machine B = exp_tds
        ↓ (after both finish + evaluated)
Phase 2 (run in parallel):   Machine A = exp_crw         Machine B = exp_all
```

Experiment 0 (the difficulty-vs-SR hypothesis test) runs **independently**
once `exp_uniform` has produced an intermediate checkpoint — it does not block
any training run.

---

## 3. One-time prerequisites (do on BOTH machines)

```bash
# a) latest code (must include the NUM_WORKERS / ITERS overrides)
git pull origin claude/vibrant-meitner-gj8m4m

# b) edit machine-specific paths, then load them
#    (LIBERO_DATASETS, SIMVLA_SMOLVLM_MODEL, WANDB_API_KEY, ...)
source paths.env

# c) shared-memory check (Docker default is 64MB and crashes the DataLoader)
df -h /dev/shm
#   - if >= ~16G: fine, can use NUM_WORKERS=4 (faster)
#   - if 64M and you have privileges: mount -o remount,size=32g /dev/shm
#   - otherwise: always prefix runs with NUM_WORKERS=0 (no shm, single-thread)

# d) 4-suite training meta (MUST match eval + TDS stats: no libero_90)
python create_libero_meta.py --data_dir $LIBERO_DATASETS \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### Generate the TDS weights once (offline, model-free, minutes)

```bash
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --alpha 0 --beta 0.5 --gamma 0.5 --out task_difficulty_2comp.csv
```

Drop `E_events` (`--alpha 0`): the first hypothesis round showed it has no
discriminative power on LIBERO. Copy `task_difficulty_2comp.csv` to the
machine that runs `exp_tds` / `exp_all`.

---

## 4. The exact commands

Shared knobs: `NUM_WORKERS=0` (drop if shm is large), `ITERS=100000`,
`SAVE_INTERVAL=20000`, `batch=64`, `learning_coef=0.1`, seed default 0.
Use `WANDB_MODE=offline` if the network is flaky (sync later).

```bash
export WANDB_MODE=offline      # or set WANDB_API_KEY in paths.env to log live

# --- Phase 1 ---
# Machine A — B0 control
NUM_WORKERS=0 ITERS=100000 SAVE_INTERVAL=20000 \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_uniform

# Machine B — TDS only
TDS_WEIGHTS_CSV=./task_difficulty_2comp.csv \
NUM_WORKERS=0 ITERS=100000 SAVE_INTERVAL=20000 \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_tds

# --- Phase 2 (after Phase 1 evaluated and TDS shows a positive effect) ---
# Machine A — CRW + boundary head only
USE_ADAPTIVE_CHUNKING=true \
NUM_WORKERS=0 ITERS=100000 SAVE_INTERVAL=20000 \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_crw

# Machine B — combined (synergy)
USE_ADAPTIVE_CHUNKING=true TDS_WEIGHTS_CSV=./task_difficulty_2comp.csv \
NUM_WORKERS=0 ITERS=100000 SAVE_INTERVAL=20000 \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_all
```

Sanity checks in the first ~10 minutes of each run:
- log prints `[20/100000] loss=... (X.XXs/it)` → note `s/it`, 100k × s/it = ETA;
- TDS runs additionally print `Transition-Density Sampling ENABLED` and, after
  ~5000 samples, `[TDS] sampling check ... OK` (empirical vs target p_k). If you
  see a TDS WARNING, stop and check the csv ↔ meta task-name match.

---

## 5. Evaluation (after each run; per checkpoint for the efficiency curve)

For each saved checkpoint (20k/40k/60k/80k/100k), start the server then run the
single-GPU sequential evaluator:

```bash
# terminal 1 — inference server
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/exp_uniform/ckpt-100000 \
    --norm_stats ./norm_stats/libero_norm.json --port 8102

# terminal 2 — all 4 suites sequentially (one GPU), 20 trials each
cd evaluation/libero
bash run_eval_seq.sh 8102 20 u100k 0 --no_video
# -> u100k_sr_all.csv  (also per-suite txt logs)
```

---

## 6. Experiment 0 — hypothesis test (independent, anytime after a baseline ckpt)

Pick a `exp_uniform` checkpoint whose average SR lands in **70–85%** (evaluate
late→early to find it), then:

```bash
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --sr_csv evaluation/libero/u100k_sr_all.csv \
    --alpha 0 --beta 0.5 --gamma 0.5 --out task_difficulty_2comp.csv
```

Read the printout: PASS = Spearman ρ < −0.4 and p < 0.05 → the premise holds,
`density_vs_sr.png` goes into the paper. The 0b component table confirms the
2-component choice. (If still ceiling-bound, evaluate an earlier checkpoint.)

---

## 7. What to report (the deliverables)

1. **Efficiency curve** (headline figure): avg SR vs training step, two lines
   (uniform vs TDS). Win = TDS line left-shifted / higher (same SR, fewer steps).
2. **Main table**: per-suite + avg SR at the chosen budget, all 4 runs,
   mean±std. Largest TDS gain expected on **LIBERO-Long**.
3. **Synergy (Q5)**: boundary-head AUROC and BGE gain, `exp_crw` vs `exp_all`.
4. **Upper-bound row**: the existing 340k=98% result — both converge at large
   budget, so the methods do not hurt final performance.

> Do **not** judge by training `loss`: TDS deliberately changes the data
> distribution, so the losses are not directly comparable across runs. Only
> rollout success rate counts.
