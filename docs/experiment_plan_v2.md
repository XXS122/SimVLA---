# Experiment Plan v2 — Toward a Publishable TDS Paper

> Roadmap for the remaining experiments. Machines: **sapi** (`/datasets/...`,
> trained exp_uniform) and **yyk** (`/workspace/...`, trained exp_tds).
> Read with `docs/results_log.md` (live numbers) and `docs/tds_design.md` (method).

---

## 0. Where we are (done)

- **Training:** exp_uniform → 100k (sapi); exp_tds → 100k (yyk, extending to 200k).
- **Headline result (validated):** at 40k steps, same-machine on sapi,
  TDS **89.75** vs uniform **61.9** = **+27.9** avg. TDS@40k ≈ uniform@100k
  ⇒ ~2.4× sample efficiency.
- **Experiment 0** (difficulty predicts failure): significant at the
  non-saturated 40k checkpoint (Spearman ρ=−0.315, p<0.05).

---

## 1. The logic: two claims, two kinds of control

| Claim | Proven by | Status |
|---|---|---|
| **A.** Oversampling hard tasks helps | TDS vs **uniform** | ✅ done (+27.9) |
| **B.** *Free* data-difficulty matches *expensive* model-difficulty | TDS vs **measured-difficulty** | ❌ **TODO — critical** |
| **C.** The gain comes from the *direction* of the signal, not any perturbation | TDS vs **inverted** weights | ❌ TODO (cheap) |
| **D.** The uniform floor prevents easy-task forgetting | TDS vs **no-clip** | ❌ TODO (cheap) |

Claim **B** is the one reviewers will hammer ("why not just use loss/SR to
find hard tasks?"). Without it the novelty is exposed. It is also cheap —
all of B/C/D are the **same training pipeline with a different weights csv**.

---

## 2. Priority matrix

| # | Experiment | Proves | Paper artifact | Cost |
|---|---|---|---|---|
| 1 | **Efficiency curve** (eval both runs @ 20/40/60/80/100k) | A, quantifies speedup | Fig. 1 (headline) | eval only (~7 more evals) |
| 2 | **Measured-difficulty baseline** | **B** (the novelty defense) | Main table row | 1× 40k train + eval |
| 3 | **Inverted-weights** ablation | C | Ablation table | 1× 40k train + eval |
| 4 | **No-clip** ablation | D | Ablation table | 1× 40k train + eval |
| 5 | Curriculum (easy→hard) *(optional)* | alt. scheduling | Ablation table | 1× train + eval |
| 6 | CRW / CRW+TDS synergy *(optional)* | 3-timescale framework | Framework table | 2× train + eval |
| 7 | CALVIN second benchmark *(optional)* | generalization | Second main table | large |

**Do 1–4 first.** 5–7 are enhancements (5 is cheap; 6 extends the story; 7 is
only for a top VLA venue with spare time — LIBERO-only is publishable per the
DoReMi precedent).

---

## 3. Protocols (exact commands)

> All ablation/baseline training runs only need **40k** steps (the headline
> comparison point) — do **not** waste time on 100k for these. Keep every
> setting identical to exp_uniform / exp_tds; only the weights csv changes.
> Shared knobs: `batch=64`, `lr_coef=0.1`, `seed=0`, `NUM_WORKERS=0` (or 4 if
> `/dev/shm` enlarged), `WANDB_MODE=offline`.

### Exp 1 — Efficiency curve (eval only)

Evaluate every checkpoint of both runs. We already have: uniform 40k/100k,
TDS 40k (and TDS 100k in progress). Remaining: uniform 20k/60k/80k, TDS
20k/60k/80k(/100k).

```bash
# generic eval of one checkpoint (run on the machine that has it; for yyk
# checkpoints copied to sapi, sed the backbone path first — see §4)
cd <repo>
pkill -f serve_smolvlm
CUDA_VISIBLE_DEVICES=<gpu> python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/<exp>/ckpt-<N> \
    --norm_stats ./norm_stats/libero_norm.json --port <port> &
# wait for "listening", then in tmux:
cd evaluation/libero && bash run_eval_seq.sh <port> 20 <prefix> <gpu> --no_video
# then rebuild the merged csv (auto-merge is unreliable):
echo "task_name,sr" > <prefix>_sr_all.csv
grep -hv "^task_name" <prefix>_per_task_*.csv | grep -v "^$" >> <prefix>_sr_all.csv
```

**Deliverable:** plot avg SR vs step for both runs. TDS line above/left of
uniform = sample-efficiency win. Report DoReMi-style: "TDS reaches uniform's
100k success at ~Xk steps = Y× fewer steps."

### Exp 2 — Measured-difficulty baseline (the key comparison)

Build sampling weights from the uniform model's measured per-task SR (the
"ask the model" approach), then train identically and compare to TDS.

```bash
# 1) make weights from uniform's per-task SR (model-coupled difficulty)
python make_empirical_weights.py \
    --sr_csv evaluation/libero/u40k_sr_all.csv \
    --temperature 2.0 --clip_rho 0.5 \
    --out task_difficulty_empirical.csv

# 2) train EXACTLY like TDS, only the csv differs (40k)
TDS_WEIGHTS_CSV=./task_difficulty_empirical.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_empirical

# 3) eval @ 40k, compare to TDS@40k and uniform@40k
```

**Expected / how to read:**
- TDS ≈ exp_empirical ⇒ **free difficulty matches expensive difficulty** →
  the novelty holds (we save a full eval loop + a baseline training run).
- Note in the paper: empirical difficulty needs train→eval→reweight→retrain
  (~2× compute + a full eval); TDS needs minutes on CPU.

### Exp 3 — Inverted-weights ablation (direction control)

Oversample the EASY tasks instead. If gains were from generic non-uniformity,
this would also help; it should instead hurt (esp. Goal/Long).

```bash
# inverted difficulty weights from the SAME transition stats (--invert flips
# the direction so EASY tasks are oversampled)
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --alpha 0 --beta 0.5 --gamma 0.5 --invert \
    --out task_difficulty_inverted.csv

TDS_WEIGHTS_CSV=./task_difficulty_inverted.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_inverted
```

**Expected:** Long/Goal drop vs uniform → direction of the signal carries the
information (it is not regularization noise).

### Exp 4 — No-clip ablation (floor control)

```bash
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --alpha 0 --beta 0.5 --gamma 0.5 --clip_rho 0 \
    --out task_difficulty_noclip.csv

TDS_WEIGHTS_CSV=./task_difficulty_noclip.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_noclip
```

**Expected:** Spatial/Object drop (easy tasks starved) while TDS-with-clip
holds → the uniform floor controls the failure mode.

### Exp 5 — Curriculum *(optional)*

Anneal uniform→TDS across training stages (use `tds_sampling.mix_with_uniform`
to make intermediate csvs). Expected: no better than static TDS → keep static.

### Exp 6 — CRW / synergy *(optional, the 3-timescale framework)*

```bash
# CRW + boundary head only (step level)
USE_ADAPTIVE_CHUNKING=true \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_crw

# CRW + TDS (synergy: does TDS make the boundary head better?)
USE_ADAPTIVE_CHUNKING=true TDS_WEIGHTS_CSV=./task_difficulty_2comp.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_all
```

Compare boundary-head AUROC of exp_crw vs exp_all; report the synergy.

### Exp 7 — CALVIN *(optional, top-venue enhancement)*

Big lift (new data format + eval harness). Only if targeting a top VLA venue
with spare time. Otherwise list as future work; LIBERO-only is defensible.

---

## 4. Ops cheat-sheet (the traps we already hit)

- **Copying a yyk checkpoint to sapi:** its `config.json` hard-codes the yyk
  backbone path. Fix before loading:
  ```bash
  sed -i 's|/workspace/hyj/simvla/model/smolvlm-500M|/datasets/models/smolvlm/SmolVLM-500M-Instruct|g' \
      runs/<exp>/ckpt-<N>/config.json
  ```
  (Or once: `ln -s /datasets/models/smolvlm/SmolVLM-500M-Instruct /workspace/hyj/simvla/model/smolvlm-500M` on sapi.)
- **Docker `/dev/shm` too small → DataLoader Bus error:** use `NUM_WORKERS=0`
  (or `mount -o remount,size=32g /dev/shm` then `NUM_WORKERS=4`).
- **Eval "秒退" / `(no per-task csvs)`:** the server isn't up on that port, or
  conda env not active. Start the server, wait for `listening`, then eval.
- **Run evals in tmux**; a stray Ctrl+C kills the client (server survives).
- **Always 20 trials/task** for every run (consistency with existing numbers).
- **Single usable GPU per machine ⇒ cannot train and eval at once** (they
  contend). Serialize, or train on one machine while the other evaluates.
- **Always rebuild `<prefix>_sr_all.csv` manually** (auto-merge is buggy).

---

## 5. Paper tables to fill

**Table 1 — Main comparison @ 40k (sapi, same machine):**

| Method @ 40k | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| uniform | 73.0 | 95.5 | 38.5 | 40.5 | 61.9 |
| measured-difficulty (Exp 2) | — | — | — | — | — |
| **TDS (ours)** | 97.5 | 98.5 | 92.0 | 71.0 | **89.75** |

**Table 2 — Ablations @ 40k:** TDS / inverted / no-clip / (curriculum).

**Fig. 1 — Efficiency curve:** avg SR vs step, uniform vs TDS (Exp 1).

**Fig. 2 — Exp 0 scatter:** difficulty d_k vs per-task SR (ρ=−0.32).

**(optional) Table 3 — Framework:** uniform / +CRW / +TDS / +both (Exp 6).

---

## 6. Why this design (research-backed)

- **DoReMi** (data-mixture reweighting, NeurIPS'23): single dataset (The Pile)
  + deep analysis was enough; headline metric "2.6× fewer steps to baseline";
  baseline = uniform/default weights; uses a *proxy model* (TDS is cheaper).
  Also: reweighting medium-difficulty domains gives *positive transfer that
  lifts all domains* — this explains our Spatial +24.5.
- **Prioritized replay / OHEM / Re-Mix:** difficulty from the *model*
  (TD-error / loss / reference model) — the family our measured-difficulty
  baseline represents.
- **Curriculum/RFCL/PLR:** standard baseline = the non-curriculum (uniform)
  version; standard metric = efficiency curve (SR vs steps).
- **VLA benchmarking (2026):** LIBERO is mandatory; CALVIN increasingly
  expected for *model* papers, but method papers can be LIBERO-only with
  strong baselines + ablations.

References: DoReMi (arXiv:2305.10429); OpenVLA-OFT (arXiv:2502.19645);
Reverse-Forward Curriculum (ICLR'24); LIBERO (NeurIPS'23).
