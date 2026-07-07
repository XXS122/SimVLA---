# Progress Log — Identifiable Continuous Latent-Action Flow Pretraining

Living document for Innovation 4 (see `README.md` for the pipeline recipe
and the paper artifact map). Updated after every experiment. Newest
status always at the top; full chronological log below for detail/audit.

---

## Current status (last updated: after rotation-fix result — decision point reached)

**The rotation-composition bug is REFUTED as the explanation for the real
data's low R².** Re-ran the probe on the *same* v2 z-labels with the
SO(3)-composition fix; numbers are essentially unchanged from before the
fix (see § 11 below for the full comparison). The bug was real and worth
having fixed (confirmed on synthetic data to single-handedly turn a
perfect affine relationship into R²≈0), but it is not what's holding
back this dataset.

**Free calculation (no rerun needed):** restricting the nonlinear probe
to just the 5 dims with an obvious visual footprint (dx, dy, dz, dyaw,
gripper — excluding roll/pitch) gives mean R² ≈ **0.342**. Still far
below the 0.6 gate. This rules out "it's fine except for 2 unobservable
dims" as a full explanation — every dimension, including the visually
obvious ones, is only partially decodable.

**This is now a genuine decision point, not a bug hunt.** Two structural
hypotheses (z-collapse/copy-shortcut; rotation-composition math) have
been found, fixed, and confirmed non-explanatory for the remaining gap.
Three ways forward were put to the user (see § 11): scale up the LAM
(cheap, uncertain payoff), add CLAM-style few-shot real-action grounding
(likely bigger payoff, shifts the paper's core "fully action-free"
claim), or pivot to Innovation 5. **Awaiting the user's choice.**

---

## Environment / hardware context

- Shared server, 8x NVIDIA A800-SXM4-80GB, training pinned to physical
  GPU 6 via `CUDA_DEVICES=6`.
- Backbone: local SmolVLM-500M at `$SIMVLA_SMOLVLM_MODEL`.
- Dataset: LIBERO (4 subsets: libero_10/goal/object/spatial), ~2000 demos,
  ~336k labeled frames after windowing.
- Container has a small `/dev/shm` (Docker default 64MB) — relevant to
  the DataLoader bus-error saga below.
- `reserve_gpu.py` (memory-only holder, no compute) used to keep other
  users off GPU 6 during runs — currently holding ~43.7GB, leaving ~10GB
  free (sized for LAM-stage peak usage, **must be re-sized up** before
  the `pretrain`/`finetune` stages, which run the full VLM and will need
  ~70GB like the original full-model training did).

---

## Chronological log

### 1. Pipeline scaffolding (commit `a45fa99`)

Built the full `latent_action/` package driven by
`python -m latent_action.run <stage>` / `./run_pipeline.sh <stage>`, all
paths from environment variables (`paths.env`):

- `make_splits.py` — demo-level low-data splits (p1/p10)
- `models.py` — `LatentActionModel` (inverse-dynamics encoder + forward
  decoder on frozen SigLIP features), VICReg variance/covariance
  regularizer, shared ego-motion-invariance augmentation, optional VQ
  bottleneck (discrete ablation)
- `train_lam.py` — LAM training loop
- `label_z.py` — per-frame z labeling (+ `--ego_aug` stress labels)
- `probe.py` — ridge probe z↔a_norm, per-dim R² report, go/no-go gate,
  adapter init (`probe.npz`)
- `run.py` — unified stage dispatcher
- Integrated into existing training stack: `LatentZActionSpace`,
  `LiberoZAdapterActionSpace` (`models/action_hub.py`), `libero_z` domain
  handler, demo-whitelist filter in `libero_hdf5.py`, new
  `train_smolvlm.py` flags (`--z_dim`, `--probe_path`, `--train_adapter`,
  `--pretrained_flow_ckpt`, `--freeze_vlm`, `--run_name`, env-var backbone
  default).

Tested end-to-end on CPU with synthetic data and a tiny local Idefics3
(probe recovers a planted affine map R²=1.0; adapter round-trips;
z-dataloader yields `[B,10,z]`; pretrain→warm-start→finetune→
generate_actions all pass).

Push required a one-off PAT (GitHub App integration on this repo is
read-only); token was used once for the URL credential and never
written to disk, then flagged for the user to revoke.

### 2. First training attempt: DataLoader bus error

```
./run_pipeline.sh train-lam
```
→ `ERROR: Unexpected bus error encountered in worker` (SIGBUS), repeated
across all 8 workers.

**First fix attempt (commit `9cb6158`):** switched
`torch.multiprocessing` to the `file_system` sharing strategy (default
tensor IPC uses `/dev/shm`, which Docker caps at 64MB). **Did not fix
it** — same error on rerun.

### 3. Root cause found: IPC payload size, not just the transport

Actual cause: workers were shipping *preprocessed* 384×384 float32
tensors through IPC (~7MB/sample, ~672MB per batch of 96) — far too much
for any small-`/dev/shm` transport, `file_system` strategy included.

**Fix (commit `bc6cdc3`):** workers now ship raw uint8 frames at native
LIBERO resolution (~100KB/sample, ~75x smaller); resize + ImageNet
normalization moved to GPU (`latent_action.data.gpu_preprocess`, same ops
and order as the existing CPU pipeline, verified numerically identical
in tests). Applied to both `train_lam.py` and `label_z.py`.

```
git pull
./run_pipeline.sh train-lam
```
→ ran without crashing.

### 4. GPU memory / throughput discussion

User noted only ~10GB used vs. ~70GB for full-model training at batch 64.
Explained: LAM (26.5M params) runs the SigLIP backbone under `no_grad`
(activations freed immediately, no optimizer state, no gradients for
~500M frozen params) — low memory here is expected and is *not* comparable
to the `pretrain`/`finetune` stages, which will run the full VLM and
return to ~70GB.

Found and fixed a real bug while discussing this: `run.py`'s own
`--iters` flag was consumed by the dispatcher and silently never
forwarded to `train_lam.py` (**commit `df5185d`**).

Recommended (and the user ran) a larger-batch config to use the spare
memory productively:
```
./run_pipeline.sh train-lam \
    --batch_size 320 --learning_rate 2e-4 --iters 20000 --num_workers 16
```

### 5. GPU reservation script (commit `b04c615`)

User wanted to occupy spare memory so other users on the shared server
can't schedule work on GPU 6 while training runs. Wrote `reserve_gpu.py`:
holds memory only (`torch.empty` blocks, no compute kernels — does not
steal SM cycles from the real training job); another process fails to
allocate at launch when free memory is near zero. `--leave-free` sizes
the headroom to the training job's peak (must cover the ~1.7x spike
during LAM's ego-augmentation steps). Ended up holding ~43.7GB with
`--leave-free 10`, confirmed via `nvidia-smi` (GPU 6: 70915MiB/81920MiB
used, 100% util from the actual training process).

### 6. LAM v1 result: probe FAILS, R²=0.03 (z-collapse)

Training (`--batch_size 320 --iters 20000`, default `--stride 1`)
completed: `recon` plateaued at ~1.47 (uninterpretable absolute scale at
the time), `var`→0.0022 (healthy, no dimension collapse), `cov`→0.073.

```
./run_pipeline.sh label
./run_pipeline.sh probe
```
```
=== affine probe z -> a_norm (val R^2) ===
  dx 0.0467  dy 0.0162  dz 0.0580  droll -0.0007  dpitch 0.0144
  dyaw 0.0177  gripper 0.0635   mean 0.0308
(reverse a->z mean R^2: 0.0100)
>>> gate (>=0.6): FAIL
```

**Diagnosis:** at 10Hz/128px, one step of arm motion is nearly invisible
in the coarse (4x-downsampled) connector features; the forward decoder
could just copy `f_t` and reconstruct `f_{t+1}` well without using `z` at
all, so `z` received no real gradient pressure — the variance regularizer
then filled it with whatever *does* have variance across the batch (scene
identity / task context), not motion.

### 7. LAM v2 fix: predict feature deltas, stride 4 (commit `ed75b43`)

Three changes targeting the three links in the diagnosis:
1. Decoder predicts `f_{t+k} - f_t` instead of `f_{t+k}` directly — kills
   the copy shortcut. `recon` is now normalized by delta energy, giving
   it an absolute, interpretable scale: **1.0 = z is useless** (zero-change
   predictor); healthy training must drop clearly below 1.
2. Default pair stride 1 → 4 (0.4s of motion — visible in coarse features).
3. `label_z.py` auto-detects stride from the LAM checkpoint; `probe.py`
   regresses z against the window-mean action.

Retrained with the same command (`--batch_size 320 --iters 20000`,
now `--stride 4` by default). Health check at ~4000/20000 steps:
`recon` 0.72–0.86 (well below 1.0 — real signal), `var` 0.00–0.13 (no
collapse), `inv` non-zero on ~half the logged steps as expected
(confirmed the augmentation branch does fire; earlier v1 summary showing
`inv=0` was just an unlucky non-augmented step, not a bug).

### 8. LAM v2 result: probe still FAILS on affine (R²=0.04), but recon proves real signal exists

```
./run_pipeline.sh label     # auto-detected stride=4
./run_pipeline.sh probe
```
```
=== affine probe z -> a_norm (val R^2) ===
  dx 0.0241  dy 0.0365  dz 0.0655  droll 0.0116  dpitch 0.0078
  dyaw 0.0328  gripper 0.0996   mean 0.0397
(reverse a->z mean R^2: 0.0116)
>>> gate (>=0.6): FAIL
```

This is the key tension that drove the next two steps: `recon` improved
a lot (proving `z` explains real feature-space delta variance) but the
*affine* probe didn't move at all — consistent with either (a) real
signal that's just non-affine, or (b) `z` explaining delta variance that
isn't the robot's own action (object motion, contact dynamics).

### 9. Nonlinear diagnostic probe added (commit `70a1e4e`)

Added `probe.py --nonlinear`: fits a small MLP z→a_norm (diagnostic only,
no adapter derived from it) on the *same* existing z-labels — no GPU
retraining needed, runs in minutes. Verified on synthetic data: an
invertible piecewise-linear (non-affine) relationship gives affine
R²=0.845 vs. nonlinear R²=0.998 (clear separation); pure noise gives
near-chance R² for both.

```
./run_pipeline.sh probe --nonlinear
```
```
=== nonlinear (MLP) probe z -> a_norm (val R^2) ===
  dx 0.3290  dy 0.3324  dz 0.2732  droll 0.0814  dpitch 0.1392
  dyaw 0.3393  gripper 0.4465   mean 0.2773
>>> diagnosis: weak/partial action signal in z, still far from usable.
```

0.28 vs. chance-level 0.04 (affine) — a real, ~7x jump, but still well
below the 0.6 gate even nonlinearly.

**Per-dimension pattern, and why it mattered:** translation (dx/dy/dz)
+ yaw + gripper scored 0.27–0.45; roll/pitch scored only 0.08/0.14. This
asymmetry is hard to explain via "z is contaminated by scene noise"
(noise has no reason to spare specific action dimensions) but fits
naturally with roll/pitch being much less visually observable than
translation/yaw/gripper for a parallel gripper viewed from
agentview+wrist cameras — i.e. it looked like partial *real* signal
gated by observability, not pure contamination. This motivated checking
the probe's own math before concluding anything about the LAM.

Also flagged separately: reverse (a→z) R²=0.0116 — near zero. This
matters beyond diagnostics: `LiberoZAdapterActionSpace` (the fine-tuning
adapter) encodes real actions into z-space training targets using a
strictly affine map; if the true a→z relationship isn't affine either,
the adapter's encode direction is architecturally suspect regardless of
what the z→a probe says. Not yet acted on — parked pending the rotation
fix's outcome.

### 10. Rotation-composition bug found and fixed (commit `b40c323`)

The stride>1 probe target averaged raw per-step Euler-angle deltas
arithmetically. Finite rotations do not compose by addition/averaging
(non-commutative) — mathematically wrong specifically for the 3 rotation
dims (translation and near-binary gripper compose linearly and are
unaffected).

**Fix:** `_windowed_action_target` now composes per-step delta rotations
as proper SO(3) elements (`scipy.spatial.transform.Rotation`, chained
multiplication) and converts the net rotation back to a per-step-scale
Euler delta. Used by both probes and the adapter-init save.

**Verified on controlled synthetic data:** built `z` as a *perfect*
affine function of the true SO(3)-composed rotation target. Evaluated
against the **old** naive-average target: `droll=0.003, dpitch=-0.0005,
dyaw=0.011` — a perfect relationship scored as R²≈0. Evaluated against
the **new** corrected target: `droll=1.000, dpitch=0.9997, dyaw=1.000`.
Translation/gripper were ≈1.0 either way, confirming they were never
affected. This is a strong structural match to the real data's
roll/pitch pattern (0.08/0.14 low, translation/yaw/gripper 0.27–0.45) —
but does **not** by itself prove the real data will recover; it only
proves the previous rotation-dimension read was contaminated by this bug
and must be re-measured.

**Next command (no retrain/relabel needed, reuses v2 `z_labels.h5`):**
```bash
git pull
./run_pipeline.sh probe --nonlinear 2>&1 | tee logs/probe_v2_fixed.log
```

### 11. Rotation fix result on real data: essentially no change → hypothesis refuted

```
=== affine probe (val R^2) ===
  dx .0241  dy .0365  dz .0655  droll .0130  dpitch .0078  dyaw .0317  gripper .0996   mean .0397
(reverse a->z mean R^2: 0.0118)
=== nonlinear (MLP) probe ===
  dx .3298  dy .3380  dz .2758  droll .0444  dpitch .1362  dyaw .3200  gripper .4460   mean .2700
```

Side-by-side with the pre-fix run (§ 8–9):

| dim | affine before | affine after | nonlinear before | nonlinear after |
|---|---|---|---|---|
| dx | .0241 | .0241 | .3290 | .3298 |
| dy | .0365 | .0365 | .3324 | .3380 |
| dz | .0655 | .0655 | .2732 | .2758 |
| droll | .0116 | .0130 | .0814 | .0444 |
| dpitch | .0078 | .0078 | .1392 | .1362 |
| dyaw | .0328 | .0317 | .3393 | .3200 |
| gripper | .0996 | .0996 | .4465 | .4460 |
| **mean** | **.0397** | **.0397** | **.2773** | **.2700** |

Identical to within MLP-training noise, on *every* dimension, not just
the rotation ones. This cleanly refutes "the composition bug explains
the real data's rotation deficit" — the fix demonstrably works (§ 10's
synthetic test), it just isn't the bottleneck here.

**Free follow-up calculation** (no code/rerun needed — just averaging the
already-reported per-dim numbers): restricting to the 5 dims with an
obvious visual footprint, dx+dy+dz+dyaw+gripper:
`(.3298+.3380+.2758+.3200+.4460)/5 = 0.342`. Still well below 0.6. So the
"roll/pitch are unobservable, the rest is fine" reframe does not by
itself rescue the gate — signal is weak-to-moderate (0.27–0.45)
across the board, not "6 good dims + 2 bad ones."

**Where this leaves the three original explanations for the v2 recon/R²
tension (§ 8):**
- Copy-shortcut / z-collapse (§ 6–7 cause) — fixed, ruled out (recon
  moved, var healthy).
- Rotation-composition math (§ 10) — fixed, ruled out for real data
  (this section).
- Remaining candidate: z's ~15–25% explained delta-energy is genuinely
  dominated by something other than the robot's own commanded action
  (object motion, contact dynamics, scene context) that happens to
  correlate moderately with the visually-large dims (translation, yaw,
  gripper) and weakly with the visually-subtle ones (roll, pitch) —
  i.e. mostly the contamination hypothesis, not a fixable bug.

**Decision point put to the user** (three options, trade-offs as
reasoned above):
1. Scale up the LAM (bigger `--lam_dim`/`--enc_depth`/`--dec_depth`,
   more iters) — cheap (~1 GPU-day), tests capacity/training-length
   before bigger changes, doesn't change the paper's claim.
2. CLAM-style few-shot grounding — mix a small amount of real
   action-labeled data into LAM training as an auxiliary supervised
   loss. Likely bigger payoff, but shifts the paper's core claim from
   "fully action-free" to "action-free + light grounding" (still
   defensible, matches published precedent, but is a real pivot in
   framing).
3. Pivot to Innovation 5 (hybrid discrete-continuous flow matching for
   the gripper) — cut losses, redeploy effort; has a cheap same-day
   motivation check (contact-frame gripper-prediction histogram on the
   existing SimVLA baseline checkpoint).

Awaiting the user's call — see "Current status" at the top.

---

## Commit reference

| Commit | What |
|---|---|
| `a45fa99` | Initial full pipeline (latent_action package + integration) |
| `9cb6158` | multiprocessing file_system sharing strategy (partial fix) |
| `bc6cdc3` | Raw uint8 IPC + GPU-side preprocessing (actual shm fix) |
| `df5185d` | Forward `--iters` to train-lam stage in `run.py` |
| `b04c615` | `reserve_gpu.py` — memory-only GPU holder |
| `ed75b43` | LAM v2 — delta prediction, stride 4, window-aligned probe target |
| `70a1e4e` | Nonlinear (MLP) diagnostic probe |
| `b40c323` | SO(3) rotation-composition fix for the probe target |

## Go/no-go gate criteria (unchanged since project start)

- Affine probe mean R² ≥ 0.6 → PASS, proceed to `pretrain`/`finetune`.
- < 0.6 after reasonable tuning → the affine-identifiability premise for
  this LAM design fails; escalate (architecture changes) or pivot.
- The nonlinear probe is diagnostic only — it does not pass the gate by
  itself, since the downstream adapter (`LiberoZAdapterActionSpace`) is
  currently a strictly affine map in both directions.

---

# Reference (stable — the framing, not the running log)

## Paper skeleton (abstract structure)

**Scientific gap.** In a VLA, the vision-language backbone inherits
internet-scale pretraining, but the generative action head (the
flow-matching expert) is trained from scratch on a few hundred hours of
robot demonstrations. Latent-action pretraining aims to close this
asymmetry by learning actions from action-free video, but existing work
either quantizes latent actions into a discrete codebook
([LAPA](https://arxiv.org/abs/2410.11758),
[UniVLA](https://arxiv.org/abs/2505.06111)) — capping fine-grained
continuous control — or, in the continuous case
([CLAM](https://arxiv.org/abs/2505.04999),
[villa-X](https://arxiv.org/abs/2507.23682)), is purely empirical with no
theory of *when* the latent space is recoverable.

**Core challenge.** Without action labels, is the continuous latent-action
space **identifiable up to an affine transform** of the true action
space? The obstacle is that camera ego-motion, lighting, and object
dynamics all produce feature-space change that is confounded with the
robot's own action.

**Method (as designed).** An inverse-dynamics encoder + forward decoder on
frozen SigLIP features, with (i) VICReg variance/covariance whitening and
(ii) shared ego-motion-invariance regularization, claimed to pin the
latent space to affine-identifiability — so a probe-initialized *frozen
affine adapter* suffices to transfer a flow expert pretrained purely on
latent actions.

**Key experiment (the gate).** A ridge probe z↔a should recover the true
actions at high R² (≥0.6) if the affine-identifiability claim holds. It
does not (see verdict below) — the claim is falsified for this data/design
in its current form.

## Why this thread is PAUSED (verdict)

The go/no-go gate — affine probe mean R² ≥ 0.6 — was never cleared:

- **v1** (stride-1, absolute reconstruction): R² = 0.03. Cause: at
  10 Hz/128 px, one step of motion is nearly invisible in coarse features,
  so the decoder copied the previous frame and z collapsed to scene
  identity.
- **v2** (delta prediction, stride 4): recon dropped from 1.0 to ~0.75
  (z explains real feature-change variance), but affine probe R² stayed at
  0.04 and the *nonlinear* (MLP) probe reached only 0.27 — z carries real
  but weak, non-affine, partly-contaminated action information.
- Two structural bugs found, fixed, and confirmed **not** to be the
  bottleneck: z-collapse (delta-prediction fix) and a rotation-composition
  error in the probe target (SO(3) fix; verified on synthetic data it can
  turn a perfect affine relation into R²≈0, but changed the real numbers
  by ~0).
- Even restricting the nonlinear probe to the 5 visually-observable dims
  (excluding roll/pitch) only reaches R² ≈ 0.34.

**Conclusion:** for LIBERO at this resolution/backbone, fully
action-free continuous latent actions are not affine-identifiable to a
usable degree. Reviving this thread would require either CLAM-style
few-shot grounding (mixing a little real-action supervision — a genuine
change to the "fully action-free" claim) or a nonlinear adapter (which
weakens the identifiability contribution). The user chose to pivot to
Innovation 3 instead of taking either.

## What carries forward to other threads

- The **delta-prediction + normalized-loss (1.0 = useless)** convention —
  reused directly by Innovation 3's PrefixStateUpdater.
- The **shm-safe data path** (`data.py::gpu_preprocess`, `_demo_frames_raw`)
  — reused by Innovation 3.
- The **diagnostic-first discipline**: gate on a mechanism-level metric
  (probe R²) cheaply before committing GPU-weeks to downstream training.

## Code map

| File | Role | Key pieces |
|---|---|---|
| `config.py` | env/workspace | `Workspace` layout under `$SIMVLA_CHECKPOINTS`; `latest_checkpoint` |
| `make_splits.py` | Stage 0b | demo-level p1/p10 low-data splits (`demos` whitelist) |
| `data.py` | data | `FramePairDataset`; `gpu_preprocess`, `_demo_frames_raw` (shm-safe, shared with Innovation 3); `iter_demos` |
| `models.py` | LAM | `FrozenVisionBackbone`; `LatentActionModel` (delta forward); `variance_covariance_reg`, `shared_ego_augment`; `VectorQuantizerEMA` (discrete ablation) |
| `train_lam.py` | Stage 1 | LAM training; logs `recon`/`var`/`cov`/`inv`/`delta_energy` |
| `label_z.py` | Stage 1b | per-step z labels (+ `--ego_aug` stress); auto-detects stride from ckpt |
| `probe.py` | Stage 1c | affine + `--nonlinear` MLP probe; SO(3) window target; go/no-go gate; adapter init |
| `run.py` | driver | `python -m latent_action.run <stage>`; multi-GPU via accelerate |
| (integration) | — | `models/action_hub.py::LatentZActionSpace`/`LiberoZAdapterActionSpace`; `datasets/domain_handler/libero_z.py` |

## How this document is maintained

Append a new dated/numbered entry to the chronological log for every
experiment or fix, in the same format as above (context → command →
result → diagnosis). Update "Current status" at the top every time so a
reader only needs that section for the current state; the log below is
for detail and audit. Keep the Reference section in sync when the framing
changes. Commit alongside the corresponding code change when there is one.
