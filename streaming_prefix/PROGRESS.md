# Progress Log — Streaming Recursive Prefix Encoding (Innovation 3)

Living document for this research thread (separate from
`latent_action/PROGRESS.md`, which tracks Innovation 4). Updated after
every experiment. Newest status at the top.

---

## Current status (last updated: after Phase B training completed)

**Phase B training SUCCEEDED (teacher-forced).** 30k-step distillation
run on real SmolVLM-500M + full LIBERO finished at **recon = 0.504**
(delta_energy 0.174). recon started at exactly 1.0 (zero-init "predict no
change" baseline) and dropped to ~0.50 — i.e. the state-update operator
explains ~50% of the frame-to-frame variance in the fused text-model
output, cleanly beating the "just copy the previous step" baseline. This
is a genuine positive (contrast Innovation 4's LAM, which stalled at
R²≈0): the mechanism works.

**IMPORTANT caveat on what 0.50 means (and does NOT mean):** this is the
*teacher-forced* training error — every step is fed the REAL previous
fused output as cached state. At deployment the operator will be fed its
OWN previous prediction between resets, so error can compound across
steps (exposure bias). 0.50 is therefore the optimistic, single-step,
ideal-cache number; it does not by itself establish closed-loop
viability. Whether recursive self-conditioning drifts geometrically
(bounded) or diverges is exactly the paper's core theoretical claim and
**must be measured in Phase C, not assumed.**

**Phase A (profiling) result that motivated all this:** at the default
10 Euler steps, VLM prefix re-encoding is **45.5%** of per-step latency;
within it the fused text-model forward is 72% (15.19ms) vs. vision tower
23% (4.86ms) — so this work targets the text-model forward, the opposite
of where VLA-Cache/TTF-VLA intervene. As Euler steps shrink (faster flow
heads), vlm_frac *rises* (to 83% at 1 step), strengthening the motivation.

**Phase C diagnostic code delivered (commit pending), not yet run on
real model.** Two scripts, both verified end-to-end on CPU/tiny model:

Run these next (needs the trained `updater_final.pt` from Phase B):
```bash
git pull
# 1. recursive drift: bounded (geometric) or divergent?
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_drift \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --updater_ckpt runs/latent_action_ws/streaming_prefix/updater_final.pt \
    --horizon 20 --num_episodes 50 \
    --output logs/drift_report.json 2>&1 | tee logs/eval_drift.log

# 2. end-to-end latency: streaming vs full re-encode
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_latency \
    --updater_ckpt runs/latent_action_ws/streaming_prefix/updater_final.pt \
    --euler_steps 1 5 10 --reset_periods 5 10 20 \
    --output logs/latency_report.json 2>&1 | tee logs/eval_latency.log
```

**What each answers, and the go/no-go read:**
- `eval_drift`: prints recursive-drift vs. teacher-forced error at each
  step k, their gap (exposure bias), the implied safe reset period for a
  drift budget, and whether drift saturates (consistent with a geometric
  bound) or diverges. **If drift diverges, the streaming approach is not
  closed-loop viable and the theory framing fails — this is the real
  go/no-go, more than the 0.50 training number.**
- `eval_latency`: prints per-step full-vs-streaming speedup and the
  amortized speedup/Hz folding in one full re-encode per reset period.
  Confirms whether Phase A's predicted headroom is realized. (Note:
  meaningless on tiny test models — the replaced text-model must be big
  enough to matter, as it is at 500M.)

After these two, the remaining Phase C work is the closed-loop LIBERO
success-rate comparison (streaming vs. full re-encode vs.
VLA-Cache/TTF-VLA baselines) on the time-vs-success Pareto plot — the
paper's headline figure — which requires wiring the updater into the
evaluation serving path (`evaluation/libero/serve_smolvlm_libero.py`),
not yet done.

---

## Environment / hardware context

Same shared server as Innovation 4 (see `latent_action/PROGRESS.md`):
8x A800-80GB, GPU 6, `reserve_gpu.py` holding spare memory. Profiling
(Phase A) needs only a few hundred MB; training (Phase B) will need
re-sizing the reservation once real-model memory use is observed (this
architecture reuses the frozen ~500M teacher VLM for forward passes only
— no backprop through it — so memory should be well under the
`pretrain`/`finetune`-stage ~70GB ballpark, but not yet measured).

---

## Chronological log

### 1. Phase A: latency profiling (commit `cba0e34`)

Built `streaming_prefix/profile_prefix.py`: times `forward_vlm_efficient`
(the VLM prefix, called once per `generate_actions` call) against the
full inference call across a range of Euler step counts, with no
trained checkpoint required (latency is weight-independent). `--breakdown`
additionally splits the VLM forward into vision-tower / connector /
fused-text-model timings.

Verified end-to-end on CPU with a tiny local Idefics3 (mocked processor
loading) before handing off — ran without error, produced the table,
breakdown, and JSON report (numbers meaningless at that scale, mechanics
confirmed correct).

**Real result** (SmolVLM-500M, batch=1, `hidden_size=768 depth=12
num_heads=12 image_size=384 num_views=2`, `--repeats 30 --warmup 10`):

```
VLM prefix forward (1x per control step): 21.00 +/- 0.05 ms
  vision tower  : 4.86 ms
  connector     : 0.07 ms
  text model    : 15.19 ms (prefix seq_len=96)

 steps   total_ms     vlm_ms  action_ms  vlm_frac       hz
     1      25.27      21.00       4.26     83.1%     39.6
     5      34.76      21.00      13.75     60.4%     28.8
    10      46.18      21.00      25.18     45.5%     21.7
    20      69.64      21.00      48.63     30.2%     14.4
```

**Interpretation:**
- At the standard 10-step config, VLM re-encoding is 45.5% of total
  latency — a large, worth-attacking share.
- The fused text-model forward (72% of the VLM's own cost) dominates
  over the vision tower (23%) by ~3x. This is the opposite of where
  existing VLA-caching literature (VLA-Cache, TTF-VLA) intervenes — they
  target the vision tower, which here is comparatively cheap already.
- As Euler steps shrink (the direction the field is moving — 1-4 step
  flow-matching action heads), vlm_frac *rises* (30%→45%→60%→83%): the
  VLM-prefix bottleneck becomes proportionally worse precisely as action
  heads get faster. Strengthens the motivation rather than weakening it.

**Decision:** proceed to Phase B, retargeted to approximate the
fused text-model forward specifically (not the vision tower).

### 2. Phase B: PrefixStateUpdater + distillation training (this commit)

Added:
- `streaming_prefix/models.py`
  - `VLMPrefixTeacher`: wraps a frozen SmolVLM to expose
    `cheap_forward` (vision tower + connector + text-embedding lookup +
    padding — ~23% of VLM cost, safe to recompute every step) and
    `expensive_forward` (the fused text-model forward — the ~72%
    target) separately.
  - `PrefixStateUpdater`: small transformer (reuses `TransformerBlock`
    from `models/transformer_smolvlm.py` for consistency with the rest
    of the codebase) predicting `vlm_features_t` as
    `cached_state_{t-1} + delta_hat`, where `delta_hat` is computed from
    `(cached_state_{t-1}, cheap_features_t)`. Output projection is
    zero-initialized, so the model starts as an exact identity (predicts
    no change) — deliberately mirroring the delta-prediction fix from
    Innovation 4's LAM (`latent_action/models.py`), including the same
    normalized-loss convention: **1.0 = predicts zero change / useless**.
  - `distillation_loss`: masked (via `attention_mask`, since padded
    batch elements shouldn't count), normalized MSE, same 1.0-baseline
    convention.
- `streaming_prefix/data.py`: `InstructionFramePairDataset` — reuses
  `latent_action.data`'s raw-uint8 frame reading (same shm-safety
  properties as the Innovation 4 fix) and adds per-episode instruction
  tokenization (no action labels needed — purely a feature-distillation
  target).
- `streaming_prefix/train_student.py`: teacher-forced distillation loop.
  Per pair `(frame_t, frame_{t+stride})` from the same episode: compute
  the REAL teacher output at both `t` and `t+stride`; train the updater
  to predict the real `t+stride` output given the real `t` output as
  cached state. **Known limitation, flagged for Phase C:** training is
  teacher-forced (always conditions on the *real* previous output); at
  inference the updater will condition on its *own* previous prediction
  between resets, which can compound drift faster than the training loss
  suggests (exposure bias) — this is exactly what the paper's geometric
  drift bound (`δ/(1−L)` argument) is meant to characterize and bound,
  and Phase C's evaluation must check it empirically, not just assume it.

**Verified** (CPU, tiny local Idefics3, synthetic LIBERO fixture):
- `PrefixStateUpdater` is an exact identity at init (`pred == cached_state`
  to 1e-6).
- `distillation_loss` on a "predict zero change" baseline is exactly
  1.0, as designed.
- The operator can overfit a single real teacher-generated pair from
  loss 1.0 down to 0.07 in 30 Adam steps — confirms the mechanism is
  learnable, not just correctly wired.
- Full `train_student.py` main() loop runs end-to-end (data loading →
  teacher forward → updater forward → loss → backward → checkpoint
  save/load with correct config round-trip, including a bug found and
  fixed: `config_dict()` was missing `mlp_ratio`, silently reloading
  checkpoints with the wrong MLP width).

**Not yet done:** real training on the actual LIBERO data + real
SmolVLM-500M backbone (the "Next action" command at the top of this
document). Everything below Phase B (drift measurement, reset-period
tuning, Pareto comparison against full re-encoding / VLA-Cache / TTF-VLA
baselines, LIBERO closed-loop success-rate evaluation) is still Phase C,
not started.

---

### 3. Phase B training result + Phase C diagnostic code (this commit)

Phase B training on real SmolVLM-500M + full LIBERO (30k steps,
batch 64, stride 1) finished at **recon=0.504** (from a 1.0 zero-init
baseline) — the state-update operator explains ~50% of the fused
text-model output's frame-to-frame variance under teacher forcing. Clear
positive; but this is single-step, ideal-cache error and does not settle
closed-loop viability (see caveat in Current status).

Added Phase C diagnostics:
- `streaming_prefix/eval_drift.py` — recursive drift measurement:
  starting from a real reset, applies the updater recursively on its own
  output for up to `--horizon` steps, recording normalized drift vs. the
  real full re-encode at each k, alongside a teacher-forced reference
  (the gap = exposure bias). Reports implied safe reset period and a
  crude saturation check for the geometric-bound claim.
- `streaming_prefix/eval_latency.py` — end-to-end per-step latency,
  streaming (cheap_forward + updater + action head) vs. full
  (cheap_forward + real text model + action head), plus amortized
  speedup/Hz folding in one reset per period.

Both verified end-to-end on CPU with the tiny local Idefics3 (structure,
normalization, teacher-forced reference, reset-period derivation, JSON
reports all correct; absolute numbers meaningless at tiny scale — an
untrained updater correctly shows drift≈1.0, a tiny text-model correctly
shows ~1.0x speedup).

## Commit reference

| Commit | What |
|---|---|
| `cba0e34` | Phase A — `profile_prefix.py` latency profiling |
| `b0fd755` | Phase B — `PrefixStateUpdater` + distillation training |
| *(this)* | Phase B result (recon=0.50) + Phase C drift/latency diagnostics |

## Open design questions for Phase C (not yet decided)

- **Reset period:** how many control steps between forced full
  re-encodings? Original theory framing suggested aligning to action-chunk
  boundaries (`num_actions`, default 10); needs an empirical drift-vs-period
  sweep once training completes.
- **Exposure-bias check:** does recursive (self-conditioned) application
  at inference drift faster than the teacher-forced training loss
  predicts? This is the paper's core theoretical claim (geometric bound)
  and must be measured directly, not assumed.
- **Stride:** currently defaulting to 1 (matches the real per-control-step
  deployment cadence at 10Hz). Unlike Innovation 4's LAM (where stride=1
  motion was invisible in pooled features), here we specifically care
  about the stride=1 operating point since that's what real deployment
  needs — but if the health-check recon doesn't move at stride=1, larger
  strides are worth a diagnostic look before concluding failure.

## How this document is maintained

Same convention as `latent_action/PROGRESS.md`: append a new entry per
experiment/fix (context → command → result → diagnosis), update "Current
status" at the top every time, commit alongside the corresponding code
change.
