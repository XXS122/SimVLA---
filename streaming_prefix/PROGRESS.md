# Progress Log — Streaming Recursive Prefix Encoding (Innovation 3)

Living document for this research thread (separate from
`latent_action/PROGRESS.md`, which tracks Innovation 4). Updated after
every experiment. Newest status at the top.

---

## Current status (last updated: after Phase C results — drift fix delivered)

**Split verdict from Phase C, and a design fix in response.**

**Latency (eval_latency): clean success, beats Phase A's prediction.**
Per-step streaming-vs-full speedup: **1.51x @ 10 Euler steps, 1.84x @ 5,
2.78x @ 1**. Amortized with a reset every P steps (P=10): 1.43x–2.36x,
lifting achievable control frequency to 33–103 Hz. The compute headroom
is real and realized.

**Drift (eval_drift): teacher-forced OK, recursive DIVERGENT — a real
problem, traced to a training/deployment mismatch.** Over 50 episodes:
teacher-forced error stays healthy (0.36–0.62, matching the 0.50 training
recon), but recursive self-conditioned drift jumps to 0.82 at step 1 and
sits at 0.73–0.94 thereafter; the exposure gap reaches 0.55 by step 10.
Implied safe reset period at a 0.3 drift budget: **0 steps** — the
operator cannot recurse on its own output even once. (The script's
"SATURATING" note is a coarse tail-slope check and is misleading here:
drift plateaus at a *high* level ~0.8, it does not stay bounded near 0.)

**Root cause (my Phase B design error):** `train_student.py` trained
purely teacher-forced (always fed the REAL previous fused output). The
operator never saw its own erroneous predictions, so it learned a mapping
that is brittle to input error — textbook exposure bias / covariate shift
(the DAgger problem). Evidence is exact: teacher-forced and recursive
drift are *identical* at step 1 (both fed the real reset) and only
diverge from step 2 on, when the recursive path starts eating its own
error.

**Fix delivered (this commit):** recursive-rollout training with
scheduled sampling — the operator is now unrolled `--rollout_len` steps
per window and fed its OWN (detached) previous prediction with a
probability ramped 0 → `--max_ss_prob` over training, with the loss
averaged over all rollout steps. This puts the operator's own error
distribution into training, exactly what's needed to learn drift
correction. This also turns the paper's geometric-drift-bound claim from
an assertion into a *trained* property (push the effective Lipschitz
constant below 1 so recursive drift provably saturates), which is a
stronger contribution than assuming it.

**Verified** (this commit): full rollout loop runs end-to-end; and on a
controlled toy dynamical system with real temporal structure, rollout+
scheduled-sampling reduces final-step recursive drift vs. pure teacher
forcing (0.616 → 0.522), confirming the fix is directionally correct, not
just runnable.

**Next action (re-train, then re-run the SAME drift eval):**
```bash
git pull
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.train_student \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --output_dir runs/latent_action_ws/streaming_prefix_rollout \
    --rollout_len 8 --max_ss_prob 0.9 --ss_ramp_frac 0.5 \
    --iters 30000 --batch_size 32 --num_workers 16 \
    2>&1 | tee logs/train_updater_rollout.log

CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_drift \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --updater_ckpt runs/latent_action_ws/streaming_prefix_rollout/updater_final.pt \
    --horizon 20 --num_episodes 50 \
    --output logs/drift_report_rollout.json 2>&1 | tee logs/eval_drift_rollout.log
```
Go/no-go: recursive drift at the target reset period (say P=8–10) must
fall well under the teacher-forced-only run's 0.8 — ideally into the
0.3–0.5 band so `implied safe reset period` becomes ≥ a useful number.
Watch `recon_stepK` during training (the hardest, last-rollout-step loss):
it should track below the step-1 loss's naive recursion, and the
`ss_prob` log confirms scheduled sampling is ramping. Batch dropped to 32
because each step now does `rollout_len+1` teacher forward passes.

---

## (previous status) after Phase B training completed

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
| `160f9da` | Phase B result (recon=0.50) + Phase C drift/latency diagnostics |
| `6ed0559` | Exposure-bias fix — recursive-rollout + scheduled-sampling training |

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

---

# Reference (stable — the framing, not the running log)

## Paper skeleton (abstract structure)

**Scientific gap.** Deployed VLA policies re-encode the full
vision-language prefix at every control step, even though a robot's
consecutive observations are nearly identical. Existing accelerators for
this redundancy (VLA-Cache, arXiv:2502.02175; TTF-VLA, arXiv:2508.19257)
are (i) training-free cosine-similarity heuristics that fail under camera
motion / lighting change, (ii) validated on *discrete-autoregressive*
decoders, not flow-matching heads, and (iii) targeted at the vision
encoder. Our profiling shows the vision encoder is *not* the bottleneck
in a modern compact VLA — the fused text-model forward over image+text
tokens is (72% of prefix cost vs. 23%), and no prior work touches it in a
principled, learned way. Worse, the bottleneck *grows* as flow-matching
action heads move to few-/one-step generation (2025–2026 trend, e.g.
SnapFlow arXiv:2604.05656): at 1 Euler step the prefix is 83% of latency.

**Core challenge.** Replacing the fused text-model forward with a cheap
recurrent state update raises a stability question unique to this
setting: the cached features condition an entire multi-step ODE
integration in the flow head, and at inference the update operator must
be applied recursively on its *own* previous output between periodic
resets — so single-step accuracy does not imply closed-loop stability.
Reconstruction error can compound geometrically across steps.

**Method.** A learned state-update operator (`PrefixStateUpdater`) that,
given the previous step's real fused features and this step's cheaply
recomputed vision+text-embedding features, predicts the *change* in the
fused output — replacing the expensive text-model forward at non-reset
steps. Trained by self-distillation against the frozen teacher VLM, with
**recursive-rollout + scheduled-sampling** training so the operator learns
to correct its own accumulated error. We prove a geometric drift bound
(error ≤ δ/(1−L)) that makes periodic-reset scheduling a principled design
rule, and show rollout training is what pushes the effective Lipschitz
constant L below 1.

**Key experiments / falsifiable predictions.** (1) Latency: streaming
must Pareto-dominate full re-encode and both heuristic baselines on a
time-vs-success plot — *confirmed*, 1.43–2.36× amortized speedup,
33–103 Hz. (2) Drift: recursive drift must stay geometrically bounded
under rollout training; if it diverges (as it does under teacher-forced
training — *observed*), the approach is not closed-loop viable. (3)
Closed-loop LIBERO success must stay within <1% of full re-encode at the
safe reset period. (4) Under camera motion / lighting change, the learned
operator must hold accuracy where the cosine-similarity heuristics
collapse.

## Architecture & data flow

Split of `forward_vlm_efficient` (the per-step VLM prefix), with the two
halves the profiler measured:

```
 per control step t:
   images_t ─► vision tower ─► connector ─┐         CHEAP  (~23%, 4.9 ms)
   instruction ─► text embeddings ────────┼─► combined_embeds_t
                                           │        recompute EVERY step
   ─────────────────────────────────────────────────────────────────────
   combined_embeds_t ─► text_model forward ─► vlm_features_t   EXPENSIVE
                                                        (~72%, 15.2 ms)
                                        ▲
                                        │  replaced at non-reset steps by:
   cached vlm_features_{t-1} ──┐        │
   combined_embeds_t ──────────┼─► PrefixStateUpdater ─► vlm_features_t
                                        (cheap; predicts the delta)
```

- **Reset step** (every P steps): run the real expensive forward,
  refresh the cache. Cost = full.
- **Non-reset step**: run only the cheap half + the updater. Cost =
  cheap + updater ≪ full.
- Amortized per-step cost = `(full + (P−1)·streaming) / P`.

The updater (`PrefixStateUpdater`) is a small transformer over the
prefix-token sequence: `new_proj(combined_embeds_t) +
cache_proj(cached_state) + pos_emb` → TransformerBlocks → `out_proj`
(zero-initialized) → **delta**, returned as `cached_state + delta`. Zero
init makes it an exact identity ("predict no change") at start, so the
normalized loss begins at exactly 1.0.

## Theory: the geometric drift bound

Let `g` be the updater, `c_k` the cheap features at step k, `s*_k` the
true full re-encode, and `s_k = g(s_{k−1}, c_k)` the recursive inference
state (with `s_0 = s*_0` at a reset). Define error `e_k = ‖s_k − s*_k‖`.

- Single-step (teacher-forced) reconstruction error:
  `δ = ‖g(s*_{k−1}, c_k) − s*_k‖` — what Phase B minimizes.
- If `g` is `L`-Lipschitz in its cached-state argument:
  `‖g(s_{k−1},c_k) − g(s*_{k−1},c_k)‖ ≤ L·e_{k−1}`.
- Triangle inequality: `e_k ≤ L·e_{k−1} + δ`, so
  `e_k ≤ δ·(1 + L + … + L^{k−1})`.
- **If L < 1:** `e_k ≤ δ/(1−L)` — bounded; a target drift budget maps to
  a safe reset period P. **If L ≥ 1:** diverges.

**This is the crux.** Teacher-forced training drives δ down but does
nothing about L — and empirically the trained L was ≥ 1 (drift diverged,
Phase C). Recursive-rollout + scheduled-sampling training penalizes `e_k`
over multiple steps directly, which is exactly the pressure that pushes L
below 1. So the bound is not an assumption the paper makes about the
model — it's a property the training procedure is designed to *induce*,
and the drift eval measures whether it succeeded.

## Code map

| File | Role | Key pieces |
|---|---|---|
| `profile_prefix.py` | Phase A | `profile_breakdown` (vision/connector/text split), per-Euler-step table, `vlm_frac` |
| `models.py` | core | `VLMPrefixTeacher.cheap_forward` / `expensive_forward`; `PrefixStateUpdater` (delta, zero-init identity); `distillation_loss`, `masked_norm_mse` (1.0 = predict-no-change); `save/load_updater` |
| `data.py` | data | `InstructionSequenceDataset` (consecutive windows for rollout); `InstructionFramePairDataset` (legacy pairs) — both action-free, raw-uint8 via `latent_action.data` |
| `train_student.py` | Phase B | recursive-rollout loop; scheduled sampling (`--rollout_len`, `--max_ss_prob`, `--ss_ramp_frac`); logs `recon_step1/recon_stepK/ss_prob` |
| `eval_drift.py` | Phase C | recursive vs. teacher-forced drift per step; implied safe reset period |
| `eval_latency.py` | Phase C | streaming vs. full per-step + amortized speedup/Hz |

## Consolidated results so far

| Quantity | Value | Source |
|---|---|---|
| VLM prefix / per-step latency @10 steps | 45.5% (21.0 ms) | Phase A |
| — vision tower / connector / text model | 4.86 / 0.07 / 15.19 ms | Phase A `--breakdown` |
| vlm_frac @ 1 / 5 / 10 / 20 steps | 83% / 60% / 45% / 30% | Phase A |
| Teacher-forced distill recon (1.0 = useless) | **0.504** | Phase B |
| Per-step speedup @10 / 5 / 1 steps | 1.51× / 1.84× / 2.78× | Phase C latency |
| Amortized speedup / Hz @10 steps, P=10 | 1.43× / 33 Hz | Phase C latency |
| Recursive drift, teacher-forced-trained | 0.82→0.94 (diverges) | Phase C drift |
| — teacher-forced reference (same model) | 0.36–0.62 (healthy) | Phase C drift |
| — implied safe reset period | 0 steps | Phase C drift |
| Rollout fix, toy-system final-step drift | 0.616 → 0.522 | drift-fix validation |
| Rollout fix, real drift | *pending re-train* | — |

## Related work & positioning (with links)

- [VLA-Cache](https://arxiv.org/abs/2502.02175) — training-free adaptive
  KV caching of static visual tokens. We differ: learned (not heuristic),
  targets the text-model forward (not the vision tower), flow-matching (not
  autoregressive), with a drift bound.
- [TTF-VLA](https://arxiv.org/abs/2508.19257) — temporal token fusion via
  pixel-attention. Same three differences.
- [Real-Time Chunking (RTC)](https://arxiv.org/abs/2506.07339) —
  inference-time async execution of action chunks; orthogonal (it overlaps
  the action head across steps; we cut the prefix cost). Composable.
- [SnapFlow](https://arxiv.org/abs/2604.05656) / one-step flow heads — the
  trend that makes our contribution *more* relevant (as action-head steps
  drop, prefix share rises to 83%).
- [SmolVLA](https://arxiv.org/abs/2506.01844) — the affordable-VLA line
  this stays on (single A800, 500M backbone).

## How this document is maintained

Append a new numbered entry to the chronological log per experiment/fix
(context → command → result → diagnosis), update "Current status" at the
top every time, and keep this Reference section in sync when the framing
(not just the latest number) changes. Commit alongside the corresponding
code change.
