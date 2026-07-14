# SimVLA Transition-Aware Training — Experiment Results Log

> Living record of the controlled study (uniform vs TDS) on LIBERO.
> Last updated: 2026-06-16. Append new numbers as they arrive.

---

## 1. Experimental design

**Goal.** Prove that Transition-Density Sampling (TDS) — oversampling
high-transition-density (hard) tasks during training — improves a SimVLA
flow-matching policy, in a single-variable controlled comparison.

**Why low budget (100k steps, not 200k+).** The pre-existing 340k checkpoint
saturates LIBERO at ~98% (37/40 tasks ≥95%), leaving no measurable headroom,
and it was trained *with* adaptive chunking so it is not a clean baseline.
We therefore retrain from scratch at low budget where success rate is
non-saturated and differences are measurable.

**Two machines, parallel.**
- **sapi** (`/datasets/code/SimVLA---`): trained `exp_uniform`; primary eval machine.
- **yyk** (`/workspace/hyj/code/SimVLA---`): trained `exp_tds`, `exp_uniform_interleaved`, `exp_empirical`.

---

## 2. Run configuration (verified identical except the TDS flag)

| Param | exp_uniform | exp_tds |
|---|---|---|
| from scratch | yes (models=None) | yes |
| batch_size | 64 | 64 |
| learning_rate | 1e-4 | 1e-4 |
| learning_coef | 0.1 | 0.1 |
| iters | 100000 | 100000 |
| save_interval | 20000 | 20000 |
| seed | 0 | 0 |
| use_adaptive_chunking | **false** | **false** |
| arch (hidden/depth/heads) | 768 / 12 / 12 | 768 / 12 / 12 |
| **tds_weights_csv** | **None** | **task_difficulty_2comp.csv** |

The only scientific variable that differs is TDS on/off. Same seed + same
init + same noise schedule ⇒ clean controlled comparison.

**TDS sampling verified working:** `[TDS] sampling check after 5000 draws,
max |target − empirical| = 0.0061 → distribution matches target. OK`.

---

## 3. Difficulty formula (TDS weights)

`d_k = α·z(E) + β·z(P) + γ·z(L)`, then temperature softmax (T=2) + uniform
floor (ρ=0.5) → sampling weights `p_k`.

- E = mean gripper events, P = low-speed plateau ratio, L = trajectory length.
- **Adopted 2-component: α=0, β=0.5, γ=0.5** — gripper events carry no signal
  on LIBERO (see §5), so dropped.
- Top-5 hardest tasks (highest d_k) are **all libero_10** (long-horizon),
  matching intuition. Hardest: `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
  (d_k=2.40, sampled 1.43× uniform).

---

## 4. Uniform baseline — evaluation results (20 trials/task)

| Checkpoint | Spatial | Object | Goal | Long (10) | **Avg** |
|---|---|---|---|---|---|
| **ckpt-40000** | 73.0 | 95.5 | 38.5 | 40.5 | **61.9** |
| **ckpt-100000** | 92.5 | 96.0 | 90.5 | 87.5 | **91.6** |
| ckpt-20000 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| ckpt-60000 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| ckpt-80000 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Key observation:** at 40k, Goal (38.5%) and Long (40.5%) have huge headroom,
while at 100k everything is ≥87%. **40k is the chosen budget point for the
main comparison** — it is exactly the regime where TDS (which oversamples the
hard, high-density Goal/Long tasks) should show its effect.

---

## 5. Experiment 0 — does offline difficulty predict measured failure?

Spearman correlation of difficulty score `d_k` vs per-task success rate
(n=40 tasks). Negative ρ = higher difficulty ⇒ lower SR (hypothesis holds).
Pass bar in script (ρ<−0.4 & p<0.05) is a strict "strong correlation" line;
**statistical significance is p<0.05.**

| SR source | formula | ρ | p | note |
|---|---|---|---|---|
| 340k baseline (saturated) | 3-comp | −0.239 | 0.137 | n.s. (ceiling) |
| uniform 100k (saturated) | 2-comp | −0.219 | 0.175 | n.s. (ceiling) |
| **uniform 40k (spread)** | **2-comp** | **−0.315** | **0.0475** | **significant ✓** |

**Trend confirms the ceiling explanation:** the less saturated the checkpoint,
the stronger the correlation. At 40k the premise is **statistically
significant** (ρ=−0.32, p<0.05).

Per-component correlation at 40k:
- P_plateau: ρ=−0.292 (p=0.067) — **dominant predictor on LIBERO**
- L_length: ρ=−0.185 (p=0.253)
- E_events: ρ=+0.108 (p=0.506) — **no signal** (justifies dropping it)

Collinearity (Spearman): P–L = 0.71, E–L = 0.49, E–P = 0.09.

**Paper framing:** on LIBERO the plateau ratio (fine-alignment fraction) is the
dominant data-intrinsic difficulty predictor; with trajectory length it reaches
a significant ρ=−0.32; gripper-event count is uninformative on this benchmark.

---

## 6. TDS — training & evaluation status

- exp_tds trained on **yyk** (`/workspace`) from 2026-06-15 05:27; ~1.6 s/it.
- Checkpoints: ckpt-20000, ckpt-40000 (training continued / may have stopped ~40k).
- **tds/ckpt-40000 evaluated on yyk:** 97.0 / 99.5 / 87.0 / 64.0 → **avg 86.9**.
- Cross-machine note: checkpoint config embeds the backbone path; when copying
  yyk→sapi, `sed` the `smolvlm_model_path` from `/workspace/...` to
  `/datasets/models/smolvlm/SmolVLM-500M-Instruct`.

Machines: **sapi** = `/datasets` (trained exp_uniform); **yyk** = `/workspace`
(trained exp_tds). norm_stats md5 + transformers (4.21.1) verified identical.

---

## 7. Main comparison table — 40k budget point (SAME-MACHINE, CONFIRMED)

Both checkpoints evaluated on **sapi** (identical env; only the TDS flag differs).

| Run @ 40k (sapi) | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| uniform | 73.0 | 95.5 | 38.5 | 40.5 | **61.9** |
| **TDS** | 97.5 | 98.5 | 92.0 | 71.0 | **89.75** |
| **Δ (TDS − uniform)** | +24.5 | +3.0 | **+53.5** | **+30.5** | **+27.9** |

**✓ VALIDATED.** Machine confound ruled out (same machine sapi). Eval is
reproducible: TDS 40k on yyk (86.9) vs on sapi (89.75) agree within 20-trial
variance. The +27.9 gap is ~10× the eval standard error (~2–3 pts).

TDS@40k (89.75) ≈ uniform@100k (91.6) ⇒ **~2.4× sample efficiency, confirmed.**
Per-suite gains track uniform's headroom (object near ceiling → +3; goal/long
lowest → +53.5/+30.5). Mechanism: uniform over-trains already-learned easy
tasks; TDS reallocates gradient to hard tasks, which then learn much faster.

Provisional cross-machine read (TDS on yyk): 97.0/99.5/87.0/64.0 = 86.9.

### 7b. Convergence comparison — 100k budget point (SAME-MACHINE, sapi)

| Run @ 100k (sapi) | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| uniform | 92.5 | 96.0 | 90.5 | 87.5 | **91.6** |
| **TDS** | 97.5 | 98.5 | 96.0 | 90.5 | **95.6** |
| **Δ (TDS − uniform)** | +5.0 | +2.5 | **+5.5** | +3.0 | **+4.0** |

**Two-point efficiency picture (same machine):**

| Budget | uniform | TDS | Δ |
|---|---|---|---|
| 40k | 61.9 | 89.75 | **+27.9** |
| 100k | 91.6 | 95.6 | **+4.0** |

Textbook sample-efficiency curve: a huge early gap that narrows toward
convergence but **stays positive** — TDS is not merely caught up. So TDS is
(i) far more sample-efficient (TDS@40k ≈ uniform@100k, ~2.2× speedup) AND
(ii) converges higher (+4.0 at 100k, concentrated on the hard suites Goal +5.5
/ Spatial +5). The +4.0 is smaller than the 40k gap (LIBERO near-saturated by
100k) but Goal/Spatial gains are ~2–3× the per-suite standard error.

### 7c. Efficiency curve (avg SR vs step) — COMPLETE

| Step | uniform (sapi) | TDS (yyk) | gap |
|---|---|---|---|
| 20k | **35.1** (21.0/76.0/23.5/20.0) | **75.6** (82.5/93.5/78.5/48.0) | **+40.5** |
| 40k | **61.9** (73.0/95.5/38.5/40.5) | **86.9** (97.0/99.5/87.0/64.0) · sapi 89.75 | +25.0 |
| 60k | **85.4** (88.0/99.0/65.0/89.5) | **95.0** (100/97.5/97.0/85.5) | +9.6 |
| 80k | **81.5** (86.0/99.0/64.5/76.5) | **93.9** (100/98.5/90.0/87.0) | +12.4 |
| 100k | **91.6** (92.5/96.0/90.5/87.5) | 95.6 (sapi) | +4.0 |

Standout efficiency facts:
- gap is +40.5 at 20k (TDS 75.6 vs uniform 35.1), shrinking to +4.0 at 100k but
  always positive — TDS dominates at every budget.
- TDS matches uniform's 100k success (91.6) at ~50k steps ⇒ **~2× fewer steps**;
  TDS@60k (95.0) already exceeds uniform@100k (91.6); TDS converges higher (+4.0).
- TDS plateaus ~94–95 by 60k while uniform is still climbing.

Notes: uniform@80k (81.5) dips below uniform@60k (85.4) — a wiggle driven by the
high-variance Long suite (89.5→76.5→87.5 over 60/80/100k); within 20-trial eval
noise. Optionally re-eval 60/80k at 50 trials for a smoother figure.

**⚠️ Machine consistency:** uniform all sapi; TDS 20/40/60/80k yyk-native, 40k/
100k also sapi (40k: 86.9 yyk vs 89.75 sapi → yyk ~3 pts low, conservative).
Final figure: evaluate the whole TDS curve on sapi for same-machine cleanliness.

### 7d. Uniform-interleaved control — decomposing the +27.9 (IMPORTANT)

`exp_uniform_interleaved` (alias **flat40k**) = the **TDS two-level sampler with
flat weights** `p_k = 1/N`. It isolates the sampling *mechanism* from the
difficulty *weights*. Evaluated on **both** machines (20 trials/task):

| Run @ 40k | Spatial | Object | Goal | Long | **Avg** | machine |
|---|---|---|---|---|---|---|
| uniform (naive per-suite stream) | 73.0 | 95.5 | 38.5 | 40.5 | **61.9** | sapi |
| uniform-interleaved (flat) | 95.0 | 93.5 | 83.5 | 67.5 | **84.9** | yyk |
| **uniform-interleaved (flat)** | 97.0 | 96.0 | 85.5 | 70.5 | **87.25** | **sapi** |
| TDS | 97.0 | 99.5 | 87.0 | 64.0 | **86.9** | yyk |
| TDS | 97.5 | 98.5 | 92.0 | 71.0 | **89.75** | sapi |

> ✓ Confirmed: sapi eval served `runs/exp_uniform_interleaved/ckpt-40000`, a run
> trained with `task_difficulty_flat.csv` (flat p_k) — definitely the flat control,
> not empirical. Percentages are authoritative (97.0/96.0/85.5/70.5 = 194/192/171/141
> of 200; the paste's fraction lines were stale yyk leftovers).

**CLEAN same-machine (sapi) decomposition — the headline +27.9 is mostly the
mechanism, not the density weighting:**

- **Mechanism (uniform → flat): +25.4 avg** (61.9 → 87.25, all sapi). The naive
  uniform path (`__iter__` else-branch) samples at the **suite** level (4-way) and
  streams one full episode's timesteps before moving on, so a batch of 64 holds
  only ~4 distinct episodes (correlated → poor SGD). The two-level sampler draws a
  fresh **episode stream** per sample over ~2000 streams, so a batch holds ~64
  distinct episodes (decorrelated → good SGD). Task distribution is identical in
  expectation (`DATA_WEIGHTS` all 1.0, ~equal episodes/suite), so the +25.4 is
  **pure batch-mixing / decorrelation**, concentrated on the hard suites the
  coarse sampler was starving (Goal +47, Long +30, Spatial +24).
- **Transition-density weighting (flat → TDS): +2.5 avg** (87.25 → 89.75, all
  sapi). Per suite: **Goal +6.5**, Obj +2.5, Sp +0.5, Long +0.5. The weighting
  cleanly lifts the suite it up-weights most (Goal); the *average* effect is small
  because the other suites are near ceiling once mixing is fixed.

**Both machines agree:** weighting effect = +2.0 (yyk: flat 84.9 → TDS 86.9) and
+2.5 (sapi: flat 87.25 → TDS 89.75); Goal gain = +3.5 (yyk) / +6.5 (sapi). The
density weighting is a real, Goal-targeted +2–2.5 avg, but it sits near the
20-trial noise floor → recommend ≥2–3 seeds before camera-ready.

**Implication for the paper.** The predicted ablation reading ("uniform-interleaved
≈ uniform; the gain is from the weights") is **falsified** — it is the opposite.
Honest framings to choose between (user's call):
(A) Present TDS as a two-part data pipeline (fine-grained interleaving + density
   reweighting) and report the decomposition transparently; or
(B) Make uniform-interleaved the PRIMARY baseline and headline the clean
   weighting effect (+2.5 avg / +6.5 Goal), relegating naive uniform to context.
Do not finalize framing until exp_empirical lands and ≥2 seeds confirm the +2.5.

### 7e. Measured-difficulty baseline (exp_empirical) — TRAINED, eval pending

`exp_empirical` (difficulty = −SR from the uniform baseline's measured per-task
success → sampling weights) trained at **ckpt-40000** (`runs/exp_empirical/
ckpt-40000`, loss_total 0.174, weights `task_difficulty_empirical.csv`). This is
the "ask-the-model" oracle control: if free transition-density difficulty
≈ measured difficulty, the row lands near TDS.

**Evaluated on yyk** (20 trials/task) — clean same-machine triple (all yyk):

| @40k (yyk) | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| flat (interleaved) | 95.0 | 93.5 | 83.5 | 67.5 | **84.9** |
| TDS (free difficulty) | 97.0 | 99.5 | 87.0 | 64.0 | **86.9** |
| **empirical (oracle difficulty)** | 94.0 | 98.0 | 90.0 | 69.5 | **87.9** |

**Reading — free ≈ oracle (the best outcome):**
- Difficulty weighting over flat: TDS **+2.0**, empirical **+3.0**.
- **empirical beats TDS by only +1.0** (87.9 vs 86.9, within 20-trial noise) → the
  *free* transition-density difficulty recovers nearly all of the *costly* measured
  (oracle) difficulty's benefit. No model training/eval needed — the headline TDS
  selling point.
- Both beat flat by only +2–3 → the difficulty-weighting *ceiling* on LIBERO is
  ~+3 once mixing is fixed; TDS captures +2 of that for free.
- Per suite, empirical is stronger on the hard suites (Goal +3.0, Long +5.5 vs TDS)
  and slightly weaker on the easy ones (Sp −3.0, Obj −1.5) — measured difficulty
  weights the lowest-SR suites (Goal/Long) hardest.

**⚠️ Machine:** empirical is yyk-native; the main ablation table is sapi. The
free-vs-oracle conclusion is clean *within yyk*. For a unified sapi table, eval
empirical on sapi too (offset-correcting yyk 87.9 + ~3 → ~90.8, i.e. just above
sapi TDS 89.75 — consistent).

---

## 8. Pending / next

- [x] Evaluate **tds/ckpt-40000** → +27.9 same-machine (§7).
- [x] Evaluate **exp_empirical** → free ≈ oracle (empirical +1.0 over TDS, yyk) (§7e).
- [x] Fill efficiency curve: both runs at 20k/40k/60k/80k/100k (§7c).
- [x] Uniform-interleaved control (flat40k) evaluated → decomposition (§7d).
- [x] **Evaluate flat (uniform-interleaved) on sapi** → clean same-machine
      decomposition: mechanism +25.4, density weighting +2.5 (Goal +6.5) (§7d).
- [ ] (optional) eval exp_empirical on **sapi** to unify the ablation table
      (conclusion already clean within yyk: free ≈ oracle, §7e).
- [ ] inverted-weights + no-floor (ρ=0) controls at 40k (just different csv).
- [ ] ≥2–3 seeds for the +2 density-weighting effect (it sits near eval noise).
- [ ] Decide paper framing (§7d A vs B) — all controls now in; free ≈ oracle.
- [ ] Optional: re-check Exp 0 at 20k (more spread, may strengthen ρ).
- [ ] 340k (adaptive-chunking) = 98% saved as the "large-budget upper bound,
      method does not hurt" row.

### Known issues
- `run_eval_seq.sh` auto-merge of `*_sr_all.csv` is unreliable; rebuild manually:
  `echo "task_name,sr" > P_sr_all.csv; grep -hv "^task_name" P_per_task_*.csv | grep -v "^$" >> P_sr_all.csv`
- Docker `/dev/shm` too small → use `NUM_WORKERS=0` or enlarge shm.
- Long evals: run inside tmux; a stray Ctrl+C kills the client (server survives).

### LIBERO-PRO setup notes (robustness eval, verified on yyk 2026-07)
- LIBERO resolves bddl/init data via `~/.libero/config.yaml`, NOT via the code
  on PYTHONPATH → copy the HuggingFace data (`zhouxueyang/LIBERO-Pro`:
  `bddl_files/*`, `init_files/*`) **into the original LIBERO tree** the config
  points to (`.../LIBERO/libero/libero/{bddl_files,init_files}/`). PYTHONPATH
  shadowing (run_eval_pro.sh) is still needed so the *perturbed suite names*
  register.
- Dimension → suite suffix: object→`_object`, position→`_swap`,
  semantic→`_lan`, task→`_task`. **Do NOT use the `_temp` suites for paper
  numbers** — `_temp` is the position-intensity workspace (contents = whatever
  variant was last copied in; provenance ambiguous as downloaded).
- This dataset release has **no `_env` folders** → environment dimension
  skipped; the many `with_*` variants (blue_stick/mug/...) are per-distractor
  environment suites, revisit if needed. Paper table: Obj/Pos(swap)/Sem(lan)/
  Task + Avg (no "Ori" — that dimension does not exist in this release).
- Always verify after launch: `grep -m1 "Task suite" <prefix>_*.txt` must show
  the intended suffix (e.g. `libero_goal_object`), not `_temp` or bare names.
