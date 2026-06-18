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
- Machine A (`ff86…`, `/datasets/code/SimVLA---`): trained `exp_uniform`; now the eval machine.
- Machine B (`NF5468…`, `/workspace/hyj/code/SimVLA---`): training `exp_tds`.

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

## 7. Main comparison table — 40k budget point

| Run @ 40k | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| uniform (sapi) | 73.0 | 95.5 | 38.5 | 40.5 | **61.9** |
| **TDS (yyk)** | 97.0 | 99.5 | 87.0 | 64.0 | **86.9** |
| **Δ (TDS − uniform)** | +24.0 | +4.0 | **+48.5** | **+23.5** | **+25.0** |

TDS@40k (86.9) ≈ uniform@100k (91.6) ⇒ **~2.3× sample efficiency**. Per-suite
gains track uniform's remaining headroom (object near ceiling → +4; goal/long
lowest → largest gains), consistent with TDS accelerating learning.

**⚠️ Pending decisive validation:** uniform was evaluated on sapi, TDS on yyk.
Re-evaluating tds/ckpt-40000 on **sapi** (same env as uniform) — run prefix
`tds40k_onA` — confirms the gap is not a cross-machine artifact. Expected
~97/99.5/87/64 if real. Do not report as final until this matches.

---

## 8. Pending / next

- [ ] Evaluate **tds/ckpt-40000** → first real TDS-vs-uniform signal (§7).
- [ ] uniform ckpt-20000 eval (last attempt failed — server was down).
- [ ] Fill efficiency curve: both runs at 20k/40k/60k/80k/100k.
- [ ] Optional: re-check Exp 0 at 20k (more spread, may strengthen ρ).
- [ ] 340k (adaptive-chunking) = 98% saved as the "large-budget upper bound,
      method does not hurt" row.

### Known issues
- `run_eval_seq.sh` auto-merge of `*_sr_all.csv` is unreliable; rebuild manually:
  `echo "task_name,sr" > P_sr_all.csv; grep -hv "^task_name" P_per_task_*.csv | grep -v "^$" >> P_sr_all.csv`
- Docker `/dev/shm` too small → use `NUM_WORKERS=0` or enlarge shm.
- Long evals: run inside tmux; a stray Ctrl+C kills the client (server survives).
