# Research Overview — VLA Architecture Innovations on SimVLA

Single entry point for the research work layered on top of the SimVLA
baseline in this repository. Two independent research threads, each with
its own living progress log. Read this file first for the big picture,
then dive into the per-thread logs.

| Thread | Directory | Status | Log |
|---|---|---|---|
| **Innovation 3** — Streaming recursive prefix encoding | `streaming_prefix/` | **ACTIVE** — latency win confirmed, drift fix in flight | [`streaming_prefix/PROGRESS.md`](streaming_prefix/PROGRESS.md) |
| **Innovation 4** — Identifiable continuous latent-action pretraining | `latent_action/` | **PAUSED** — probe gate not cleared (see its log for the verdict) | [`latent_action/PROGRESS.md`](latent_action/PROGRESS.md) |

Both build on the same frozen SmolVLM-500M backbone and the same LIBERO
data pipeline that SimVLA already ships, and both deliberately reuse
existing modules (`models/transformer_smolvlm.py`, `datasets/…`,
`models/action_hub.py`) rather than forking them.

---

## The SimVLA baseline these build on

SimVLA (this repo's `readme.md`) is a compact VLA:

```
   multi-view images ─┐
                       ├─► SmolVLM-500M (SigLIP vision + connector +
   language instruction┘      Idefics3 text model)  ──► vlm_features
                                                             │
                                       proprio ──┐           │
                              noised action x_t ─┼───────────┤
                                     flow time t ┘           ▼
                          SmolVLMActionTransformer (flow-matching head)
                                                             │
                                                     velocity field v_t
                                              (Euler-integrated to actions)
```

- Backbone: `models/modeling_smolvlm_vla.py::forward_vlm_efficient`
  (vision tower → connector → fuse image+text tokens → text-model
  forward → `vlm_features`).
- Action head: `models/transformer_smolvlm.py::SmolVLMActionTransformer`
  (flow matching; two conditioning modes — concat and AdaLN).
- Action space + normalization: `models/action_hub.py`.
- Data: LIBERO HDF5 via `datasets/domain_handler/libero_hdf5.py`.

The two research threads attack two different structural questions about
this pipeline:

- **Innovation 3** attacks the *inference data flow*: the backbone is
  re-run from scratch every control step even though consecutive frames
  barely change. Can we replace the expensive part with a learned
  recurrent state update?
- **Innovation 4** attacks the *action-head initialization*: the
  flow-matching head is trained from scratch on robot demos while the
  backbone enjoys internet-scale pretraining. Can the head be pretrained
  on action-free video via identifiable latent actions?

---

## Shared infrastructure (built once, reused by both)

Delivered while building Innovation 4, reused by Innovation 3:

- **`reserve_gpu.py`** — memory-only GPU holder for the shared server
  (keeps co-tenants off the GPU you're training on without stealing SM
  cycles). Usage in its docstring.
- **`latent_action/data.py::gpu_preprocess`** and `_demo_frames_raw` —
  DataLoader workers ship raw uint8 frames (~100 KB/sample) and
  resize+normalize on GPU, avoiding the `/dev/shm` bus-error that Docker's
  64 MB default shm caused with preprocessed-float IPC. Both threads use
  this.
- **`paths.env`** convention — all scripts read paths from environment
  variables (`SIMVLA_SMOLVLM_MODEL`, `LIBERO_DATASETS`,
  `SIMVLA_CHECKPOINTS`, `WANDB_*`, `CUDA_DEVICES`, `NUM_GPUS`,
  `SIMVLA_Z_DIM`). See `paths.env.example`.

## Hardware / environment

- Shared server, 8× NVIDIA A800-SXM4-80GB; work pinned to GPU 6 via
  `CUDA_DEVICES=6`.
- Local SmolVLM-500M at `$SIMVLA_SMOLVLM_MODEL`; LIBERO (4 subsets,
  ~2000 demos) at `$LIBERO_DATASETS`.
- Docker container with a small `/dev/shm` — see the gpu_preprocess note
  above.
- W&B project `simvla` (user `xxshyj`).

## Recurring engineering lessons (apply to any future thread here)

1. **Small `/dev/shm` + multi-worker DataLoader = SIGBUS.** Ship raw
   uint8 through IPC, preprocess on GPU. (Innovation 4, commit `bc6cdc3`.)
2. **Delta prediction beats absolute prediction when inputs barely
   change.** A model that can copy its input will, leaving the conditioning
   signal gradient-starved. Predict the *change* and normalize the loss so
   1.0 = "predicted no change". Used by both the LAM (Innovation 4) and the
   PrefixStateUpdater (Innovation 3).
3. **Teacher forcing ≠ deployment.** Any model applied recursively on its
   own output at inference must be *trained* recursively (rollout +
   scheduled sampling), or it drifts. (Innovation 3, Phase C.)
4. **Probe/measure the middle quantity, not just the end metric.** LIBERO
   success rate is saturated; every claim here is validated on a
   mechanism-level diagnostic (probe R², recon vs. a 1.0 baseline, drift
   curves, latency breakdowns) before touching success rate.

## Git / provenance

All work on branch `claude/vla-research-innovations-wjijet`. Commits show
as "Unverified" on GitHub (no signing key in the execution container);
author/committer identity is correct. See each thread's log for its
commit table.
