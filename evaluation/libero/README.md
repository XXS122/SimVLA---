# Evaluation on LIBERO

## 1. Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## 2. Start Server

```bash
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
```

or 

```
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
```

## 3. Run Evaluation

Quick evaluation on selected tasks:

Full evaluation on all task suites:

```bash
conda activate libero
bash run_eval_all.sh 8102 10 "eval_simvla_150k" "0 1 2 3"
bash run_eval_all.sh 8102 50 "eval_simvla_150k" "0 1 2 3"
```

## 4. Test-Time Scaling (TTS)

Inference-side compute scaling: sample N candidate action chunks per state
(best-of-N with a training-free selector), vary the number of Euler steps,
and probe per-state uncertainty via the velocity-field variance across
noise seeds. The policy checkpoint is used as-is — no retraining.

**Start the server with a diagnostics log** (one server serves the whole sweep;
clients override TTS parameters per request):

```bash
conda activate simvla
python serve_smolvlm_libero.py \
    --checkpoint <ckpt> \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102 \
    --diag_log ./tts_sweep/diag.jsonl
```

**Run a single configuration:**

```bash
conda activate libero
python libero_client.py --port 8102 --task_suite libero_spatial \
    --num_trials 20 --no_video \
    --num_samples 8 --ode_steps 10 --selector consensus \
    --log_results ./tts_sweep/results_libero_spatial_N8_S10_consensus.jsonl
```

**Run the full (N x steps) sweep:**

```bash
bash run_tts_sweep.sh 8102 libero_spatial 20 ./tts_sweep consensus
# grid override: N_LIST="1 4 16" S_LIST="5 10 20" bash run_tts_sweep.sh ...
```

**Analyze** (scaling table, uncertainty-vs-gripper-event alignment,
episode-level uncertainty/success correlation):

```bash
python analyze_tts.py --results_dir ./tts_sweep --diag ./tts_sweep/diag.jsonl \
    --out ./tts_sweep/summary.csv --plots ./tts_sweep
```

Selectors: `consensus` (candidate closest to the candidate mean),
`smoothness` (lowest squared jerk), `first` (plain sampling baseline).
`--adaptive_threshold <v>` collapses to a single candidate after the first
Euler step whenever the uncertainty probe falls below `v` (adaptive compute).
