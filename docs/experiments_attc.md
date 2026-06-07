# ATTC 推理加速实验记录

**模型检查点**：`./runs/simvla_libero_small/step_200000`（根据实际路径修改）
**评估轮数**：每个任务集 20 trials（快速验证），正式发表用 50 trials
**GPU 分配**：服务端独占 1 块（如 GPU0），评估端 4 块并行（如 GPU1-4，跑 4 个任务集）

> ⚠️ **重要：成功率和延迟要分开测**
> - **成功率（SR%）**：用 `run_eval_all.sh` 4 路并行测（快）。
> - **延迟（latency）**：必须用**单任务集、单连接**单独测（见下方"延迟测量"）。
>   因为 4 路并行时 4 个 client 共用一个 server 串行排队，测出的延迟约是真实单流的
>   4 倍且失真，**不能用并行跑的日志延迟**。

---

## 运行方式说明（成功率）

每个实验分两步：先开服务端（终端1），再跑评估（终端2）。

```bash
# 终端1：启动推理服务（替换下方 [实验参数]）
source paths.env
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port [端口] [实验参数]
# 等待日志出现 "SimVLA server listening on 0.0.0.0:XXXX" 再执行终端2

# 终端2：跑评估（GPU 用 1-4，避开服务端的 GPU0）
cd evaluation/libero
bash run_eval_all.sh [端口] 20 [结果前缀] "1 2 3 4"

# 查看成功率（client 打印的是 "Total success rate"）
grep -i "Total success rate" [结果前缀]_*.txt
```

从服务端日志读取缓存命中率（连接关闭时打印）：
```
[ATTC] cumulative hit rate: XX.X%
```

---

## 延迟测量（单连接，必须单独测）

延迟实验只跑**一个任务集、一个 client**，避免排队干扰。每个配置都要单独测一次延迟：

```bash
# 终端1：启动服务（同上，换对应实验参数和端口）
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port [端口] [实验参数]

# 终端2：只跑一个任务集、10 trials（GPU 与服务端错开）
cd evaluation/libero
CUDA_VISIBLE_DEVICES=1 python libero_client.py \
  --host 127.0.0.1 --port [端口] --client_type websocket \
  --task_suite libero_goal --num_trials 10 --no_video

# client 跑完后，从【服务端终端】日志读取：
#   [Latency summary] n=XX  avg=XXXms  p50=XXXms  p95=XXXms
```

> 所有配置的延迟都用**同一个任务集（libero_goal）+ 同样 trials**测，才可横向对比。

---

## 一、主实验（Main Results）

### B0 — Baseline（无缓存）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8100

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8100 20 exp_B0 "1 2 3 4"
```

---

### C1 — Fixed-τ（固定阈值，对标 VLA-Cache）

> 用专门的 `--cache_fixed_threshold` 参数启用固定阈值模式（绕过自适应公式，
> 阈值恒为常数）。这才是真正的固定阈值基线。
>
> **如何选 τ 值（公平对比的关键）**：先跑完 C2（自适应）拿到它的平均 hit rate，
> 然后调 `--cache_fixed_threshold` 让 C1 的 hit rate 与 C2 **尽量接近**（即同等加速比），
> 再比成功率——这叫 "iso-speedup" 对比，最公平。
> 起始可试 `0.02`，hit rate 太低就调大、太高就调小。

```bash
# 终端1（先用 0.02 起步，再根据 hit rate 调整）
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8101 \
  --use_cache \
  --cache_fixed_threshold 0.02 \
  --cache_warmup 3

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8101 20 exp_C1 "1 2 3 4"
```

---

### C2 — ATTC-ours（α=1.0，β=0.9，推荐配置）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8102 \
  --use_cache \
  --cache_alpha 1.0 \
  --cache_beta 0.9 \
  --cache_warmup 3

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8102 20 exp_C2 "1 2 3 4"
```

---

## 二、Ablation 1：α 敏感性（固定 β=0.9）

### A-α1 — α=0.5（激进缓存）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8110 \
  --use_cache \
  --cache_alpha 0.5 \
  --cache_beta 0.9

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8110 20 exp_A_alpha05 "1 2 3 4"
```

---

### A-α2 — α=1.0（与 C2 相同，直接复用结果）

---

### A-α3 — α=1.5（保守缓存）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8111 \
  --use_cache \
  --cache_alpha 1.5 \
  --cache_beta 0.9

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8111 20 exp_A_alpha15 "1 2 3 4"
```

---

### A-α4 — α=2.0（极保守缓存）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8112 \
  --use_cache \
  --cache_alpha 2.0 \
  --cache_beta 0.9

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8112 20 exp_A_alpha20 "1 2 3 4"
```

---

## 三、Ablation 2：β 敏感性（固定 α=1.0）

### A-β1 — β=0.7（短时间窗口，约3帧）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8120 \
  --use_cache \
  --cache_alpha 1.0 \
  --cache_beta 0.7

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8120 20 exp_A_beta07 "1 2 3 4"
```

---

### A-β2 — β=0.9（与 C2 相同，直接复用结果）

---

### A-β3 — β=0.99（长时间窗口，约100帧）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8121 \
  --use_cache \
  --cache_alpha 1.0 \
  --cache_beta 0.99

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8121 20 exp_A_beta099 "1 2 3 4"
```

---

## 四、结果汇总表

### 主实验

| 实验ID | 方法 | Spatial SR% | Object SR% | Goal SR% | L10 SR% | 平均 SR% | avg 延迟(ms) | p50(ms) | p95(ms) | Hit Rate |
|--------|------|:-----------:|:----------:|:--------:|:-------:|:--------:|:-----------:|:-------:|:-------:|:--------:|
| B0 | Baseline（无缓存） | | | | | | | | | — |
| C1 | Fixed-τ | | | | | | | | | |
| **C2** | **ATTC-ours** | | | | | | | | | |

**B0 加速比**（填完 B0 和 C2 后计算）：`B0_avg / C2_avg = ___×`

---

### Ablation 1：α 敏感性（β=0.9 固定）

| 实验ID | α | 平均 SR% | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|:--------:|:-----------:|:--------:|------|
| A-α1 | 0.5 | | | | 缓存最激进 |
| A-α2 | **1.0** | | | | **推荐值（=C2）** |
| A-α3 | 1.5 | | | | |
| A-α4 | 2.0 | | | | 缓存最保守 |

---

### Ablation 2：β 敏感性（α=1.0 固定）

| 实验ID | β | 时间窗口 | 平均 SR% | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|---------|:--------:|:-----------:|:--------:|------|
| A-β1 | 0.7 | ~3帧 | | | | EMA 反应快 |
| A-β2 | **0.9** | ~10帧 | | | | **推荐值（=C2）** |
| A-β3 | 0.99 | ~100帧 | | | | EMA 反应慢 |

---

## 五、实验成功判断标准

| 指标 | 最低要求 | 理想值 |
|------|---------|--------|
| C2 平均 SR% | ≥ B0 × 95% | ≥ B0 × 98% |
| C2 avg 延迟 | ≤ B0 × 70% | ≤ B0 × 60% |
| C2 Hit Rate | ≥ 40% | 50–70% |
| C2 vs C1 SR% | C2 ≥ C1 | C2 明显 > C1 |

**核心结论（写论文时用）**：
- C2 ≈ B0 的成功率 → 说明缓存不影响精度
- C2 延迟 << B0 → 说明推理加速有效
- C2 > C1 → 说明自适应阈值优于固定阈值（体现创新）
- Ablation 曲线在 α=1.0 附近最优 → 说明参数选择合理

---

## 六、实验执行顺序建议

```
1. B0   （必须最先跑，是所有对比的基准）
2. C2   （核心方法，尽早验证可行性）
3. C1   （对比固定阈值，证明自适应的优势）
4. A-α1/α3/α4  （α 消融，复用 C2 的 β=0.9）
5. A-β1/β3     （β 消融，复用 C2 的 α=1.0）
```
