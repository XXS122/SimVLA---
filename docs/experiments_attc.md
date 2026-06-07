# ATTC 推理加速实验记录

**模型检查点**：`./runs/simvla_libero_small/step_200000`（根据实际路径修改）
**评估轮数**：每个任务集 20 trials（快速验证），正式发表用 50 trials
**GPU 分配**：服务端占 1 块，评估端 4 块并行（4 个任务集）

---

## 运行方式说明

每个实验分两步：先开服务端（终端1），再跑评估（终端2）。

```bash
# 终端1：启动推理服务（替换下方 [实验参数]）
source paths.env
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port [端口] [实验参数]
# 等待日志出现 "SimVLA server listening on 0.0.0.0:XXXX" 再执行终端2

# 终端2：跑评估（替换端口和结果前缀）
cd evaluation/libero
bash run_eval_all.sh [端口] 20 [结果前缀] "0 1 2 3"

# 查看结果
grep -E "success|Success|Total" [结果前缀]_*.txt
```

从服务端日志读取延迟（连接关闭时打印）：
```
[Latency summary] n=XX  avg=XXXms  p50=XXXms  p95=XXXms
[ATTC] cumulative hit rate: XX.X%
```

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
bash run_eval_all.sh 8100 20 exp_B0 "0 1 2 3"
```

---

### C1 — Fixed-τ（固定阈值，对标 VLA-Cache）

> 用极大的 β（0.999）和极小的 α（0.01）模拟固定阈值：EMA 几乎不更新，τ ≈ 初始值 0.1。

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8101 \
  --use_cache \
  --cache_alpha 0.01 \
  --cache_beta 0.999 \
  --cache_warmup 1

# 终端2
cd evaluation/libero
bash run_eval_all.sh 8101 20 exp_C1 "0 1 2 3"
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
bash run_eval_all.sh 8102 20 exp_C2 "0 1 2 3"
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
bash run_eval_all.sh 8110 20 exp_A_alpha05 "0 1 2 3"
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
bash run_eval_all.sh 8111 20 exp_A_alpha15 "0 1 2 3"
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
bash run_eval_all.sh 8112 20 exp_A_alpha20 "0 1 2 3"
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
bash run_eval_all.sh 8120 20 exp_A_beta07 "0 1 2 3"
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
bash run_eval_all.sh 8121 20 exp_A_beta099 "0 1 2 3"
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
