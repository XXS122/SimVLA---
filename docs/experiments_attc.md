# ATTC 推理加速实验记录（单卡 · libero_goal）

**模型检查点**：`./runs/simvla_libero_small/step_200000`（按实际路径改）
**当前阶段**：只测 `libero_goal` 一个任务集；效果好再扩展到其它 3 个套件
**硬件**：单卡。服务端（策略模型）和客户端（LIBERO 仿真）共用 GPU0
**评估轮数**：20 trials（测试阶段够用；正式发表用 50）

> 单卡 + 单任务集时，只有一个连接、无排队，**成功率和延迟可以同一次跑出来**，
> 不需要像多卡那样分开测。

---

## 结果汇总表（边做边填）

### 主实验（libero_goal）

| 实验ID | 配置 | Goal SR% | avg 延迟(ms) | p50(ms) | p95(ms) | Hit Rate | 加速比 |
|--------|------|:--------:|:-----------:|:-------:|:-------:|:--------:|:------:|
| B0 | Baseline（无缓存） | | | | | — | 1.00× |
| C1 | Fixed-τ=0.02 | | | | | | |
| **C2** | **ATTC α=1.0 β=0.9** | | | | | | |

> 加速比 = `B0_avg延迟 / 本行avg延迟`。先填 B0，再算其它行。

### Ablation 1：α 敏感性（β=0.9 固定）

| 实验ID | α | Goal SR% | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|:--------:|:-----------:|:--------:|------|
| A-α1 | 0.5 | | | | 缓存最激进 |
| A-α2 | **1.0** | | | | **=C2，复用** |
| A-α3 | 1.5 | | | | |
| A-α4 | 2.0 | | | | 缓存最保守 |

### Ablation 2：β 敏感性（α=1.0 固定）

| 实验ID | β | 时间窗口 | Goal SR% | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|---------|:--------:|:-----------:|:--------:|------|
| A-β1 | 0.7 | ~3帧 | | | | EMA 反应快 |
| A-β2 | **0.9** | ~10帧 | | | | **=C2，复用** |
| A-β3 | 0.99 | ~100帧 | | | | EMA 反应慢 |

---

## 通用运行方式

每个实验两个终端：先开服务端（终端1），等它 listening 后再跑客户端（终端2）。

```bash
# ---------- 终端1：启动服务（替换 [端口] 和 [实验参数]）----------
source paths.env
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port [端口] [实验参数]
# 等日志出现 "SimVLA server listening on 0.0.0.0:[端口]" 再开终端2

# ---------- 终端2：跑 libero_goal 评估 ----------
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port [端口] --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > [结果文件].txt 2>&1
```

跑完后读取三类数据：

```bash
# 成功率 → 看客户端结果文件
grep -i "Total success rate" [结果文件].txt

# 延迟 + 命中率 → 看【服务端终端】日志（客户端断开时打印）
#   [Latency summary] n=XX  avg=XXXms  p50=XXXms  p95=XXXms
#   [ATTC] cumulative hit rate: XX.X%
```

---

## 一、主实验

### B0 — Baseline（无缓存）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8100

# 终端2
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port 8100 --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > exp_B0_goal.txt 2>&1
```

### C2 — ATTC-ours（α=1.0，β=0.9，推荐配置）

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8102 --use_cache --cache_alpha 1.0 --cache_beta 0.9 --cache_warmup 3

# 终端2
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port 8102 --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > exp_C2_goal.txt 2>&1
```

### C1 — Fixed-τ（固定阈值，对标 VLA-Cache）

> 真正的固定阈值用 `--cache_fixed_threshold`。**公平对比方法**：先跑完 C2 拿到它的
> hit rate，再调这里的阈值让 C1 的 hit rate 与 C2 接近（同等加速比），然后比成功率。
> 起始用 `0.02`，hit rate 太低就调大、太高就调小。

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8101 --use_cache --cache_fixed_threshold 0.02 --cache_warmup 3

# 终端2
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port 8101 --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > exp_C1_goal.txt 2>&1
```

---

## 二、Ablation 1：α 敏感性（β=0.9 固定）

只改服务端的 `--cache_alpha`，客户端命令同上（换端口和结果文件名）。

```bash
# A-α1  α=0.5
... --port 8110 --use_cache --cache_alpha 0.5 --cache_beta 0.9    # 结果存 exp_A_alpha05_goal.txt

# A-α2  α=1.0  —— 与 C2 相同，直接复用 C2 结果，不用再跑

# A-α3  α=1.5
... --port 8111 --use_cache --cache_alpha 1.5 --cache_beta 0.9    # 结果存 exp_A_alpha15_goal.txt

# A-α4  α=2.0
... --port 8112 --use_cache --cache_alpha 2.0 --cache_beta 0.9    # 结果存 exp_A_alpha20_goal.txt
```

完整命令示例（A-α3）：
```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/simvla_libero_small/step_200000 \
  --norm_stats ./norm_stats/libero_norm.json \
  --port 8111 --use_cache --cache_alpha 1.5 --cache_beta 0.9

# 终端2
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port 8111 --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > exp_A_alpha15_goal.txt 2>&1
```

---

## 三、Ablation 2：β 敏感性（α=1.0 固定）

只改服务端的 `--cache_beta`：

```bash
# A-β1  β=0.7
... --port 8120 --use_cache --cache_alpha 1.0 --cache_beta 0.7    # 结果存 exp_A_beta07_goal.txt

# A-β2  β=0.9  —— 与 C2 相同，直接复用 C2 结果，不用再跑

# A-β3  β=0.99
... --port 8121 --use_cache --cache_alpha 1.0 --cache_beta 0.99   # 结果存 exp_A_beta099_goal.txt
```

---

## 四、实验成功判断标准

| 指标 | 最低要求 | 理想值 |
|------|---------|--------|
| C2 Goal SR% | ≥ B0 × 95% | ≥ B0 × 98% |
| C2 avg 延迟 | ≤ B0 × 70% | ≤ B0 × 60% |
| C2 Hit Rate | ≥ 40% | 50–70% |
| C2 vs C1 SR% | C2 ≥ C1 | C2 明显 > C1 |

**核心结论（写论文用）**：
- C2 ≈ B0 成功率 → 缓存不掉精度
- C2 延迟 << B0 → 推理加速有效
- C2 > C1 → 自适应阈值优于固定阈值（创新点成立）
- α/β 消融在推荐值附近最优 → 参数选择合理

---

## 五、执行顺序建议

```
1. B0   （基准，最先跑）
2. C2   （核心方法，尽早验证可行性）
3. C1   （固定阈值对比，τ 调到与 C2 同等命中率）
4. A-α1/α3/α4  （α 消融，β 固定 0.9）
5. A-β1/β3     （β 消融，α 固定 1.0）
```

> libero_goal 上 C2 若达标（成功率几乎不掉 + 明显加速），再把上述实验
> 扩展到 `libero_spatial / libero_object / libero_10`（命令里 `--task_suite` 换掉即可）。
