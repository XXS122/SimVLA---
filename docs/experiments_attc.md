# ATTC 推理加速实验记录（单卡 · libero_goal）

**模型检查点**：`$SIMVLA_CHECKPOINTS`（从 `paths.env` 读取，按实际路径改）
**当前阶段**：先测 `libero_goal` 一个任务集；效果稳定再扩展到其它 3 个套件
**硬件**：单卡。服务端（策略模型）和客户端（LIBERO 仿真）共用 GPU0
**评估轮数**：测试阶段 5–20 trials；正式发表用 50

> 单卡 + 单任务集时，只有一个连接、无排队，**成功率、轨迹步数、延迟可以同一次跑出来**，
> 不需要像多卡那样分开测。

---

## 0. 实现说明 / 踩坑记录（重要）

ATTC 是**纯推理侧**改动（无需重新训练），核心在 `models/modeling_smolvlm_vla.py`
的 `forward_vlm_with_cache()`：按 camera view 计算当前帧与"上次编码帧"的 L1 像素差，
自适应阈值 `τ = EMA_mean + α·EMA_std` 决定该 view 复用缓存特征还是重新过 SigLIP。

落地过程中修了两个**会导致成功率崩溃**的 bug，记录如下：

### Bug 1 — `ema_std` 初值过大（commit `850cbe0`）
- **现象**：缓存一开就几乎全命中（hit rate ≈ 85%），成功率从 95% 崩到 3.5%。
- **根因**：`ema_std` 初值设成 `0.1`，而 LIBERO 经 ImageNet 归一化后的真实帧差只有
  `~0.04–0.07`。导致阈值 `τ ≈ 0 + 1.0×0.1 = 0.10` 远高于真实帧差 → 几乎每帧都判"没变化"
  → 一直复用第 3 步的陈旧特征。
- **修复**：`ema_std` 初值改 `0.0`；并让 **warmup 期也更新 EMA**（之前只在 warmup 之后更新），
  这样开始缓存时统计量已校准到真实帧差量级，阈值从低位（≈EMA_mean）随数据自然上升。

### Bug 2 — 参考帧"漂移"（commit `e085f3a`）
- **现象**：即使阈值修好，缓慢运动时一个 view 仍被无限期命中，轨迹质量下降。
- **根因**：`prev_pixel_values` 每步都被覆盖成"当前帧"，于是 diff 测的是**相邻帧**之差。
  机械臂缓慢移动时每步差都 < τ，该 view 一直命中，但复用的特征是很多帧之前算的，
  真实场景早已**累积漂移**远离缓存特征 → 模型用严重陈旧的视觉信息出动作。
- **修复**：参考帧**只在该 view 被重新编码时刷新**；命中的 view 保留旧参考帧，
  使 diff 测"自上次编码以来的累积漂移"。漂移超阈值才重编码并重置参考帧，
  从根本上限制任何复用特征的陈旧程度。

> 经验：像素差缓存的阈值必须按**输入的实际数值范围**校准（ImageNet 归一化后 ~[-2,2]，
> 帧差 ~0.05）；且"判断基准"必须是缓存特征对应的那一帧，而不是上一帧。

---

## 一、结果汇总表（边做边填）

### 主实验（libero_goal）

| 实验ID | 配置 | Goal SR% | 平均步数 | avg 延迟(ms) | p50(ms) | p95(ms) | Hit Rate | 加速比 |
|--------|------|:--------:|:-------:|:-----------:|:-------:|:-------:|:--------:|:------:|
| B0 | Baseline（无缓存） | | | | | | — | 1.00× |
| C1 | Fixed-τ=0.10 | | | | | | | |
| **C2** | **ATTC α=1.0 β=0.9** | | | | | | | |

> 加速比 = `B0_avg延迟 / 本行avg延迟`。先填 B0，再算其它行。
> **平均步数越接近 baseline 越好** —— 步数暴涨说明缓存损害了动作精度（见第三节）。

#### 已验证的初步数据（task 0 单任务, 5 trials）

| 配置 | Goal SR% | 平均步数 | avg 延迟(ms) | Hit Rate | 加速比 |
|------|:--------:|:-------:|:-----------:|:--------:|:------:|
| Baseline | 100% (5/5) | ~118（116–131，极稳定） | 166.2 | — | 1.00× |
| ATTC α=1.0 | 100% (5/5) | ~414（190–675，波动大） | 140.6 | 84.7% | 1.18× |

> 关键观察：**成功率都满分，但 ATTC 的轨迹步数是 baseline 的 ~3.5 倍**。详见第三节。

### Ablation 1：α 敏感性（β=0.9 固定）

| 实验ID | α | Goal SR% | 平均步数 | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|:--------:|:-------:|:-----------:|:--------:|------|
| A-α1 | 0.5 | | | | | 缓存最保守 |
| A-α2 | **1.0** | | | | | **=C2，复用** |
| A-α3 | 1.5 | | | | | |
| A-α4 | 2.0 | | | | | 缓存最激进 |

> 注：α 越大 → 阈值越高 → 越容易判"没变化" → 命中越多、越激进（缓存越陈旧）。

### Ablation 2：β 敏感性（α=1.0 固定）

| 实验ID | β | 时间窗口 | Goal SR% | 平均步数 | avg 延迟(ms) | Hit Rate | 备注 |
|--------|---|---------|:--------:|:-------:|:-----------:|:--------:|------|
| A-β1 | 0.7 | ~3帧 | | | | | EMA 反应快 |
| A-β2 | **0.9** | ~10帧 | | | | | **=C2，复用** |
| A-β3 | 0.99 | ~100帧 | | | | | EMA 反应慢 |

---

## 二、通用运行方式

每个实验两个终端：先开服务端（终端1，base 环境），等它 listening 后再跑客户端（终端2，libero 环境）。

```bash
# ---------- 终端1：启动服务（base 环境，替换 [端口] 和 [实验参数]）----------
source paths.env
conda activate base
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint "$SIMVLA_CHECKPOINTS" \
  --norm_stats ./norm_stats/libero_norm.json \
  --smolvlm_model "$SIMVLA_SMOLVLM_MODEL" \
  --port [端口] [实验参数]
# 等日志出现 "SimVLA server listening on 0.0.0.0:[端口]" 再开终端2

# ---------- 终端2：跑 libero_goal 评估（libero 环境）----------
conda activate libero
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
  --host 127.0.0.1 --port [端口] --client_type websocket \
  --task_suite libero_goal --num_trials 20 --no_video \
  > [结果文件].txt 2>&1

# 只测单个 task（快速验证）：加 --task_id 0
```

跑完后读取四类数据：

```bash
# 成功率 + 轨迹步数 → 看客户端结果文件
grep -i "Total success rate"        [结果文件].txt
grep    "Average steps per episode" [结果文件].txt

# 延迟 + 命中率 → 看【服务端终端】日志（客户端断开时打印）
#   [Latency summary] n=XX  avg=XXXms  p50=XXXms  p95=XXXms
#   Connection closed — ATTC session stats: hit rate=XX.X%
```

---

## 三、关键发现与局限（写论文必读）

### 发现 1：延迟只省 ~15%，远低于命中率（84.7%）的直觉
缓存只跳过了**视觉编码（SigLIP）**，但 `text_model` 视觉-语言融合 + 动作 ODE（默认 10 步）
**每个控制步都照常跑**。84.7% 命中只换来 166→140ms（1.18×）的加速，说明
**视觉编码只占总推理时间约 18%**，大头在 text_model 与 ODE。
→ 想要更大加速，必须进一步缓存/跳过 text_model 或减少 ODE 步数，那是更难的方向。

### 发现 2：激进缓存不掉成功率，但显著拉长轨迹（速度/质量权衡）
task 0 上 baseline 步数极稳定（116–131），而 ATTC α=1.0 波动巨大（190–675），平均 ~3.5×。
复用的陈旧视觉特征降低了动作精度，机械臂"绕路/磨蹭"，在简单任务上还能完成，
**但在更难的套件上这种退化很可能直接变成失败**。
→ α=1.0 偏激进。后续可调低 α（命中率↓、特征更新鲜、轨迹更短），在
"加速比"与"轨迹步数"之间找甜点；这正是 α 消融要回答的问题。

---

## 四、各实验命令

> 服务端命令统一用第二节模板，下面只列**各实验的 `[实验参数]` 和端口/结果文件**。
> 客户端命令除端口/结果文件外完全一致（`--task_suite libero_goal --num_trials 20 --no_video`）。

### 主实验
| 实验 | 端口 | 服务端 `[实验参数]` | 结果文件 |
|------|:----:|--------------------|----------|
| B0 Baseline | 8100 | （无） | `exp_B0_goal.txt` |
| C2 ATTC | 8102 | `--use_cache --cache_alpha 1.0 --cache_beta 0.9 --cache_warmup 3` | `exp_C2_goal.txt` |
| C1 Fixed-τ | 8101 | `--use_cache --cache_fixed_threshold 0.10 --cache_warmup 3` | `exp_C1_goal.txt` |

> C1 公平对比：先跑 C2 拿到 hit rate，再调 `--cache_fixed_threshold` 让 C1 命中率与 C2 接近
> （同等加速比），然后比成功率和轨迹步数。起始用 `0.10`，命中太低就调大、太高就调小。
> （注：旧值 `0.02` 对 ImageNet 归一化图像太低，会 0% 命中、退化成 baseline。）

### Ablation 1：α 敏感性（β=0.9 固定）
| 实验 | 端口 | `[实验参数]` | 结果文件 |
|------|:----:|-------------|----------|
| A-α1 | 8110 | `--use_cache --cache_alpha 0.5 --cache_beta 0.9 --cache_warmup 3` | `exp_A_alpha05_goal.txt` |
| A-α2 | — | =C2，复用，不用再跑 | — |
| A-α3 | 8111 | `--use_cache --cache_alpha 1.5 --cache_beta 0.9 --cache_warmup 3` | `exp_A_alpha15_goal.txt` |
| A-α4 | 8112 | `--use_cache --cache_alpha 2.0 --cache_beta 0.9 --cache_warmup 3` | `exp_A_alpha20_goal.txt` |

### Ablation 2：β 敏感性（α=1.0 固定）
| 实验 | 端口 | `[实验参数]` | 结果文件 |
|------|:----:|-------------|----------|
| A-β1 | 8120 | `--use_cache --cache_alpha 1.0 --cache_beta 0.7 --cache_warmup 3` | `exp_A_beta07_goal.txt` |
| A-β2 | — | =C2，复用，不用再跑 | — |
| A-β3 | 8121 | `--use_cache --cache_alpha 1.0 --cache_beta 0.99 --cache_warmup 3` | `exp_A_beta099_goal.txt` |

> **一键串行跑全部**（单卡过夜）：`source paths.env && bash evaluation/libero/run_eval_overnight.sh 20 0`
> 结果汇总在 `evaluation/libero/eval_overnight_<时间戳>/SUMMARY.txt`（含成功率/平均步数/延迟/命中率）。

---

## 五、实验成功判断标准

> 不再只看延迟——**成功率 + 轨迹步数**是一对必须同时看的指标。

| 指标 | 最低要求 | 理想值 |
|------|---------|--------|
| C2 Goal SR% | ≥ B0 × 95% | ≥ B0 × 98% |
| C2 平均步数 | ≤ B0 × 1.5 | ≈ B0（≤ B0 × 1.2） |
| C2 avg 延迟 | < B0 | ≤ B0 × 85% |
| C2 Hit Rate | ≥ 40% | 50–70%（过高会拉长轨迹） |
| C2 vs C1 | 同命中率下 C2 SR/步数 ≥ C1 | C2 明显更优 |

**核心结论（写论文用）**：
- C2 ≈ B0 成功率 **且** 步数接近 → 缓存不掉精度（仅省延迟才算干净的加速）
- C2 延迟 < B0 → 推理加速有效（但注意上限 ~15%，见第三节）
- C2 优于 C1（同命中率）→ 自适应阈值优于固定阈值（创新点成立）
- α/β 消融找到"加速 vs 轨迹质量"甜点 → 参数选择合理

---

## 六、执行顺序建议

```
1. B0   （基准，最先跑：记下 SR / 平均步数 / 延迟）
2. C2   （核心方法，尽早验证：对比 B0 的步数是否暴涨）
3. C1   （固定阈值对比，τ 调到与 C2 同等命中率）
4. A-α1/α3/α4  （α 消融，β 固定 0.9，找速度/步数甜点）
5. A-β1/β3     （β 消融，α 固定 1.0）
```

> libero_goal 上 C2 若达标（成功率不掉 + 步数接近 baseline + 有加速），再把上述实验
> 扩展到 `libero_spatial / libero_object / libero_10`（命令里 `--task_suite` 换掉即可）。
