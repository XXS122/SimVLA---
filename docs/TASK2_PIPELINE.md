# 任务二：能量校验的一步流策略与推理时扩展定律

> Energy-Verified One-Step Flow Policies and Inference-Time Scaling Laws for Flow-Based VLAs

本文档描述任务二的完整实验管线：把 SimVLA 的 10 步 Flow Matching 动作头自蒸馏为
**MeanFlow 一步生成器**（A100 / SAPI 站点），异步训练一个共享 VLM 特征的
**动作能量校验器**（A800 / YYK 站点），部署时单卡并行采样 K 个一步候选、按能量选优
（Best-of-K），并测量成功率随 K 的**推理时扩展定律**。

## 方法概要

**MeanFlow 蒸馏。** 原始 SimVLA 训练瞬时速度场 v(x_t, t)，推理需 10 步 Euler 积分。
我们训练平均速度场 u(x_t, r, t)，满足 (t−r)·u = ∫ᵣᵗ v dτ，其自洽恒等式

```
u = v_t − (t − r) · d/dt u,    d/dt u = v_t·∇ₓu + ∂ₜu
```

用一次前向模式 JVP 计算全导数，目标做 stop-gradient（`models/modeling_smolvlm_vla.py`
的 `forward_meanflow`）。区间条件 (t−r) 通过**零初始化**投影注入
（`models/transformer_smolvlm.py` 的 `interval_proj`），因此学生在 r=t 处与教师逐位等价，
可直接从 SFT 权重热启动。r=t 的样本占比 (1−meanflow_ratio)，此时目标退化为普通 Flow
Matching，起稳定作用。一步生成：x₀ = x₁ − u(x₁, 0, 1)，保留噪声条件以维持多模态。

**能量校验器。** 轻量 Transformer（约 10M 参数，`models/verifier.py`），输入冻结 SFT
骨干的 VLM 特征（M 个可学习查询做注意力池化）、本体感受与归一化动作块，输出标量能量。
InfoNCE 训练：正样本 = 数据集动作；负样本 = 一步生成器自身的提议分布（跨站点运来的
checkpoint 采样）+ 高斯扰动 GT + 批内乱序 GT；与 GT 距离小于 `min_neg_dist` 的生成器
提议按假阴性掩掉。

**Best-of-K 推理。** VLM 前向一次；动作头以 batch=K 并行做一步生成；校验器一次前向
打分取 argmin 能量。理论部分（论文写作时）：argmin-能量的 best-of-K 等价于对基策略的
KL 正则化策略改进；预注册假设为误差 err(K) = a·K^(−b) + c 且 R² ≥ 0.9。

## 双站点分工与传输物

| 阶段 | 站点 | 产物 | 传输 |
|---|---|---|---|
| 0. SFT 教师 | A100 (SAPI) | `task2_sft/ckpt-200000` (~2GB) | → A800（对象存储/人工拷贝） |
| 1. MeanFlow 蒸馏 | A100 (SAPI) | `task2_meanflow/ckpt-50000` | → A800 |
| 2. 校验器 | A800 (YYK) | `task2_verifier/ckpt-50000` (~40MB) | → A100 |
| 3. Best-of-K 评估 + 扩展定律 | 任一单卡 | CSV / 拟合 JSON / 图 | — |

站点间**只搬运 checkpoint**，无梯度通信。阶段 1 与阶段 2 可异步并行迭代：A800 收到更新
的生成器 checkpoint 后重跑（或续训）校验器即可（异步对抗共训练）。

## 操作步骤

每台机器先准备本机的 `paths.env`（模板见 `paths.env.example`，已被 .gitignore 忽略），
然后 `source paths.env`。

### 阶段 0（A100）：SFT 教师

```bash
source paths.env
bash scripts/task2_train_sft.sh 64 "$SIMVLA_CHECKPOINTS/task2_sft"
```

约 3–4 天（单卡 80GB、batch 64、200k iters）。产物中的最终 ckpt 同时拷贝一份到 A800。

### 阶段 1（A100）：MeanFlow 一步蒸馏

```bash
source paths.env
bash scripts/task2_train_meanflow.sh "$SIMVLA_CHECKPOINTS/task2_sft/ckpt-200000"
```

要点：VLM 骨干全程冻结（特征 detach），只训动作头；JVP 需要数学版注意力，脚本自动关闭
fused SDPA；混合精度关闭（JVP 数值稳定性优先）。约 1–2 天 / 50k iters。可选
`EXTRA_ARGS="--teacher_velocity"` 用冻结教师头的 v(x_t,t) 替代条件速度做蒸馏目标。

### 阶段 2（A800）：能量校验器

```bash
source paths.env
bash scripts/task2_train_verifier.sh \
    /path/to/shipped/task2_sft/ckpt-200000 \
    /path/to/shipped/task2_meanflow/ckpt-50000
```

约 1 天 / 50k iters。生成器 checkpoint 每次更新后重跑本脚本即完成一轮异步共训练。

### 阶段 3：Best-of-K 评估与扩展定律

需要 `simvla` 与 `libero` 两个 conda 环境（见 `evaluation/libero/README.md`）。

```bash
cd evaluation/libero
source ../../paths.env
TEACHER_CKPT=/path/to/task2_sft/ckpt-200000 \
bash run_scaling_eval.sh \
    /path/to/task2_meanflow/ckpt-50000 \
    /path/to/task2_verifier/ckpt-50000 \
    libero_spatial 10 8102

python fit_scaling_law.py --csv scaling_results/scaling_results.csv \
    --out_json scaling_results/scaling_fit.json \
    --out_plot scaling_results/scaling_fit.png
```

sweep 自动跑三组：教师 10 步 flow 基线、一步生成器（K=1）、Best-of-K（K∈{2,4,8,16,32}）。
四个套件依次换 `libero_spatial/libero_object/libero_goal/libero_10`；正式数字用
`num_trials=50`。

也可单独起服务器手动评测：

```bash
python serve_smolvlm_libero.py --checkpoint <generator> \
    --norm_stats ../../norm_stats/libero_norm.json \
    --mode bok --num_samples 16 --verifier <verifier> \
    --latency_log lat.jsonl --port 8102
```

## 预注册的可证伪结论

1. **Pareto**：K=16 的能量选优一步策略，延迟比 10 步基线低 ≥5×，且四套件平均成功率
   不低于基线 −0 且目标 +3 点；若成功率显著低于基线，方法失败。
2. **扩展定律**：bok 各套件 err(K) 幂律拟合 R² ≥ 0.9（`fit_scaling_law.py` 自动判定
   SUPPORTED/REJECTED）；R² < 0.9 即推翻扩展定律假设。
3. **校验器有效性**：K 相同的条件下，bok 成功率应单调不低于 onestep（K=1）；
   若 bok(K) ≤ onestep 恒成立，说明能量函数没有学到有效排序，假设被证伪。

## 消融清单（论文实验节）

- meanflow_ratio ∈ {0.25, 0.5, 0.75}
- 条件速度 vs `--teacher_velocity`
- 负样本来源消融：仅扰动 / 仅生成器 / 全部
- 校验器规模：hidden 256/384/512, depth 2/4/6
- 一步 vs 2 步 vs 4 步 MeanFlow 积分（steps 参数）下的 Pareto 前沿
