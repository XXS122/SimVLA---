# 实验设计：自适应动作分块（Adaptive Action Chunking via Change-Rate Weighting）

> 本文档配套 `models/transformer_smolvlm.py` / `models/modeling_smolvlm_vla.py` 中实现的
> 变化率加权损失（CRW）与边界预测头（Boundary Head）。

---

## 1. 实验目标（要回答的科学问题）

| 编号 | 问题 | 验证方式 |
|---|---|---|
| **Q1** | 变化率加权能否提升整体任务成功率？ | 主实验：与固定 MSE baseline 对比 LIBERO 四套件 SR |
| **Q2** | 提升是否主要来自"接触/转向"等高变化关键步？ | 分段分析：高变化步 vs 平滑步的动作误差 |
| **Q3** | 边界预测头学到的分数是否真的对应物理上的关键时刻？ | 可视化 boundary 分数 vs 真实接触事件 |
| **Q4** | 各组件分别贡献多少？ | 消融：仅加权 / 仅边界头 / 二者结合 |
| **Q5** | 对动作平滑度有无副作用？ | 测 jerk（三阶差分），确认没有牺牲平滑性 |

---

## 2. 基准与指标

### 2.1 Benchmark：LIBERO

| Task Suite | 侧重点 | 说明 |
|---|---|---|
| **LIBERO-Spatial** | 空间布局泛化 | 同物体不同摆放 |
| **LIBERO-Object** | 物体泛化 | 同布局不同物体 |
| **LIBERO-Goal** | 目标/技能泛化 | 同场景不同任务目标 |
| **LIBERO-Long (libero_10)** | 长时程 | 多阶段任务，**最能体现关键步加权的价值** |

### 2.2 主指标

- **Success Rate (SR)**：每个 suite 在 seeds `{0,1,2,3}` × `50 episodes` 下的平均成功率，报告 `mean ± std`
- **Avg SR**：四套件平均

### 2.3 诊断指标（用于回答 Q2/Q3/Q5）

| 指标 | 定义 | 回答 |
|---|---|---|
| **High-Δ MSE** | 真值变化率前 20% 的时间步上的动作 MSE | Q2 |
| **Low-Δ MSE** | 后 50% 平滑步上的动作 MSE | Q2 |
| **Boundary AP** | boundary 分数对"接触事件标签"的平均精度 | Q3 |
| **Jerk** | 执行轨迹三阶差分均方根 `‖a_{t+1}-2a_t+a_{t-1}‖` | Q5 |

> 接触事件标签可由仿真器的 contact flag 或夹爪状态突变近似得到。

---

## 3. 对照组设置

固定其余所有超参（backbone、num_actions、lr schedule、训练步数），仅切换以下配置：

| 组别 | 配置 | 命令片段 |
|---|---|---|
| **B0 (Baseline)** | 固定 MSE，concat 模式 | 默认，不加 flag |
| **B1 (Baseline-AdaLN)** | 固定 MSE，AdaLN 模式 | `--use_adaln` |
| **A0 (Ours)** | CRW + Boundary，concat | `--use_adaptive_chunking` |
| **A1 (Ours-AdaLN)** | CRW + Boundary，AdaLN | `--use_adaln --use_adaptive_chunking` |

> 主对比：**B0 vs A0** 和 **B1 vs A1**（控制变量，只开/关自适应分块）。

---

## 4. 消融实验（回答 Q4）

在 concat 模式下，逐项拆解：

| 消融 | step_weights | boundary head | 实现方式 |
|---|---|---|---|
| **Ab-0** 全关（=B0） | 全 1.0 | 无 | 默认 |
| **Ab-1** 仅加权 | `[0.5,1.5]` | 无 | 代码里临时设 `chunk_loss_weight=0` 且保留加权 |
| **Ab-2** 仅边界头 | 全 1.0 | 有 | 临时改：加权关、boundary 开 |
| **Ab-3** 完整（=A0） | `[0.5,1.5]` | 有 | `--use_adaptive_chunking` |

### 4.1 加权区间敏感性

固定其余设置，扫 `step_weights` 的区间上下界：

| 设置 | 区间 | 含义 |
|---|---|---|
| W1 | `[1.0, 1.0]` | 无加权（对照） |
| W2 | `[0.75, 1.25]` | 弱加权 |
| W3 | `[0.5, 1.5]`  | **默认** |
| W4 | `[0.25, 1.75]` | 强加权 |
| W5 | `[0.0, 2.0]`  | 极端（平滑步几乎被忽略） |

> 预期是个倒 U 型：太弱无效果，太强会让模型在平滑段欠拟合、轨迹发飘。

### 4.2 边界损失权重敏感性

扫 `--chunk_loss_weight ∈ {0.0, 0.05, 0.1, 0.2, 0.5}`，看辅助任务权重对主任务 SR 的影响。

---

## 5. 训练协议

为保证可比性与控制成本：

- **统一训练步数**：所有组别训到相同 iters（如 150k），用同一 `train_smolvlm_small.sh` 设置
- **统一随机种子**：训练 seed 固定，差异只来自配置
- **每组评测 4 个推理 seed**：`{0,1,2,3}`，消除环境随机性
- **算力**：单卡即可复现小模型；主表用 small（768/12/12），如有资源补 large（1024/24/16）

### 5.1 复现命令

```bash
# Baseline (B0)
bash train_smolvlm_small.sh 32 1.0 ./runs/b0_baseline

# Ours (A0)
python train_smolvlm.py \
  --train_metas_path ./datasets/metas/libero_train.json \
  --norm_stats_path ./norm_stats/libero_norm.json \
  --action_mode libero_joint --num_actions 10 \
  --output_dir ./runs/a0_adaptive \
  --use_adaptive_chunking --chunk_loss_weight 0.1

# 评测（先起 server 再跑 client）
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
  --checkpoint ./runs/a0_adaptive/ckpt-150000 \
  --norm_stats ./norm_stats/libero_norm.json --port 8102
bash evaluation/libero/run_eval_all.sh 8102 50 "a0_adaptive" "0 1 2 3"
```

---

## 6. 预期结果表（论文主表骨架）

| Method | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| SimVLA (B0) | – | – | – | – | – |
| + Adaptive Chunking (A0) | – | – | – | – | **↑** |
| SimVLA-AdaLN (B1) | – | – | – | – | – |
| + Adaptive Chunking (A1) | – | – | – | – | **↑** |

**预期趋势**：
- LIBERO-Long 提升最明显（长时程任务里关键步占比高、错一步全盘皆输）
- High-Δ MSE 显著下降，Low-Δ MSE 略升但 SR 不降 → 证明"把误差从重要的地方挪到不重要的地方"是值得的
- Jerk 不增甚至略降 → 没有牺牲平滑性

---

## 7. 坑预警

1. **变化率被归一化方式影响**：`change_rate` 是在 *normalized* 动作空间上算的。若某些维度（如 gripper）归一化尺度和其余维度差异大，范数会被它主导。
   → **缓解**：必要时对 gripper 维单独处理，或用逐维标准化后再求范数。

2. **`num_actions` 太小**：当 chunk 长度只有 ~5 步时，加权差异不明显。
   → 建议在 `num_actions ≥ 10` 上验证。

3. **boundary head 训练初期梯度为 0**：head 末层初始化为 0（故意为之，保证初始不扰动主任务），前期 boundary_loss 下降慢属正常。

4. **公平性**：A0 比 B0 多了一个 head 的参数（很小）。若审稿人质疑，补一个"B0 + 同等参数量的 dummy head"对照。

5. **接触标签获取**：Q3 依赖接触事件标签，LIBERO 仿真器需确认能导出 contact / gripper 状态时间戳。
