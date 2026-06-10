# TDS：转变密度采样（Transition-Density Sampling）

> 配套实现：`transition_density_stats.py`（实验 0 离线统计）、
> `datasets/tds_sampling.py` + `datasets/dataset_smolvlm.py`（训练侧流式集成）、
> `evaluation/libero/libero_client.py --per_task_csv`（逐任务 SR 导出）。

---

## 1. 动机：问数据，而不是问模型

Re-Mix / AdaMix 一类难度感知混合方法回答"哪个任务难"的方式都是**问模型**：
loss 高、成功率低的任务就是难的。两个内生缺陷：

1. 需要 reference model 或训练中周期性评估——多一条评估回路，开销大、引入额外超参；
2. 难度估计与被训练的模型耦合——换 backbone 难度排序就变，结论不可迁移。

TDS 反过来**问数据本身**：一个操作任务难不难，在 demo 被采集出来的那一刻就已经写在
动作信号里了。三个数据内在分量：

| 分量 | 含义 | 对应难度机制 |
|---|---|---|
| `E_events` | 平均 gripper 开/合事件数 | 子任务越多越难（抓取/放置事件密度） |
| `P_plateau` | 低速精细对准段占比 | 容错窗口越窄越难 |
| `L_length` | 平均轨迹长度 | 误差累积空间越大越难 |

三者**训练前离线扫一遍即可**：零训练开销、与模型无关、完全可复现。

真正把它从"采样 trick"升级为创新点的，是它与前两个组件共享同一个信号：
步级加权用的变化率、BGE 触发用的低速平台、TDS 难度分数里的平台占比——三者是
**同一个量在三个时间尺度上的读出**（单步 → 单 chunk → 整个任务）。论文论点：
*转变结构是操作任务的内在难度坐标系，它同时回答了"哪一步重要、何时该闭环、
哪个任务该多学"三个问题*。loss-based 难度信号在步级和执行级没有对应物，给不出
这个叙事。

为保证三个尺度严格一致，`transition_density_stats.py` 顶部的
`PLATEAU_THRESH = 0.25` / `PLATEAU_MIN_LEN = 3` 与 BGE 触发条件使用同一组常数。

## 2. 难度分数与采样权重

对每个任务 hdf5（≈50 条 demo）：

1. **逐维归一化变化率**（`per_dim_normalized_change_rate`）：先逐维差分、每维除以
   自己在 episode 内的最大幅度、再**排除 gripper 维**取均值——修复 gripper 阶跃
   支配 L2 范数的问题；gripper 单独走事件通道（符号翻转计数，对 0/1 与 -1/1
   编码都鲁棒）。
2. 三分量跨任务 z-score 后加权求和得 `d_k`（默认 α=β=γ=1/3，由实验 0b 决定最终形态）。
3. 温度化 softmax 得 `p_k`（softmax 前先平移到正区间，避免负数开 1/T 次幂出 NaN）。
4. **下限 clip**（默认 0.5 × uniform）+ 重归一化——保护简单任务不被灾难性欠采样。

## 3. 实验设计：每张表在防什么质疑

| 实验 | 防的质疑 |
|---|---|
| **实验 0（假设验证）** | "难度定义是拍脑袋的"。用已有 uniform baseline 对 40 个任务逐个评 SR，与离线 `d_k` 做 Spearman 检验；ρ < −0.4 且 p < 0.05 则假设成立（PASS），放进 motivation。0b 看各分量判别力，0c 看分量共线性，决定公式最终形态 |
| **主表递进（行1→2→3→5）** | "贡献不清"。每加一个粒度涨多少一目了然 |
| **行 7：loss-based 复现** | 生死线。TDS 打平或超过即赢——零开销达到同等效果，省掉整条评估回路 |
| **难度反转消融** | "任何非 uniform 扰动都可能有效"（正则化效应）。权重反转后性能应下降，证明方向性本身携带信息 |
| **无 clip 对照** | "简单任务灾难性遗忘"。预期无 clip 时 Spatial/Object 掉、有 clip 不掉——展示对失败模式的掌控 |
| **交叉实验** | "三个 trick 拼盘"。TDS 训出的模型边界头 AUROC 是否更高、BGE 增益是否更大——难任务的转变样本被多采样，边界头见的转变更多。若成立，三个粒度存在协同，"统一框架"有定量证据 |
| **课程式消融** | uniform→TDS / TDS→uniform 双向退火（`tds_sampling.mix_with_uniform` 生成中间权重），主方案用静态版 |

## 4. 工作流

### 第一步：离线统计（今天就能跑，无需训练）

```bash
source paths.env
python transition_density_stats.py \
    --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --temperature 2.0 --clip_rho 0.5 \
    --out task_difficulty.csv
```

Sanity check：Top-5 难任务应该被 LIBERO-Long（libero_10）霸榜——这本身就是一个
直觉校验。

### 第二步：导出 baseline 逐任务 SR（实验 0a）

用已有 uniform baseline 跑评估，每个 suite 单独一个 csv（并行评估时避免写竞争），
然后拼接：

```bash
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py --port 8102 \
    --task_suite libero_goal --num_trials 20 \
    --per_task_csv sr_goal.csv
# ... 其余 3 个 suite 同理 ...
awk 'FNR==1 && NR!=1 {next} {print}' sr_*.csv > baseline_per_task_sr.csv
```

csv 的 `task_name` 用 `<task.name>_demo`（即 hdf5 文件名去后缀），与统计脚本的
key 严格一致，可直接对接。

### 第三步：假设检验（实验 0a/0b/0c）

```bash
python transition_density_stats.py \
    --data_root $LIBERO_DATASETS \
    --sr_csv baseline_per_task_sr.csv
# 输出 Spearman 检验 PASS/FAIL、各分量判别力、分量相关矩阵、density_vs_sr.png
```

### 第四步：实验 0 通过后，开启 TDS 训练

```bash
TDS_WEIGHTS_CSV=./task_difficulty.csv bash train_smolvlm_small.sh
# 或直接: python train_smolvlm.py ... --tds_weights_csv ./task_difficulty.csv
```

消融变体只需换 csv：难度反转（对 `d_k` 取负重算 p_k）、无 clip（`--clip_rho 0`）、
课程式（`mix_with_uniform` 生成中间权重）。

## 5. 训练侧集成的设计决策

**为什么不用 `WeightedRandomSampler`（原方案 A）**：SimVLA 的
`SmolVLMDataReader` 是无限流式 `IterableDataset`，PyTorch 的 sampler 只支持
map-style dataset。改走方案 B——**两级采样**：每个 yield 先按 `p_k` 抽任务文件，
再从该任务的常驻 episode 流里取下一个样本（`SmolVLMDataReader._iter_tds`）。

由于加权抽样发生在**每个样本**粒度（而非每个文件/episode），任务 k 占训练样本的
比例精确收敛到 `p_k`，无需做 map-style 方案里的 `p_k / n_k` 权重摊派，且天然兼容
demo 数或轨迹长度不一致的数据集。

其余要点：

- **只动采样，不碰模型/损失**——与步级加权完全正交，消融开关互不污染；
  `--tds_weights_csv` 不传时走原 uniform 代码路径，逐 bit 向后兼容。
- **静默 bug 防护**：`SamplingMonitor` 在 worker 0 训练前 5000 个样本后自动打印
  实际采样频率 vs 目标 `p_k`，最大偏差超 2% 给出大写 WARNING（采样 bug 不会
  crash，不查可能训完两周才发现 TDS 没生效）。
- **csv 没覆盖的任务文件**（例如 meta 里含 libero_90 但统计只算了 4 个 suite）
  回退到中位数 `p_k` 并打印告警，按"平均难度任务"采样。
- **内存代价**：每个任务文件保持一个挂起的 episode 生成器（约一条 demo 的解码
  数组，~10–20 MB），130 个任务文件 × 4 个 dataloader worker 约 5–10 GB 主存；
  只训 4 个 suite（40 文件）时约 2.5 GB。
- 连续 3 轮读不出样本的坏文件会被踢出采样池并重归一化权重，不会拖死训练。
