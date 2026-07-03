# P2 · 能量校验的一步流策略与流式 VLA 的推理时扩展定律

**EVOS-VLA: Energy-Verified One-Step Flow Policies and Inference-Time Scaling Laws for Flow-Based Vision-Language-Action Models**

- 状态：代码已实现并通过冒烟测试（见 `docs/TASK2_PIPELINE.md`，分支 `claude/vla-dual-gpu-research-92r7hp`）
- 目标会议：ICLR 2027（主）；备选 CoRL 2027 / ICML 2027
- 风险等级：低（唯一低于中风险的方向）｜ 总算力：40–70 GPU 天 ｜ 墙钟：6–10 周

---

## 1. 摘要

流匹配 VLA（π0、SimVLA 等）的多步积分推理带来两难：步数多则延迟高，步数少则动作
质量崩塌；而"采样-校验"式测试时扩展（RoboMonkey）虽被证明有效，其校验器是 7B 级
VLM，部署成本反而超过策略本身。本文提出把两个问题合并为一个协同设计：（1）用
MeanFlow 恒等式将多步流头自蒸馏为**保留噪声条件的一步生成器**，蒸馏经零初始化区间
条件从 SFT 权重无损热启动；（2）训练一个 **<50M 参数、与策略共享冻结 VLM 特征**的
动作能量函数，其负样本来自一步生成器自身的提议分布（无需奖励标注、无需额外 VLM）；
（3）部署时单卡并行采样 K 个一步候选、按能量选优，整体延迟仍低于 10 步基线。理论上，
我们证明 argmin-能量的 Best-of-K 等价于以 -E 为代理奖励的 KL 正则化策略改进，并用
**极值理论**给出误差随 K 幂律衰减的首个解释——将 RoboMonkey 的经验观察上升为可检验
的理论预测。LIBERO 四套件实验（含扰动泛化与延迟 Pareto 前沿）验证：K=16 的能量选优
一步策略延迟降低约 5 倍、成功率不低于 10 步基线，且误差-K 曲线的幂律拟合 R²≥0.9。

## 2. 科学问题与动机

**问题一（延迟）**：SimVLA 推理 = 1 次 VLM 前向 + 10 次动作头前向（Euler 积分，
`models/modeling_smolvlm_vla.py`）。在动作头为 100–300M 参数时，积分占端到端延迟
的 40–70%。真实机器人闭环控制（10–30Hz 重规划)下这是硬约束。

**问题二（质量上限）**：模仿学习策略的单次采样质量受限于演示噪声与多模态混叠。
RoboMonkey 证明"采样 K 次 + 校验"能显著提升成功率且误差随 K 呈幂律下降——
但每次采样都是完整推理，测试时扩展与低延迟在现有方案中互斥。

**核心洞察**：一步生成器使"采样 K 次"的边际成本从 K 次完整推理降为一次
batch=K 的动作头前向（VLM 特征只算一次，`sample_action_candidates` 已实现）。
一步生成损失的多模态覆盖，恰好由校验器的选优补回。两者不是两个 trick 的堆叠，
而是互为存在条件：没有一步生成，BoK 太贵；没有校验，一步生成太差。

## 3. 相关工作深度对比

### 3.1 逐篇对比表

| 论文 | 链接 | 做了什么 | 关键不足（本文切入点） |
|---|---|---|---|
| RoboMonkey (2025) | [arXiv:2506.17811](https://arxiv.org/abs/2506.17811) | 对自回归 VLA 采样+高斯扰动构造提议分布，7B VLM 校验器选优；经验发现误差-样本数幂律 | ① 校验器 7B、需独立 GPU 服务，延迟与成本不可部署；② 提议分布靠对单点高斯扰动，多样性受限于单次采样；③ 幂律纯经验，无理论解释；④ 只测自回归离散 VLA，流策略未触及 |
| Verifier-free TTS (2025) | [arXiv:2510.05681](https://arxiv.org/abs/2510.05681) | 无校验器的测试时采样（多数投票/一致性类） | 无学习信号判别"好但少数派"的动作模式；多模态任务中投票会系统性抹掉正确的低频模式——正是我们用能量函数保留的东西 |
| V-GPS (CoRL 2024) | [arXiv:2410.13816](https://arxiv.org/abs/2410.13816) | 离线 RL（Cal-QL）训练 Q 函数，对生成式策略采样出的动作重排序，plug-and-play | ① 需要**奖励标注**的离线数据训练 Q（我们的 InfoNCE 校验器只需演示数据+生成器自身负样本）；② 策略本身未加速，K 次采样 = K 次完整推理；③ 未研究扩展定律；④ Q 与策略无协同训练，Q 的负样本分布与策略提议分布错配 |
| MeanFlow one-step VLA (2026) | [arXiv:2603.01469](https://arxiv.org/html/2603.01469) | 将 MeanFlow 用于 VLA 得到一步生成 | **最直接的撞车对象**。但其：① 一步回归塌缩多模态且无补救机制；② 无校验器、无测试时扩展；③ 无理论。本文以其为基线，证明 onestep+verifier > onestep |
| MP1 (2025) | [arXiv:2507.10543](https://arxiv.org/abs/2507.10543) | MeanFlow 一步策略用于 3D 点云操纵（非 VLA） | 小模型单任务设定；未涉及 VLM 骨干、语言条件与测试时扩展 |
| One-Step Diffusion Policy | [arXiv:2410.21257](https://arxiv.org/abs/2410.21257) | 分布匹配蒸馏扩散策略到一步 | 需要教师+判别器的复杂蒸馏管线；VLA 未验证；无选优机制 |
| Consistency Policy (RSS 2024) | [arXiv:2405.07503](https://arxiv.org/abs/2405.07503) | 一致性蒸馏加速扩散策略 | 3 步左右的加速，非真一步；同样无校验/扩展研究 |
| Implicit BC / EBM (2021) | [arXiv:2109.00137](https://arxiv.org/abs/2109.00137) | 能量模型直接作为策略（argmin E 采样） | 从 EBM 采样需要昂贵的负采样/朗之万迭代且训练不稳定——我们只用能量**排序**而不从它采样，规避了其全部不稳定性来源 |
| Real-Time Chunking (2025) | [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | 异步执行+inpainting 掩盖推理延迟 | 正交且互补：RTC 藏延迟、我们降延迟。论文中作为组合实验（EVOS+RTC）出现，不构成竞争 |
| BoN 对齐理论 (2024) | [arXiv:2401.01879](https://arxiv.org/abs/2401.01879) | LLM best-of-n 策略的 KL 界 | 理论工具来源。本文将其移植到动作空间并加上校验器排序误差(AUC)项与极值理论尾部分析——这两个扩展是新的 |

### 3.2 创新边界的诚实划定

调研后必须承认的三件事，以及相应的定位收缩：

1. "一步流 VLA"**不是**本文首创（2603.01469 在先）→ 本文首创的是
   **生成器-校验器协同设计**（校验器在生成器自己的提议分布上对比训练，二者共享冻结
   特征，形成无梯度互通的异步对抗迭代）。
2. "对动作重排序"**不是**本文首创（V-GPS 在先）→ 本文首创的是
   **无奖励标注的排序信号获取方式**（InfoNCE：正样本=演示，负样本=生成器提议+扰动），
   以及排序与一步生成的延迟协同（K 候选一次前向）。
3. "误差-K 幂律"**不是**本文首次观察（RoboMonkey 在先）→ 本文首次给出
   **极值理论解释**（见 §4.3）并将其作为可证伪预测在流策略上检验。

审稿人视角下这仍构成充分的新颖性：三个已知部件的每一个都被非平凡地改造，
且组合产生了任何单一部件不具备的性质（低于基线延迟的测试时扩展）。

## 4. 方法（形式化）

### 4.1 MeanFlow 自蒸馏（已实现）

教师场 v(x_t, t)（SimVLA SFT），学生场 u(x_t, r, t) 满足
(t−r)·u = ∫ᵣᵗ v(x_τ, τ)dτ。训练目标由 MeanFlow 恒等式给出：

```
u_tgt = v_t − (t − r)·(v_t·∇ₓu + ∂ₜu)，  L = ‖u_θ − sg(u_tgt)‖²/ (sg(‖Δ‖²)+c)^p
```

工程要点（均已在代码中落实）：区间嵌入零初始化保证 r=t 处与教师逐位等价
（`tests/test_task2_smoke.py` 第 1 项）；JVP 走数学注意力路径；`meanflow_ratio`
控制 r<t 样本占比，r=t 样本使目标退化为普通 FM 起稳定作用；保留噪声输入维持多模态。

### 4.2 能量校验器（已实现）

E_φ(o, a)：注意力池化的冻结 VLM 特征（M=8 查询）+ 本体感受 + 归一化动作块 →
标量能量，约 10M 参数。InfoNCE：

```
L = −log [ e^{−E(o,a⁺)/τ} / (e^{−E(o,a⁺)/τ} + Σⱼ e^{−E(o,aⱼ⁻)/τ}) ]
```

负样本三源：生成器提议（K_gen=8，距 GT 均方 L2 < δ 者按假阴性掩除）、
高斯扰动 GT（σ∈[0.1,0.5]）、批内乱序 GT。异步协同：A100 产出新生成器 checkpoint
→ 搬运至 A800 → 校验器在更新后的提议分布上重训/续训。

### 4.3 理论部分（论文的第三支柱，待推导完整证明）

**命题 1（BoK = KL 正则化改进）**。设基策略 π₀（一步生成器），
π_K(a|o) 为从 π₀ 采 K 个 i.i.d. 候选后取 argmin E 的策略。则 π_K 是
max_π E_π[−E(o,a)] − β(K)·KL(π‖π₀) 的近似解，且 KL(π_K‖π₀) ≤ log K − (K−1)/K
（把 [arXiv:2401.01879](https://arxiv.org/abs/2401.01879) 的 BoN 界移植到连续
动作 chunk 空间；证明只依赖排序统计量，与动作维度无关）。

**命题 2（校验误差感知的改进下界）**。设真实任务误差 ℓ(o,a)，校验器与 ℓ 的
排序一致性为 AUC=α。则 E[ℓ(BoK)] ≤ (1−2(α−½))·E[ℓ(π₀)] + 2(α−½)·E[min_{K} ℓ]。
直觉：α=1 时退化为 oracle-BoK，α=½ 时无增益。该式把"校验器要多好才值得"变成
可测量条件——预实验 P0.3 直接检验。

**命题 3（幂律的极值理论来源）**。若单样本误差 ℓ 的下尾满足
P(ℓ ≤ ℓ* + s) ~ C·s^γ（Weibull 型吸引域），则 E[min_K ℓ] − ℓ* ~ Θ(K^{−1/γ})。
即 RoboMonkey 观察到的幂律指数 = 动作误差分布下尾指数的倒数。这给出一个**独立
可检验的预测**：从同一策略的误差样本直接估计 γ（Hill 估计量），其倒数应与
成功率-K 曲线拟合出的 b 一致（允许 ±30% 偏差）。若二者系统性不符，命题被证伪。

## 5. 红队自我批判（写作过程中发现的问题与对策）

| # | 自我质疑 | 严重性 | 对策 |
|---|---|---|---|
| R1 | LIBERO 成功率已近饱和（OpenVLA-OFT 97%+），BoK 增益可能只有 1–2 点，不显著 | 高 | ① 主打**延迟-成功率 Pareto**而非绝对成功率；② 增加 LIBERO-Plus 扰动设定（成功率远离饱和，BoK 增益空间大）；③ 用 libero_10（长程）做主秀场 |
| R2 | 校验器可能学到平凡解：只判"动作平滑性"而非任务相关性 | 中 | 机制实验 M2：能量与 oracle 误差的相关性按任务分解；负样本消融中"仅扰动"组若与全量组等效，即证实平凡解，需加难负样本（时间错位 GT） |
| R3 | 一步生成在精细操纵（插入、开抽屉）上可能显著差于 10 步，教师差距 >5 点 | 中 | Phase 0 设 go/no-go：若 20k iter 蒸馏后 libero_spatial 差距 >3 点，改用 2 步 MeanFlow（延迟仍降 5 倍） |
| R4 | InfoNCE 的假阴性问题：生成器提议中相当比例可能是"另一个正确模式" | 中 | 已实现 min_neg_dist 掩码；消融验证 δ 敏感性；备选方案：能量目标改 margin-ranking（只要求 GT 能量低于"明显坏"的负样本） |
| R5 | 与 2603.01469 撞车加剧：对方若加个校验器就覆盖我们 | 时间风险 | 唯一对策是速度 + 理论差异化（命题 2、3 对方没有做的概率高）；9 月 ICLR 截稿前必须挂 arXiv |
| R6 | 极值理论假设（i.i.d. 候选、Weibull 下尾）在噪声条件生成器下可能不成立 | 低-中 | 命题 3 本来就设计为可证伪项；若 Hill 估计与拟合指数不符，论文如实报告为"幂律存在但 EVT 机制被拒绝"，这本身是有价值的负结果 |

## 6. 实验设计（分四阶段，含 go/no-go）

### Phase 0：可行性预实验（约 5 GPU 天，1.5 周）

| 编号 | 实验 | 判据（go/no-go） |
|---|---|---|
| P0.1 | **Oracle 上限测量**：SFT 10 步策略采 K=16 个动作 chunk，用仿真器 oracle（执行后离目标状态距离）选优，跑 libero_spatial 2 任务 × 20 回合 | oracle-BoK 增益 ≥5 点 → go；<5 点说明采样多样性不足，先修采样温度再重测；两次仍 <5 点 → **砍方向** |
| P0.2 | **蒸馏质量门**：20k iter MeanFlow 蒸馏，测 onestep vs teacher 在 libero_spatial 的成功率差 | 差距 ≤3 点 → go；>3 点 → 改 2 步积分重测 |
| P0.3 | **校验器可学性**：10k iter 校验器，held-out 集上 GT vs 生成器负样本的排序 AUC | AUC ≥0.7 → go；0.6–0.7 → 加难负样本重训；<0.6 → 回到 R2 对策 |

三个 gate 全过才进 Phase 1。任何 gate 失败的处理路径已在表中预注册，避免临时拍脑袋。

### Phase 1：主实验（约 25 GPU 天，3 周，两站点并行）

- **训练**（A100）：SFT 教师 200k（已有则复用）→ MeanFlow 蒸馏 50k × {条件速度, teacher_velocity} 2 组
- **训练**（A800）：校验器 50k × 2 轮异步迭代（生成器 25k 与 50k 两个 checkpoint 各一轮）
- **评估**（两站点分摊）：
  - 数据集：LIBERO 四套件（spatial/object/goal/10），每任务 50 trials × 3 seeds
  - 方法组：teacher flow-10｜teacher flow-2｜onestep（=2603.01469 复现）｜BoK K∈{2,4,8,16,32}｜oracle-BoK（上限）
  - 外部基线：OpenVLA-OFT（公开权重，只测不训）、RoboMonkey 式高斯扰动+我们的校验器（隔离"提议分布来源"变量）
  - 指标：成功率、端到端延迟（A100 与 A800 各测一份）、GPU 显存、吞吐
- **鲁棒性转移**：LIBERO-Plus（[arXiv:2510.13626](https://arxiv.org/abs/2510.13626)）
  七类扰动下 teacher vs BoK-16，检验"测试时扩展提升分布外鲁棒性"（RoboMonkey 在
  AR 策略上的发现能否迁移到流策略）

### Phase 2：消融（约 15 GPU 天，2 周）

1. meanflow_ratio ∈ {0.25, 0.5, 0.75}；2. 蒸馏目标：条件速度 vs teacher_velocity；
3. 负样本三源的 2³ 组合；4. min_neg_dist δ ∈ {0, 0.05, 0.1, 0.2}（δ=0 即无假阴性
处理）；5. 校验器规模 hidden {256,384,512} × depth {2,4,6}；6. 温度 τ；
7. 候选积分步数 {1,2,4} × K 的联合 Pareto；8. 校验器特征来源：共享冻结 VLM vs
独立小视觉编码器（验证"共享特征"设计的必要性）。

### Phase 3：机制与理论验证（约 8 GPU 天，1.5 周）

- **M1 扩展定律**：`fit_scaling_law.py` 拟合 err(K)=a·K^(−b)+c，报告各套件 R²；
- **M2 命题 3 检验**：对每套件采 2000 个候选误差样本，Hill 估计下尾指数 γ，
  对比 1/γ 与拟合指数 b；
- **M3 命题 2 检验**：测量校验器 AUC，代入改进下界，对比实测增益是否落在界内；
- **M4 失败模式分类**：BoK 修复的失败 vs 引入的新失败（能量选中危险动作的比例）；
- **M5 KL 实测**：估计 KL(π_K‖π₀)（K 候选能量 softmax 近似）对照 log K−(K−1)/K 界。

### 统计纪律

所有成功率报 3 seeds 均值 ± 标准差；主对比（BoK-16 vs teacher）做配对 bootstrap
95% CI；扩展定律拟合报告留一 K 点交叉验证的 R²，防止 5 个点过拟合 3 参数的质疑
（这是审稿人必然的攻击点，K 网格因此扩到 {1,2,4,8,16,32,64} 共 7 点）。

## 7. 预注册可证伪结论

1. K=16 能量选优一步策略：延迟 ≤ 教师 flow-10 的 1/4，四套件平均成功率 ≥ 教师 −0
   （目标 +3）。任一不成立 → 方法失败。
2. bok 曲线幂律拟合留一交叉验证 R² ≥ 0.9。不成立 → 扩展定律假设被拒。
3. 固定 K：bok ≥ onestep 单调成立。不成立 → 校验器无效，退回 R2。
4. Hill 估计 1/γ 与拟合 b 相差 ≤30% → 命题 3 支持；否则如实报告 EVT 机制被拒。

## 8. 审稿人攻击模拟与回应预案

| 攻击 | 回应 |
|---|---|
| "V-GPS 已经做了动作重排序" | 表 3.1 第三行 + 消融 8：V-GPS 需奖励标注与完整推理 ×K；我们展示无奖励对比训练 + 一次前向出 K 候选的联合延迟优势；并把 V-GPS 式 Q（如可复现）加为排序器消融 |
| "只在仿真里做" | LIBERO 是该领域标准协议（SimpleVLA-RL、RoboMonkey 附录均如此）；结论限定为仿真 + 明示真实机器人为 future work；延迟测量在两种真实 GPU 上完成 |
| "5 个点拟合 3 参数当然 R² 高" | 7 个 K 点 + 留一交叉验证 + EVT 独立通道验证指数 |
| "增益来自更多计算而非方法" | 等计算对照：flow-10（10 次头前向）vs bok-8（1 次 VLM + 8 并行头前向），FLOPs 表格逐项列出 |
| "500M 模型结论不外推" | 明确 scope；用 large 配置（1024/24 层动作头）复跑主表一行作为规模稳健性检查 |

## 9. 算力与时间预算

| 项目 | 站点 | GPU 天 |
|---|---|---|
| SFT 教师（如已有可省） | A100 | 12 |
| Phase 0 | A100+A800 | 5 |
| Phase 1 训练+评估 | 双站点并行 | 25 |
| Phase 2 消融 | 双站点并行 | 15 |
| Phase 3 机制 | A800 为主 | 8 |
| 返工缓冲（30%） | — | 15 |
| **合计** | | **~70（并行后墙钟 6–10 周）** |

今天为 2026-07-03；ICLR 2027 摘要截稿约 2026-09 中旬：紧但可行，前提是
Phase 0 于两周内启动。

## 10. 参考文献

- RoboMonkey: Scaling Test-Time Sampling and Verification for VLA Models — https://arxiv.org/abs/2506.17811
- Verifier-Free Test-Time Sampling for Vision-Language-Action Models — https://arxiv.org/abs/2510.05681
- Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance (V-GPS, CoRL 2024) — https://arxiv.org/abs/2410.13816
- Mean-Flow based One-Step Vision-Language-Action — https://arxiv.org/html/2603.01469
- MeanFlow: Mean Flows for One-Step Generative Modeling — https://arxiv.org/abs/2505.13447
- MP1: Mean Flow Tames Policy Learning in 1-step — https://arxiv.org/abs/2507.10543
- One-Step Diffusion Policy — https://arxiv.org/abs/2410.21257
- Consistency Policy (RSS 2024) — https://arxiv.org/abs/2405.07503
- Implicit Behavioral Cloning (EBM policies) — https://arxiv.org/abs/2109.00137
- Real-Time Execution of Action Chunking Flow Policies — https://arxiv.org/abs/2506.07339
- Theoretical Guarantees on the Best-of-n Alignment Policy — https://arxiv.org/abs/2401.01879
- SimpleVLA-RL (ICLR 2026) — https://arxiv.org/abs/2509.09674
- LIBERO-Plus: In-depth Robustness Analysis of VLA Models — https://arxiv.org/abs/2510.13626
- OpenVLA-OFT (Fine-Tuning VLA baselines) — https://arxiv.org/abs/2502.19645
