# P3 · VLA 共训练中的梯度干扰：测量、理论与连续谱知识绝缘

**When Does Action Learning Destroy Semantic Knowledge? Measurement, Theory, and Continuous-Spectrum Insulation for VLA Co-Training**

- 目标会议：NeurIPS 2027（主）；备选 TPAMI（测量+理论部分适合长文）
- 风险等级：中 ｜ 总算力：100–150 GPU 天 ｜ 墙钟：10–16 周
- **调研后的重大定位调整**：见 §3.2——零空间投影方法本身已不新颖，本文必须以
  测量与理论为第一、第二贡献，方法退居第三

---

## 1. 摘要

在 VLM 骨干上接续连续动作头（流匹配/扩散）并端到端微调，会损害骨干的语义能力与
指令泛化——这一现象已被 Knowledge Insulation（π0.5 系）等工作反复确认，但领域对它
的理解停留在"是否发生"与"全有或全无的处方"（stop-gradient 或冻结）层面。三个基础
问题无人回答：干扰发生在**哪些层、哪些参数子空间**？其大小由什么量控制？处方空间中
除了 0 与 1 之外的**连续谱**是否存在更优点？本文给出三部曲：（一）测量——提出
Fisher 加权梯度冲突谱，在 SimVLA 共训练全程逐层追踪流匹配梯度与语言建模知识子空间
的重叠，绘制 VLA 训练的首张"干扰地图"，并区分两种混杂的机制（表征漂移 vs 输出头
挤占）；（二）理论——在二阶近似下证明语义遗忘量的上界由动作梯度在骨干 Fisher 主
子空间上的累计投影范数控制，从而把 SimVLA 代码中 `learning_coef=0.1`、
`freeze_steps=1000` 这类工程魔法数字统一为该上界中可推导的量；（三）方法——将
上界的最小化转写为逐层连续绝缘系数 α_ℓ，其两个端点恰为完全冻结与朴素共训练，
KI 的 stop-gradient 是其特例。实验在操纵成功率 × 语义保持率（VQA/POPE/指令泛化）
二维平面上比较 Pareto 前沿，预注册的判决性预测是：由干扰谱推导的 α_ℓ 配置严格
支配 KI 与均匀学习率折减。

## 2. 科学问题

**现象**：KI（[arXiv:2505.23705](https://arxiv.org/abs/2505.23705)）实测表明
连续动作头的梯度回传显著拖慢训练并损害骨干知识迁移；ChatVLA
（[arXiv:2502.14420](https://arxiv.org/abs/2502.14420)）实测端到端动作微调后
VQA 能力严重退化。**但**：另一条证据链（[Forget-Me-Not 项目](https://continual-vlas.github.io/forget-me-not/)）
声称预训练 VLA 在持续学习操纵任务序列时对**旧操纵任务**出人意料地抗遗忘。两条
证据并不矛盾——前者测的是语义知识、后者测的是技能记忆——却提示遗忘在 VLA 中是
**高度子空间特异的**。这正是本文的出发点：遗忘不是标量而是有结构的张量，
应该被逐层、逐子空间地测量与控制。

**形式化**：骨干参数 θ，语义能力由参考分布 D_sem 上的损失 L_sem 度量，其局部
几何由 Fisher 信息 F_sem 刻画。共训练更新 Δθ_t 来自动作损失梯度 g_act。问：
累计语义损伤 L_sem(θ_T) − L_sem(θ₀) 与序列 {Proj_{F_sem}(g_act,t)} 的何种泛函
相关？该关系能否反推出最优的逐层干预？

## 3. 相关工作深度对比

### 3.1 对比表

| 论文 | 链接 | 做了什么 | 关键不足（本文切入点） |
|---|---|---|---|
| Knowledge Insulation (2025) | [arXiv:2505.23705](https://arxiv.org/abs/2505.23705) | 流头全程 stop-grad + 离散 token 分支维持骨干学习 | 二值处方；无定位（哪层哪子空间）；无量化理论；离散分支引入额外训练成本。本文将其纳入 α_ℓ≡0 特例 |
| AEGIS (2026) | [arXiv:2604.16067](https://arxiv.org/pdf/2604.16067) | 锚点约束的梯度隔离 | 仍是启发式约束强度；无干扰谱测量，无 Fisher 几何 |
| GNSP (2025) | [arXiv:2507.19839](https://arxiv.org/abs/2507.19839) | VLM **顺序持续学习**中把新任务梯度投影到旧知识零空间 | **方法撞车的主要对象，必须诚实处理**：投影算子形式相似。差异有三：① 设定不同——顺序 CL vs 我们的**同时性共训练**（动作与语义梯度每步同时存在，零空间是动态的）；② GNSP 无逐层干扰测量与遗忘上界理论；③ 未触及 VLA/连续控制。本文引用并明确将方法贡献定位为"迁移+理论化"，而非发明 |
| Adam-NSCL (2021) | [arXiv:2103.07113](https://arxiv.org/abs/2103.07113) | 零空间投影做持续学习的开创工作之一 | 同上：CL 设定、CNN 规模；作为方法谱系的根引用 |
| PCGrad (2020) | [arXiv:2001.06782](https://arxiv.org/abs/2001.06782) | 多任务梯度手术（冲突时投影） | 逐 batch 瞬时冲突，无知识子空间概念（两个任务对称）；我们的设定不对称：语义知识是要保护的存量而非并行任务。作为基线 |
| EWC (2016) | [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) | Fisher 对角正则防遗忘 | 对角近似丢失子空间结构；作为基线与 Fisher 工具的出处 |
| ChatVLA (2025) | [arXiv:2502.14420](https://arxiv.org/abs/2502.14420) | MoE 分离控制与理解 + 分阶段对齐 | 结构性绕开而非理解问题；推理时带 MoE 开销；其"spurious forgetting vs task interference"二分是纯行为学的，我们给出参数空间对应物 |
| Actions as Language (2025) | [arXiv:2509.22195](https://arxiv.org/abs/2509.22195) | 全离散动作 token 化避免连续头，声称无遗忘 | 证实"连续头是祸源"但代价是放弃连续控制的精度优势；恰好构成我们的一个对照组：离散化=把动作梯度天然限制在词表嵌入子空间 |
| UAM (2026) | [arXiv:2605.15735](https://arxiv.org/html/2605.15735v1) | 双流视角分析 VLA 遗忘 | 行为学分析为主；无 Fisher 几何、无干预算子 |
| Forget-Me-Not (2026) | [项目页](https://continual-vlas.github.io/forget-me-not/) | 预训练 VLA 对旧操纵技能抗遗忘 | 表面上与 KI 矛盾的反例——本文的子空间视角恰好统一二者（技能记忆与语义知识居于不同子空间），把"矛盾"转化为本文假设的证据 |
| 信息论约束持续 VLA (2026) | [arXiv:2603.13335](https://arxiv.org/pdf/2603.13335) | 互信息约束的持续 VLA 对齐 | 顺序 CL 设定；信息论界难以逐层操作化 |

### 3.2 定位调整（红队结论，写作前必须内化）

第一版方案把"Fisher 零空间投影"当作核心创新。调研发现 GNSP、Adam-NSCL、NSP 系列
已把该方法族做得相当充分。**若以方法为主贡献投稿，novelty 一栏必死**。调整后的
贡献排序：

1. **测量**（首创性最稳）：同时性共训练中的逐层 Fisher 加权冲突谱 + 训练全程动态
   追踪 + "语义遗忘 vs 技能遗忘"的子空间分离实验。此前无人做过 VLA 版本。
2. **理论**（区分度最高）：遗忘上界 + 把 learning_coef/freeze_steps/stop-grad/LoRA
   四种工程处方统一为同一上界的不同松弛。
3. **方法**（谱系上的增量）：动态零空间 + 逐层连续 α_ℓ，明确写"将 GNSP 式投影
   推广到同时性共训练并由测量谱自适应配置"。

## 4. 方法

### 4.1 测量协议（贡献一）

- 语义参考集：VQAv2-val 子集(5k) + POPE + 指令改写泛化集（LIBERO 指令的 GPT 改写，
  测指令跟随而非视觉）；语义 Fisher F_ℓ 分块（K-FAC 近似）在 A800 上离线计算一次。
- 冲突谱定义：C_ℓ(t) = ‖P_{F_ℓ,k} g_act,ℓ(t)‖² / ‖g_act,ℓ(t)‖²，其中 P 为 F_ℓ 前
  k 主方向投影。每 500 步记录全层谱，绘制 (层 × 训练步) 干扰热图。
- 平行行为学轨迹：每 5k 步评 VQA/POPE/成功率，建立"谱 → 行为退化"的时序对应
  （谱是否**先行**于行为退化是关键问题——若是，谱可做早期预警信号，应用价值+1）。

### 4.2 理论（贡献二，证明義务）

二阶近似下 ΔL_sem ≈ Σ_t [g_semᵀΔθ_t + ½Δθ_tᵀ F_sem Δθ_t]。在 g_sem≈0（骨干已
收敛于语义任务）时首项消失，得

```
ΔL_sem ≲ ½ Σ_t η² · ‖P_{F} g_act,t‖²_{F-norm} + O(‖Δθ‖³)
```

推论：(a) 均匀学习率折减 c（SimVLA 的 learning_coef）以 c² 缩放上界但同等拖慢
动作学习；(b) 投影法在不损失正交分量的前提下将上界压为高阶项；(c) 最优逐层系数
α_ℓ* 有闭式近似 ∝ 冲突谱 C_ℓ 的函数。需要严肃对待的理论缺口：三阶项在 200k 步
累计下是否可控（计划用实测 ‖Δθ‖ 轨迹数值验证界的紧致性，作为理论诚实性检查）。

### 4.3 干预算子（贡献三）

g̃_act,ℓ = (1−α_ℓ)·g_act,ℓ + α_ℓ·(I − P_{F_ℓ,k}) g_act,ℓ。α_ℓ 三种配置对比：
均匀标量（≈GNSP 迁移版）、由谱推导的闭式 α_ℓ*、在线自适应（谱的滑动估计）。
k（子空间维数）与 K-FAC 更新频率为超参。与 Adam 的相互作用是已知暗坑
（[arXiv:2604.22407](https://arxiv.org/pdf/2604.22407) 指出梯度修改在 Adam 下的
失效模式）：投影在动量累积**之前**施加，并加一组 SGD 对照实验。

## 5. 红队自我批判

| # | 质疑 | 严重性 | 对策 |
|---|---|---|---|
| R1 | SmolVLM-500M 语义能力本就弱，VQA 掉几个点效应量太小，故事撑不起来 | **高** | Phase 0.1 先测效应量；若 ΔVQA <3 点，主实验直接上 SmolVLM2-2.2B（80G 单卡可容），500M 降为快速消融平台 |
| R2 | GNSP 撞车质疑："你们就是 GNSP 用在 VLA" | 高 | §3.2 的贡献排序 + 相关工作里整段明写谱系；测量与理论部分 GNSP 完全没有，审稿人可核验 |
| R3 | K-FAC 主子空间估计的 k 选取任意，结论可能对 k 敏感 | 中 | k 作为显式消融轴；报告谱能量覆盖率(90%/95%/99%)三档下结论稳定性 |
| R4 | "语义保持有什么用？机器人只要会操纵" | 中（动机级） | 三个下游证据：指令改写泛化集成功率（语义直接服务操纵）、KI 论文自己的泛化论证、开放词汇物体操纵小实验（保持的语义 → 未见物体类别成功率） |
| R5 | 干扰谱可能显示冲突均匀分布（无结构），整个"子空间特异"假设崩 | 中 | 这正是预注册证伪条件之一；若谱熵接近均匀，论文转向负结果+行为学解释,但先发 workshop 短文止损（Phase 0 只花 10 GPU 天） |
| R6 | Pareto 前沿需要太多训练点 | 中 | 50k 短训做扫描（已验证 SimVLA 在 50k 有可分性）、只对 3 个胜出配置跑 200k 全程 |

## 6. 实验设计

### Phase 0：效应量与谱结构验证（约 10 GPU 天，2 周）

| 编号 | 实验 | go/no-go |
|---|---|---|
| P0.1 | 现有 SFT ckpt vs 原始 SmolVLM：VQAv2/POPE/指令改写集三项落差 | 任一落差 ≥5 点 → go（500M 平台成立）；<3 点 → 换 2.2B 重测；2.2B 仍 <3 点 → **砍方向** |
| P0.2 | 1k 步共训练的即时冲突谱（无需 K-FAC，先用逐层梯度余弦近似） | 谱有层间结构（熵显著低于均匀）→ go |
| P0.3 | K-FAC 管线在 A800 上的可行性：2.2B 分块 Fisher 的内存/耗时 | <3 天完成一次 → go；否则降为对角+top-k 混合近似 |

### Phase 1：测量主实验（约 30 GPU 天，3 周）

朴素共训练 × 3 学习率 × 2 模型规模，全程记录干扰谱 + 行为轨迹 →
产出干扰地图与"谱先行性"分析。此阶段产物独立成文（workshop 或主文 §4），
即使后续方法失败也有可发表增量。

### Phase 2：干预对比（约 60 GPU 天，5 周）

- 方法组：朴素共训练｜learning_coef ∈ {0.01,0.1,0.3}｜freeze（α=1）｜
  KI 复现（stop-grad + 离散辅助）｜LoRA(r=16/64)｜PCGrad｜EWC｜
  α 均匀｜α_ℓ* 闭式｜α_ℓ 在线
- 每组 50k 步 ×2 seeds 扫 Pareto；胜出 3 组跑 200k ×3 seeds
- 指标平面：x=LIBERO 四套件平均成功率，y=语义保持率（三项归一均值）；
  外加指令改写泛化成功率（连接两轴的关键指标）

### Phase 3：机制与统一性分析（约 15 GPU 天）

- 用实测 Δθ 轨迹回填理论界，画"预测遗忘 vs 实测遗忘"散点（理论紧致性）；
- 语义 vs 技能双参考集的子空间夹角（统一 KI 与 Forget-Me-Not 矛盾的直接证据）；
- Adam vs SGD 下投影效果差异（回应 2604.22407）；
- 离散动作分支（Actions-as-Language 式）在本框架内的解释：其梯度天然落在
  词表嵌入子空间，测其与 F_sem 主子空间的重叠作为"天然低冲突"解释。

## 7. 预注册可证伪结论

1. 干扰谱层间结构显著（谱熵 < 0.8×均匀熵）且集中于中深层 FFN；若均匀 → 核心
   假设证伪，执行 R5 止损路径。
2. 等成功率下 α_ℓ* 的语义保持 ≥ KI +5 点、≥ 均匀 α +2 点；若 α_ℓ* 不优于均匀
   α，则"逐层谱信息有用"被证伪（退化为 GNSP 迁移，论文不投主会）。
3. 理论界与实测遗忘的相关 ρ>0.6 且界不被violate；否则报告界失效区间。
4. 谱变化在行为退化前 ≥5k 步出现（先行性）；不成立则删去预警应用叙事。

## 8. 审稿人攻击模拟

| 攻击 | 回应 |
|---|---|
| "GNSP/NSCL 已有零空间投影" | 贡献排序声明 + 设定差异（同时性 vs 顺序）+ 理论与测量为主体；方法节标题直接写 "Extending null-space projection to simultaneous co-training" |
| "Fisher 是局部量，200k 步后早失效" | 动态重估（每 25k 步刷新 F）为默认配置，静态版是消融；理论诚实性检查（Phase 3 第一项）主动暴露界失效点 |
| "测量类论文没有 SOTA 数字" | NeurIPS 有测量/理解类论文传统（如 lottery ticket 系）；且 Phase 2 提供超过 KI 的 Pareto 数字兜底 |
| "只有 SmolVLM 系" | 500M+2.2B 两规模；争取加一组 π0-style 开源权重（openpi）的谱测量（只测不训，5 GPU 天） |

## 9. 算力与时间预算

| 项目 | GPU 天 |
|---|---|
| Phase 0 | 10 |
| Phase 1 测量 | 30 |
| Phase 2 干预 | 60 |
| Phase 3 分析 | 15 |
| 缓冲 30% | 35 |
| **合计** | **~150（墙钟 10–16 周）** |

分工：A800 承担全部 K-FAC/评测电池/谱记录后处理；A100 承担全部训练。
启动时点：P2 投稿后（约 2026-09 中）开 Phase 0/1，Phase 1 产物可先投
ICLR 2027 workshop 占位，主文瞄准 NeurIPS 2027（约 2027-05 截稿）。

## 10. 参考文献

- Knowledge Insulating VLA Models — https://arxiv.org/abs/2505.23705
- AEGIS: Anchor-Enforced Gradient Isolation — https://arxiv.org/pdf/2604.16067
- GNSP: Gradient Null Space Projection for VLM Continual Learning — https://arxiv.org/abs/2507.19839
- Adam-NSCL: Training Networks in Null Space for Continual Learning — https://arxiv.org/abs/2103.07113
- PCGrad: Gradient Surgery for Multi-Task Learning — https://arxiv.org/abs/2001.06782
- EWC: Overcoming Catastrophic Forgetting — https://arxiv.org/abs/1612.00796
- ChatVLA — https://arxiv.org/abs/2502.14420
- Actions as Language: VLM→VLA without Catastrophic Forgetting — https://arxiv.org/abs/2509.22195
- UAM: Dual-Stream Perspective on Forgetting in VLA Training — https://arxiv.org/html/2605.15735v1
- Forget-Me-Not: Pretrained VLAs are Surprisingly Resistant to Forgetting — https://continual-vlas.github.io/forget-me-not/
- Information-Theoretic Constraints for Continual VLA Alignment — https://arxiv.org/pdf/2603.13335
- Hidden Failure Modes of Gradient Modification under Adam — https://arxiv.org/pdf/2604.22407
- VLM4VLA: Revisiting VLMs in VLA — https://arxiv.org/html/2601.03309v1
- openpi (π0/π0.5 权重) — https://github.com/Physical-Intelligence/openpi
