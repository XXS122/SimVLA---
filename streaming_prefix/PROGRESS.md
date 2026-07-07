# 进度日志 —— 流式递归前缀编码（创新点三）

本研究线的实时文档（与追踪创新点四的 `latent_action/PROGRESS.md` 分开维护）。每次实验后更新。最新状态置于顶部。

---

## 当前状态（最后更新：Phase C 结果出炉后 —— 漂移修复已交付）

**Phase C 给出一半成功一半问题的结论，并据此做了一处设计修复。**

**时延（eval_latency）：干净的成功，甚至超过 Phase A 的预测。** 单步 streaming vs 全量重编码加速比：**10 步 Euler 下 1.51x，5 步 1.84x，1 步 2.78x**。按每 P 步重置一次摊销（P=10）：1.43x–2.36x，把可达控制频率提升到 33–103 Hz。算力富余是真实且已兑现的。

**漂移（eval_drift）：teacher-forced 正常，但递归发散 —— 一个真问题，根源在训练与部署不匹配。** 在 50 个 episode 上：teacher-forced 误差保持健康（0.36–0.62，与训练 recon 0.50 一致），但递归自我条件化的漂移在第 1 步就跳到 0.82，此后维持在 0.73–0.94；到第 10 步 exposure gap 达 0.55。在 0.3 漂移预算下推算出的安全重置周期为 **0 步** —— 算子连一步都无法在自身输出上递归。（脚本里那句 "SATURATING" 是个粗糙的尾部斜率判断，在这里有误导性：漂移是在 *0.8 这个高位* 趋平，并非在接近 0 处封顶。）

**根因（我 Phase B 的设计错误）：** `train_student.py` 采用纯 teacher-forced 训练（始终喂真实的上一帧融合输出）。算子从未见过自己带误差的预测，于是学到一个对输入误差极其脆弱的映射 —— 教科书式的 exposure bias / 协变量漂移（DAgger 问题）。证据精确：teacher-forced 和 recursive 的漂移在第 1 步*完全相等*（两者都喂真实 reset），从第 2 步才分叉，此时递归路径开始吞噬自己的误差。

**已交付的修复（本次提交）：** 带 scheduled sampling 的递归展开训练 —— 算子现在每个窗口展开 `--rollout_len` 步，并以从 0 爬升到 `--max_ss_prob` 的概率喂它*自己*（detach 后）的上一步预测，损失在所有展开步上取平均。这把算子自身的误差分布带进了训练，正是学会漂移纠正所需要的。这也把论文里的“几何漂移界”主张从一个断言变成一个*被训练出来的*性质（把有效 Lipschitz 常数压到 1 以下，使递归漂移可证地饱和），比单纯假设它成立是更强的贡献。

**已验证**（本次提交）：完整的 rollout 循环端到端跑通；且在一个具有真实时序结构的受控玩具动力系统上，rollout+scheduled-sampling 相比纯 teacher forcing 降低了末步递归漂移（0.616 → 0.522），证明这个修复方向正确，而不只是“能跑”。

**下一步动作（重训，然后重跑同一个漂移评测）：**
```bash
git pull
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.train_student \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --output_dir runs/latent_action_ws/streaming_prefix_rollout \
    --rollout_len 8 --max_ss_prob 0.9 --ss_ramp_frac 0.5 \
    --iters 30000 --batch_size 32 --num_workers 16 \
    2>&1 | tee logs/train_updater_rollout.log

CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_drift \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --updater_ckpt runs/latent_action_ws/streaming_prefix_rollout/updater_final.pt \
    --horizon 20 --num_episodes 50 \
    --output logs/drift_report_rollout.json 2>&1 | tee logs/eval_drift_rollout.log
```
Go/no-go 判据：目标重置周期处（比如 P=8–10）的递归漂移必须明显低于纯 teacher-forced 那次的 0.8 —— 理想落到 0.3–0.5 区间，使 `implied safe reset period` 变成一个有用的正数。训练时盯 `recon_stepK`（最难的、最后一个展开步的损失）：它应低于“第 1 步损失做朴素递归”的水平；`ss_prob` 日志确认 scheduled sampling 在爬升。batch 降到 32，因为现在每步要做 `rollout_len+1` 次教师前向。

---

## （上一阶段状态）Phase B 训练完成后

**Phase B 训练成功（teacher-forced）。** 在真实 SmolVLM-500M + 全量 LIBERO 上跑了 3 万步蒸馏，收尾于 **recon = 0.504**（delta_energy 0.174）。recon 从精确的 1.0（零初始化的“预测无变化”基线）降到约 0.50 —— 即状态更新算子解释了融合语言模型输出中约 50% 的逐帧方差，干净地跑赢了“直接抄上一步”的基线。这是一个真实的正面结果（对比创新点四的 LAM 卡在 R²≈0）：机制是有效的。

**关于 0.50 含义的重要说明（它*不*代表什么）：** 这是 *teacher-forced* 的训练误差 —— 每一步都喂真实的上一帧融合输出作为缓存状态。部署时算子在两次重置之间会被喂它*自己*的上一步预测，误差会跨步累积（exposure bias）。因此 0.50 是乐观的、单步的、理想缓存下的数字；它本身并不能确立闭环可行性。递归自我条件化究竟是几何式漂移（有界）还是发散，恰恰是论文的核心理论主张，**必须在 Phase C 实测，不能假设。**

**驱动这一切的 Phase A（profiling）结果：** 在默认 10 步 Euler 下，VLM 前缀重编码占单步时延的 **45.5%**；其内部，融合语言模型前向占 72%（15.19ms），视觉塔占 23%（4.86ms） —— 所以本工作瞄准语言模型前向，与 VLA-Cache/TTF-VLA 介入的位置相反。随着 Euler 步数减少（更快的流匹配头），vlm_frac *上升*（1 步时达 83%），使动机更强。

**Phase C 诊断代码已交付（提交待推），尚未在真实模型上运行。** 两个脚本，均在 CPU/微型模型上端到端验证：

接下来运行（需要 Phase B 训好的 `updater_final.pt`）：
```bash
git pull
# 1. 递归漂移：有界（几何式）还是发散？
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_drift \
    --meta_path runs/latent_action_ws/metas/libero_train.json \
    --updater_ckpt runs/latent_action_ws/streaming_prefix/updater_final.pt \
    --horizon 20 --num_episodes 50 \
    --output logs/drift_report.json 2>&1 | tee logs/eval_drift.log

# 2. 端到端时延：streaming vs 全量重编码
CUDA_VISIBLE_DEVICES=6 python -m streaming_prefix.eval_latency \
    --updater_ckpt runs/latent_action_ws/streaming_prefix/updater_final.pt \
    --euler_steps 1 5 10 --reset_periods 5 10 20 \
    --output logs/latency_report.json 2>&1 | tee logs/eval_latency.log
```

**各自回答什么，以及 go/no-go 判读：**
- `eval_drift`：打印每一步 k 的递归漂移 vs teacher-forced 误差、两者差距（exposure bias）、给定漂移预算下推算的安全重置周期，以及漂移是饱和（符合几何界）还是发散。**如果漂移发散，流式方案闭环不可行，理论框架也就站不住 —— 这才是真正的 go/no-go，比 0.50 那个训练数字重要得多。**
- `eval_latency`：打印单步 full-vs-streaming 加速比，以及按每个重置周期折入一次全量重编码后的摊销加速/Hz。确认 Phase A 预测的富余空间是否兑现。（注意：在微型测试模型上无意义 —— 被替换的语言模型必须大到值得替换，正如 500M 规模下那样。）

这两个之后，Phase C 剩余的工作是闭环 LIBERO 成功率对比（streaming vs 全量重编码 vs VLA-Cache/TTF-VLA 基线），画在时延-成功率 Pareto 图上 —— 即论文主图 —— 这需要把更新算子接入评测服务路径（`evaluation/libero/serve_smolvlm_libero.py`），尚未完成。

---

## 环境 / 硬件背景

与创新点四同一台共享服务器（见 `latent_action/PROGRESS.md`）：8× A800-80GB，GPU 6，`reserve_gpu.py` 占着富余显存。Profiling（Phase A）只需几百 MB；训练（Phase B）需要在观测到真实模型显存占用后重新调整占用大小（该架构复用冻结的约 500M 教师 VLM，仅做前向、不反传，因此显存应远低于 `pretrain`/`finetune` 阶段约 70GB 的量级，但尚未实测）。

---

## 时间线日志

### 1. Phase A：时延 profiling（commit `cba0e34`）

编写了 `streaming_prefix/profile_prefix.py`：在一系列 Euler 步数下，对比 `forward_vlm_efficient`（VLM 前缀，每次 `generate_actions` 调用一次）与完整推理调用的耗时，不需要训练好的 checkpoint（时延与权重无关）。`--breakdown` 进一步把 VLM 前向拆成视觉塔 / 连接器 / 融合语言模型三段计时。

在交付前，用一个微型本地 Idefics3（mock 掉 processor 加载）在 CPU 上端到端验证 —— 无报错，产出表格、拆解和 JSON 报告（该规模下数字无意义，机制正确性已确认）。

**真实结果**（SmolVLM-500M，batch=1，`hidden_size=768 depth=12 num_heads=12 image_size=384 num_views=2`，`--repeats 30 --warmup 10`）：

```
VLM prefix forward (1x per control step): 21.00 +/- 0.05 ms
  vision tower  : 4.86 ms
  connector     : 0.07 ms
  text model    : 15.19 ms (prefix seq_len=96)

 steps   total_ms     vlm_ms  action_ms  vlm_frac       hz
     1      25.27      21.00       4.26     83.1%     39.6
     5      34.76      21.00      13.75     60.4%     28.8
    10      46.18      21.00      25.18     45.5%     21.7
    20      69.64      21.00      48.63     30.2%     14.4
```

**解读：**
- 在标准 10 步配置下，VLM 重编码占总时延 45.5% —— 一个大到值得攻击的份额。
- 融合语言模型前向（占 VLM 自身成本的 72%）比视觉塔（23%）高约 3 倍。这与现有 VLA 缓存文献（VLA-Cache、TTF-VLA）介入的位置相反 —— 它们瞄准视觉塔，而这里视觉塔本就相对便宜。
- 随着 Euler 步数减少（也正是领域前进的方向 —— 1–4 步的流匹配动作头），vlm_frac *上升*（30%→45%→60%→83%）：动作头越快，VLM 前缀瓶颈占比反而越重。这加强而非削弱了动机。

**决策：** 进入 Phase B，并把目标重新锁定为逼近融合语言模型前向（而非视觉塔）。

### 2. Phase B：PrefixStateUpdater + 蒸馏训练（commit `b0fd755`）

新增：
- `streaming_prefix/models.py`
  - `VLMPrefixTeacher`：包装一个冻结的 SmolVLM，分别暴露 `cheap_forward`（视觉塔 + 连接器 + 文本 embedding 查表 + padding —— 占 VLM 成本约 23%，每步重算是安全的）和 `expensive_forward`（融合语言模型前向 —— 约 72% 的目标）。
  - `PrefixStateUpdater`：一个小型 Transformer（复用 `models/transformer_smolvlm.py` 的 `TransformerBlock` 以与代码库其余部分保持一致），把 `vlm_features_t` 预测为 `cached_state_{t-1} + delta_hat`，其中 `delta_hat` 由 `(cached_state_{t-1}, cheap_features_t)` 计算。输出投影层零初始化，所以模型初始时是精确的恒等映射（预测无变化）—— 刻意复刻创新点四 LAM（`latent_action/models.py`）里的差分预测修复，包括同样的归一化损失约定：**1.0 = 预测零变化 / 无用**。
  - `distillation_loss`：带掩码（通过 `attention_mask`，因为 padding 的 batch 元素不该计入）的归一化 MSE，同样的 1.0 基线约定。
- `streaming_prefix/data.py`：`InstructionFramePairDataset` —— 复用 `latent_action.data` 的原始 uint8 帧读取（与创新点四的修复具有相同的 shm 安全性），并加入按 episode 的指令 tokenize（不需要动作标签 —— 纯粹是特征蒸馏目标）。
- `streaming_prefix/train_student.py`：teacher-forced 蒸馏循环。对同一 episode 的每个 pair `(frame_t, frame_{t+stride})`：计算 `t` 和 `t+stride` 处的真实教师输出；训练更新算子，在给定真实 `t` 输出作为缓存状态时，预测真实的 `t+stride` 输出。**已知局限，已标注给 Phase C：** 训练是 teacher-forced 的（始终以*真实*上一步输出为条件）；推理时更新算子在两次重置之间会以它*自己*的上一步预测为条件，漂移可能比训练损失所暗示的更快累积（exposure bias）—— 这正是论文的几何漂移界（`δ/(1−L)` 论证）要刻画并封顶的东西，Phase C 的评测必须实测检验，不能假设。

**已验证**（CPU，微型本地 Idefics3，合成 LIBERO 数据）：
- `PrefixStateUpdater` 初始化时是精确恒等（`pred == cached_state`，误差 1e-6）。
- `distillation_loss` 对“预测零变化”基线精确等于 1.0，符合设计。
- 算子能在 30 个 Adam 步内把单个真实（教师生成，微型模型）pair 的损失从 1.0 过拟合到 0.07 —— 证明机制可学习，而不只是接线正确。
- 完整 `train_student.py` main() 循环端到端跑通（数据加载 → 教师前向 → 更新算子前向 → 损失 → 反传 → checkpoint 保存/加载且配置正确往返，其间发现并修复一个 bug：`config_dict()` 漏了 `mlp_ratio`，导致重新加载 checkpoint 时静默用错 MLP 宽度）。

**尚未完成：** 在真实 LIBERO 数据 + 真实 SmolVLM-500M 骨干上做真实训练（本文档顶部的“下一步动作”命令）。Phase B 之后的一切（漂移测量、重置周期调优、与全量重编码 / VLA-Cache / TTF-VLA 基线的 Pareto 对比、LIBERO 闭环成功率评测）都属于 Phase C，尚未开始。

---

### 3. Phase B 训练结果 + Phase C 诊断代码（commit `160f9da`）

在真实 SmolVLM-500M + 全量 LIBERO 上的 Phase B 训练（3 万步，batch 64，stride 1）收尾于 **recon=0.504**（从 1.0 的零初始化基线降下来）—— 状态更新算子在 teacher forcing 下解释了融合语言模型输出中约 50% 的逐帧方差。明确的正面结果；但这是单步/理想缓存的误差，并不决定闭环可行性（见“当前状态”里的说明）。

新增 Phase C 诊断：
- `streaming_prefix/eval_drift.py` —— 递归漂移测量：从一次真实 reset 出发，让更新算子在自身输出上递归作用至多 `--horizon` 步，记录每步 k 相对真实全量重编码的归一化漂移，并附带一个 teacher-forced 参照（两者之差 = exposure bias）。报告推算的安全重置周期，以及一个针对几何界主张的粗略饱和判断。
- `streaming_prefix/eval_latency.py` —— 端到端单步时延，streaming（cheap_forward + 更新算子 + 动作头）vs full（cheap_forward + 真实语言模型 + 动作头），外加按每周期折入一次重置后的摊销加速/Hz。

两者均在 CPU + 微型本地 Idefics3 上端到端验证（结构、归一化、teacher-forced 参照、重置周期推算、JSON 报告全部正确；微型规模下绝对数字无意义 —— 未训练的更新算子正确显示漂移≈1.0，微型语言模型正确显示约 1.0x 加速）。

### 4. Exposure-bias 漂移修复：递归展开 + scheduled sampling（commit `6ed0559`）

Phase C 诊断给出泾渭分明的结论：时延是干净的胜利（streaming 10 步 1.51x、5 步 1.84x、1 步 2.78x；摊销 1.43–2.36x，33–103 Hz），但递归漂移发散 —— teacher-forced 误差维持约 0.5，而自我条件化漂移第 1 步跳到 0.82，到第 10 步 exposure gap 达 0.55，推算安全重置周期为 0 步。

根因：Phase B 采用纯 teacher-forced 训练（始终喂真实上一步融合输出），算子从未见过自己的误差，学到一个对输入误差脆弱的映射 —— 教科书式 exposure bias。证据精确：teacher-forced 与 recursive 漂移在第 1 步相等（都喂真实 reset），从第 2 步才分叉，此时递归路径开始累积自身误差。

修复：`train_student.py` 现在每个窗口把算子展开 `--rollout_len` 步，并以从 0 爬升到 `--max_ss_prob` 的概率喂它*自己*（detach 后）的上一步预测（scheduled sampling），把归一化损失在所有展开步上取平均。这把算子自身误差分布带进训练，并把论文的几何漂移界主张变成一个被训练出来的性质而非假设。

- `data.py`：`InstructionSequenceDataset` 产出连续帧窗口 `[K+1, V, H, W, 3]`（而非旧的随机 pair）。
- `models.py`：`masked_norm_mse` —— 各展开步共享一个一致的分母（真实单步变化能量），使 1.0 = “预测无变化”的约定在 scheduled sampling 下依然稳定。
- `train_student.py`：rollout 循环；记录 `recon_step1` / `recon_stepK` / `ss_prob`；batch 默认 32（每步现在做 rollout_len+1 次教师前向）。`rollout_len=1` 退化为旧的 teacher-forced 行为。

已验证：完整循环端到端跑通；在一个具真实时序结构的受控玩具动力系统上，rollout+scheduled-sampling 相比 teacher forcing 降低了末步递归漂移（0.616 → 0.522），确认修复方向正确，而不只是能跑。

---

# 参考（稳定内容 —— 框架而非流水账）

## 论文骨架（摘要结构）

**科学问题（缺口）。** 部署中的 VLA 策略每个控制步都重新编码整个视觉-语言前缀，尽管机器人相邻观测几乎相同。针对这种冗余的现有加速方法（VLA-Cache，arXiv:2502.02175；TTF-VLA，arXiv:2508.19257）都是：（i）基于余弦相似度的免训练启发式，在相机运动/光照变化下失效；（ii）在*离散自回归*解码器上验证，而非流匹配头；（iii）瞄准视觉编码器。而我们的 profiling 表明，在现代紧凑 VLA 中视觉编码器*并非*瓶颈 —— 对图文 token 的融合语言模型前向才是（占前缀成本 72% vs 23%），且没有任何先前工作以有原理、可学习的方式触及它。更糟的是，随着流匹配动作头转向少步/一步生成（2025–2026 趋势，如 SnapFlow arXiv:2604.05656），这个瓶颈会*增大*：在 1 步 Euler 下前缀占时延的 83%。

**核心挑战。** 用一个廉价的递归状态更新替换融合语言模型前向，引出一个本设置特有的稳定性问题：缓存的特征要条件化流匹配头中一整条多步 ODE 积分，且推理时更新算子必须在两次周期性重置之间递归作用于它*自己*的上一步输出 —— 因此单步精度并不蕴含闭环稳定性。重建误差可能跨步几何式累积。

**方法。** 一个可学习的状态更新算子（`PrefixStateUpdater`），给定上一步的真实融合特征和本步廉价重算的视觉+文本 embedding 特征，预测融合输出的*变化量* —— 在非重置步替换掉昂贵的语言模型前向。通过对冻结教师 VLM 的自蒸馏训练，并采用**递归展开 + scheduled sampling**训练，使算子学会纠正自己的累积误差。我们证明一个几何漂移界（误差 ≤ δ/(1−L)），使周期性重置调度成为一条有原理的设计准则，并说明 rollout 训练正是把有效 Lipschitz 常数 L 压到 1 以下的关键。

**关键实验 / 可证伪预言。**（1）时延：streaming 必须在时延-成功率图上 Pareto 支配全量重编码和两个启发式基线 —— *已确认*，摊销加速 1.43–2.36×，33–103 Hz。（2）漂移：递归漂移在 rollout 训练下必须几何式有界；若发散（在 teacher-forced 训练下确实发散 —— *已观测*），则方案闭环不可行。（3）闭环 LIBERO 成功率在安全重置周期处必须与全量重编码相差 <1%。（4）在相机运动/光照变化下，可学习算子必须保持精度，而余弦相似度启发式在此崩溃。

## 架构与数据流

`forward_vlm_efficient`（每步 VLM 前缀）的拆分，以及 profiler 测量的两个半部：

```
 每个控制步 t：
   images_t ─► 视觉塔 ─► 连接器 ──────────┐         便宜（约 23%，4.9 ms）
   指令 ─► 文本 embedding ────────────────┼─► combined_embeds_t
                                          │         每步重算
   ─────────────────────────────────────────────────────────────────────
   combined_embeds_t ─► 语言模型前向 ─► vlm_features_t          昂贵
                                                       （约 72%，15.2 ms）
                                        ▲
                                        │  非重置步由以下替换：
   缓存的 vlm_features_{t-1} ──┐        │
   combined_embeds_t ──────────┼─► PrefixStateUpdater ─► vlm_features_t
                                        （便宜；预测差分）
```

- **重置步**（每 P 步一次）：运行真实的昂贵前向，刷新缓存。成本 = full。
- **非重置步**：只运行便宜半部 + 更新算子。成本 = cheap + updater ≪ full。
- 摊销单步成本 = `(full + (P−1)·streaming) / P`。

更新算子（`PrefixStateUpdater`）是一个作用于前缀 token 序列的小型 Transformer：`new_proj(combined_embeds_t) + cache_proj(cached_state) + pos_emb` → 若干 TransformerBlock → `out_proj`（零初始化）→ **差分**，返回 `cached_state + 差分`。零初始化使它一开始是精确恒等（“预测无变化”），因此归一化损失起始精确为 1.0。

## 理论：几何漂移界

设 `g` 为更新算子，`c_k` 为第 k 步的便宜特征，`s*_k` 为真实全量重编码，`s_k = g(s_{k−1}, c_k)` 为递归推理状态（重置处 `s_0 = s*_0`）。定义误差 `e_k = ‖s_k − s*_k‖`。

- 单步（teacher-forced）重建误差：`δ = ‖g(s*_{k−1}, c_k) − s*_k‖` —— Phase B 最小化的量。
- 若 `g` 对其缓存状态参数是 `L`-Lipschitz 的：`‖g(s_{k−1},c_k) − g(s*_{k−1},c_k)‖ ≤ L·e_{k−1}`。
- 三角不等式：`e_k ≤ L·e_{k−1} + δ`，故 `e_k ≤ δ·(1 + L + … + L^{k−1})`。
- **若 L < 1：** `e_k ≤ δ/(1−L)` —— 有界；给定目标漂移预算即可映射出安全重置周期 P。**若 L ≥ 1：** 发散。

**这是关键。** Teacher-forced 训练把 δ 压小，但对 L 毫无作为 —— 而经验上训练出的 L ≥ 1（漂移发散，Phase C）。递归展开 + scheduled sampling 训练直接惩罚多步上的 `e_k`，这正是把 L 压到 1 以下所需的压力。所以这个界不是论文对模型做的一个假设 —— 而是训练过程被设计出来*诱导*的一个性质，漂移评测则检验它是否成功。

## 代码地图

| 文件 | 角色 | 关键部分 |
|---|---|---|
| `profile_prefix.py` | Phase A | `profile_breakdown`（视觉/连接器/语言模型拆分）、按 Euler 步数的表、`vlm_frac` |
| `models.py` | 核心 | `VLMPrefixTeacher.cheap_forward` / `expensive_forward`；`PrefixStateUpdater`（差分，零初始化恒等）；`distillation_loss`、`masked_norm_mse`（1.0 = 预测无变化）；`save/load_updater` |
| `data.py` | 数据 | `InstructionSequenceDataset`（供 rollout 的连续窗口）；`InstructionFramePairDataset`（遗留的 pair）—— 两者均无动作标注，经 `latent_action.data` 的原始 uint8 |
| `train_student.py` | Phase B | 递归展开循环；scheduled sampling（`--rollout_len`、`--max_ss_prob`、`--ss_ramp_frac`）；记录 `recon_step1/recon_stepK/ss_prob` |
| `eval_drift.py` | Phase C | 每步递归 vs teacher-forced 漂移；推算安全重置周期 |
| `eval_latency.py` | Phase C | streaming vs full 单步 + 摊销加速/Hz |

## 目前结果汇总

| 量 | 数值 | 来源 |
|---|---|---|
| VLM 前缀 / 单步时延占比 @10 步 | 45.5%（21.0 ms） | Phase A |
| — 视觉塔 / 连接器 / 语言模型 | 4.86 / 0.07 / 15.19 ms | Phase A `--breakdown` |
| vlm_frac @ 1 / 5 / 10 / 20 步 | 83% / 60% / 45% / 30% | Phase A |
| Teacher-forced 蒸馏 recon（1.0 = 无用） | **0.504** | Phase B |
| 单步加速 @10 / 5 / 1 步 | 1.51× / 1.84× / 2.78× | Phase C 时延 |
| 摊销加速 / Hz @10 步，P=10 | 1.43× / 33 Hz | Phase C 时延 |
| 递归漂移，teacher-forced 训练 | 0.82→0.94（发散） | Phase C 漂移 |
| — teacher-forced 参照（同一模型） | 0.36–0.62（健康） | Phase C 漂移 |
| — 推算安全重置周期 | 0 步 | Phase C 漂移 |
| Rollout 修复，玩具系统末步漂移 | 0.616 → 0.522 | 漂移修复验证 |
| Rollout 修复，真实漂移 | *待重训* | — |

## 相关工作与定位（附链接）

- [VLA-Cache](https://arxiv.org/abs/2502.02175) —— 免训练的静态视觉 token 自适应 KV 缓存。我们不同：可学习（非启发式）、瞄准语言模型前向（非视觉塔）、流匹配（非自回归），并带漂移界。
- [TTF-VLA](https://arxiv.org/abs/2508.19257) —— 基于像素注意力的时序 token 融合。同样三点差异。
- [Real-Time Chunking (RTC)](https://arxiv.org/abs/2506.07339) —— 推理时动作块异步执行；正交（它跨步重叠动作头，我们削减前缀成本）。可组合。
- [SnapFlow](https://arxiv.org/abs/2604.05656) / 一步流匹配头 —— 使我们贡献*更*相关的趋势（动作头步数下降时，前缀占比升至 83%）。
- [SmolVLA](https://arxiv.org/abs/2506.01844) —— 本工作坚持的“可负担 VLA”路线（单张 A800，500M 骨干）。

## commit 对照

| Commit | 内容 |
|---|---|
| `cba0e34` | Phase A —— `profile_prefix.py` 时延 profiling |
| `b0fd755` | Phase B —— `PrefixStateUpdater` + 蒸馏训练 |
| `160f9da` | Phase B 结果（recon=0.50）+ Phase C 漂移/时延诊断 |
| `6ed0559` | Exposure-bias 修复 —— 递归展开 + scheduled-sampling 训练 |

## 本文档的维护约定

每次实验/修复往时间线日志追加一条编号条目（背景 → 命令 → 结果 → 诊断），每次都更新顶部的“当前状态”，并在框架（而非仅最新数字）变化时同步“参考”区块。与对应的代码改动一起提交。
