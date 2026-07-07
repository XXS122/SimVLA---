# 进度日志 —— 可辨识连续潜动作流匹配预训练

创新点四的实时文档（流水线配方和论文产物对照表见 `README.md`）。每次实验后更新。最新状态始终置顶；下方是完整时间线日志，供细节/审计。

---

## 当前状态（最后更新：旋转修复结果出炉 —— 到达决策点）

**旋转合成 bug 被证伪，它不是真实数据 R² 偏低的原因。** 用 SO(3) 合成修复后在*同一份* v2 z-labels 上重跑探针；数字与修复前基本没变（完整对比见下方 §11）。这个 bug 真实存在、也值得修（合成数据上确认它能单独把一个完美仿射关系做成 R²≈0），但它不是拖住这份数据集的原因。

**免费计算（无需重跑）：** 把非线性探针限制到仅 5 个视觉上明显的维度（dx、dy、dz、dyaw、gripper —— 排除 roll/pitch），均值 R² ≈ **0.342**。仍远低于 0.6 闸门。这排除了“除了 2 个不可观测维度外都没问题”作为完整解释 —— 每个维度，包括视觉上明显的那些，都只能被部分解码。

**现在这是一个真正的决策点，不再是找 bug。** 两个结构性假设（z 坍缩/复制捷径；旋转合成数学）都已被发现、修复，并确认与剩余差距无关。已向用户提出三条前进路径（见 §11）：加大 LAM（便宜、收益不确定）、加入 CLAM 式少样本真实动作 grounding（收益可能更大，但改变论文“完全无动作标注”的核心主张）、或转向创新点五。**等待用户抉择。**

---

## 环境 / 硬件背景

- 共享服务器，8× NVIDIA A800-SXM4-80GB，训练固定在物理 GPU 6，经 `CUDA_DEVICES=6` 指定。
- 骨干：本地 SmolVLM-500M，位于 `$SIMVLA_SMOLVLM_MODEL`。
- 数据集：LIBERO（4 个子集：libero_10/goal/object/spatial），约 2000 条 demo，窗口化后约 33.6 万个已标注帧。
- 容器 `/dev/shm` 较小（Docker 默认 64MB）—— 与下方 DataLoader bus-error 的经历相关。
- `reserve_gpu.py`（纯显存占用、不做计算）用于在跑训练时把其他用户挡在 GPU 6 之外 —— 当前占约 43.7GB，留约 10GB 空闲（按 LAM 阶段峰值设定，在 `pretrain`/`finetune` 阶段前**必须调大**，那些阶段会跑完整 VLM，会像最初的全模型训练一样需要约 70GB）。

---

## 时间线日志

### 1. 流水线搭建（commit `a45fa99`）

构建了完整的 `latent_action/` 包，由 `python -m latent_action.run <stage>` / `./run_pipeline.sh <stage>` 驱动，所有路径来自环境变量（`paths.env`）：

- `make_splits.py` —— demo 级低数据划分（p1/p10）
- `models.py` —— `LatentActionModel`（冻结 SigLIP 特征上的逆动力学编码器 + 前向解码器）、VICReg 方差/协方差正则、共享自运动不变性增广、可选 VQ 瓶颈（离散消融）
- `train_lam.py` —— LAM 训练循环
- `label_z.py` —— 逐帧 z 标注（+ `--ego_aug` 压力标签）
- `probe.py` —— 岭回归探针 z↔a_norm、逐维 R² 报告、go/no-go 闸门、适配器初始化（`probe.npz`）
- `run.py` —— 统一阶段调度器
- 集成进现有训练栈：`LatentZActionSpace`、`LiberoZAdapterActionSpace`（`models/action_hub.py`）、`libero_z` 域处理器、`libero_hdf5.py` 里的 demo 白名单过滤、`train_smolvlm.py` 新增参数（`--z_dim`、`--probe_path`、`--train_adapter`、`--pretrained_flow_ckpt`、`--freeze_vlm`、`--run_name`、环境变量骨干默认值）。

在 CPU 上用合成数据和微型本地 Idefics3 端到端测试通过（探针恢复植入的仿射映射 R²=1.0；适配器往返正常；z-dataloader 产出 `[B,10,z]`；pretrain→热启动→finetune→generate_actions 全部通过）。

推送需要一次性 PAT（本仓库的 GitHub App 集成是只读的）；token 仅用于一次 URL 凭据、从未写入磁盘，随后已提醒用户撤销。

### 2. 首次训练尝试：DataLoader bus error

```
./run_pipeline.sh train-lam
```
→ `ERROR: Unexpected bus error encountered in worker`（SIGBUS），8 个 worker 全部反复报错。

**第一次修复尝试（commit `9cb6158`）：** 把 `torch.multiprocessing` 切换到 `file_system` 共享策略（默认张量 IPC 用 `/dev/shm`，Docker 上限 64MB）。**没修好** —— 重跑仍是同样的错。

### 3. 找到根因：是 IPC 载荷大小，不只是传输方式

真正原因：worker 通过 IPC 传输*预处理后*的 384×384 float32 张量（约 7MB/样本，batch 96 时约 672MB）—— 对任何小 `/dev/shm` 传输（包括 `file_system` 策略）都太大。

**修复（commit `bc6cdc3`）：** worker 现在传原始 LIBERO 分辨率的 uint8 帧（约 100KB/样本，缩小约 75 倍）；resize + ImageNet 归一化移到 GPU（`latent_action.data.gpu_preprocess`，与现有 CPU 流水线相同的操作和顺序，测试验证数值完全一致）。同时应用于 `train_lam.py` 和 `label_z.py`。

```
git pull
./run_pipeline.sh train-lam
```
→ 无崩溃跑起来。

### 4. GPU 显存 / 吞吐讨论

用户注意到仅用约 10GB，而 batch 64 的全模型训练用约 70GB。解释：LAM（26.5M 参数）在 `no_grad` 下跑 SigLIP 骨干（中间激活立即释放，无优化器状态，约 500M 冻结参数无梯度）—— 这里显存低是预期的，且*不能*与 `pretrain`/`finetune` 阶段类比，那些阶段会跑完整 VLM、回到约 70GB。

讨论中发现并修复一个真 bug：`run.py` 自己的 `--iters` 参数被调度器吞掉、从未转发给 `train_lam.py`（**commit `df5185d`**）。

建议（用户也照做了）用更大 batch 的配置来有效利用富余显存：
```
./run_pipeline.sh train-lam \
    --batch_size 320 --learning_rate 2e-4 --iters 20000 --num_workers 16
```

### 5. GPU 占用脚本（commit `b04c615`）

用户想占用富余显存，让共享服务器上其他用户在训练进行时无法在 GPU 6 上调度任务。编写了 `reserve_gpu.py`：只占显存（`torch.empty` 块，无计算核 —— 不偷占真实训练任务的 SM 算力）；当空闲显存接近零时，其他进程启动时分配失败。`--leave-free` 把余量设为训练任务的峰值（必须覆盖 LAM 自运动增广步约 1.7 倍的尖峰）。最终用 `--leave-free 10` 占约 43.7GB，经 `nvidia-smi` 确认（GPU 6：70915MiB/81920MiB 已用，100% 利用率来自真实训练进程）。

### 6. LAM v1 结果：探针失败，R²=0.03（z 坍缩）

训练（`--batch_size 320 --iters 20000`，默认 `--stride 1`）完成：`recon` 平台在约 1.47（当时绝对尺度无法解读），`var`→0.0022（健康，无维度坍缩），`cov`→0.073。

```
./run_pipeline.sh label
./run_pipeline.sh probe
```
```
=== affine probe z -> a_norm (val R^2) ===
  dx 0.0467  dy 0.0162  dz 0.0580  droll -0.0007  dpitch 0.0144
  dyaw 0.0177  gripper 0.0635   mean 0.0308
(reverse a->z mean R^2: 0.0100)
>>> gate (>=0.6): FAIL
```

**诊断：** 在 10Hz/128px 下，一步机械臂运动在粗糙（4 倍下采样）的连接器特征里几乎不可见；前向解码器可以直接抄 `f_t` 就把 `f_{t+1}` 重建得很好，完全用不上 `z`，于是 `z` 得不到真正的梯度压力 —— 方差正则随后把 `z` 填满了 batch 里*确实*有方差的东西（场景身份 / 任务上下文），而非运动。

### 7. LAM v2 修复：预测特征差分，stride 4（commit `ed75b43`）

三处改动，对应诊断的三个环节：
1. 解码器预测 `f_{t+k} - f_t` 而非直接预测 `f_{t+k}` —— 铲除复制捷径。`recon` 现在按差分能量归一化，给它一个绝对、可解读的尺度：**1.0 = z 无用**（零变化预测器）；健康训练必须明显降到 1 以下。
2. 默认 pair stride 1 → 4（0.4 秒的运动 —— 在粗糙特征里可见）。
3. `label_z.py` 从 LAM checkpoint 自动检测 stride；`probe.py` 把 z 对窗口均值动作做回归。

用同样的命令重训（`--batch_size 320 --iters 20000`，现在默认 `--stride 4`）。约 4000/20000 步的健康检查：`recon` 0.72–0.86（远低于 1.0 —— 真实信号），`var` 0.00–0.13（无坍缩），`inv` 在约一半的记录步上非零（符合预期，确认增广分支确实触发；早前 v1 摘要里 `inv=0` 只是恰好记录在非增广步，不是 bug）。

### 8. LAM v2 结果：仿射探针仍失败（R²=0.04），但 recon 证明真实信号存在

```
./run_pipeline.sh label     # 自动检测 stride=4
./run_pipeline.sh probe
```
```
=== affine probe z -> a_norm (val R^2) ===
  dx 0.0241  dy 0.0365  dz 0.0655  droll 0.0116  dpitch 0.0078
  dyaw 0.0328  gripper 0.0996   mean 0.0397
(reverse a->z mean R^2: 0.0116)
>>> gate (>=0.6): FAIL
```

这是驱动后续两步的关键矛盾：`recon` 大幅改善（证明 `z` 解释了真实的特征空间差分方差），但*仿射*探针完全没动 —— 这与两种可能一致：(a) 真实信号只是非仿射，或 (b) `z` 解释的差分方差不是机器人自己的动作（物体运动、接触动态）。

### 9. 加入非线性诊断探针（commit `70a1e4e`）

加入 `probe.py --nonlinear`：在*同一份*现有 z-labels 上拟合一个小 MLP z→a_norm（仅诊断，不据此导出适配器）—— 无需 GPU 重训，几分钟跑完。合成数据上验证：一个可逆的分段线性（非仿射）关系给出仿射 R²=0.845 vs 非线性 R²=0.998（清晰区分）；纯噪声两者都接近随机水平。

```
./run_pipeline.sh probe --nonlinear
```
```
=== nonlinear (MLP) probe z -> a_norm (val R^2) ===
  dx 0.3290  dy 0.3324  dz 0.2732  droll 0.0814  dpitch 0.1392
  dyaw 0.3393  gripper 0.4465   mean 0.2773
>>> diagnosis: weak/partial action signal in z, still far from usable.
```

0.28 vs 随机水平 0.04（仿射）—— 真实的、约 7 倍的跃升，但即使非线性也仍远低于 0.6 闸门。

**逐维模式，及其重要性：** 平移（dx/dy/dz）+ yaw + gripper 得分 0.27–0.45；roll/pitch 仅 0.08/0.14。这种不对称很难用“z 被场景噪声污染”解释（噪声没理由专门放过特定动作维度），但用“对 agentview+wrist 相机下的平行夹爪而言，roll/pitch 在视觉上远比平移/yaw/gripper 难观测”能自然解释 —— 即看起来像是被可观测性门控的部分*真实*信号，而非纯污染。这促使我们在对 LAM 下任何结论之前，先检查探针自身的数学。

另外单独标注：反向（a→z）R²=0.0116 —— 接近零。这超出诊断意义：`LiberoZAdapterActionSpace`（微调适配器）用严格仿射映射把真实动作编码成 z 空间训练目标；如果真实的 a→z 关系本身也不是仿射的，那么无论 z→a 探针说什么，适配器的编码方向在架构上就是可疑的。尚未处理 —— 暂搁置，等旋转修复的结果。

### 10. 发现并修复旋转合成 bug（commit `b40c323`）

stride>1 的探针目标把逐步欧拉角差分做了算术平均。有限旋转不能靠相加/平均合成（不可交换）—— 专门对 3 个旋转维在数学上是错的（平移和近二值的夹爪线性合成、不受影响）。

**修复：** `_windowed_action_target` 现在把逐步差分旋转作为正规 SO(3) 元素合成（`scipy.spatial.transform.Rotation`，链式相乘），再把净旋转转回逐步尺度的欧拉差分。两个探针和适配器初始化保存都用它。

**在受控合成数据上验证：** 把 `z` 构造成真实 SO(3) 合成旋转目标的*完美*仿射函数。用**旧的**朴素平均目标评测：`droll=0.003, dpitch=-0.0005, dyaw=0.011` —— 一个完美关系被打成 R²≈0。用**新的**修正目标评测：`droll=1.000, dpitch=0.9997, dyaw=1.000`。两种方式下平移/夹爪都是 ≈1.0，确认它们从未受影响。这与真实数据的 roll/pitch 模式（0.08/0.14 偏低，平移/yaw/gripper 0.27–0.45）在结构上高度吻合 —— 但这本身**并不**证明真实数据会恢复；它只证明之前对旋转维的判读被这个 bug 污染，必须重测。

**下一步命令（无需重训/重标，复用 v2 `z_labels.h5`）：**
```bash
git pull
./run_pipeline.sh probe --nonlinear 2>&1 | tee logs/probe_v2_fixed.log
```

### 11. 旋转修复在真实数据上的结果：基本无变化 → 假设被证伪

```
=== affine probe (val R^2) ===
  dx .0241  dy .0365  dz .0655  droll .0130  dpitch .0078  dyaw .0317  gripper .0996   mean .0397
(reverse a->z mean R^2: 0.0118)
=== nonlinear (MLP) probe ===
  dx .3298  dy .3380  dz .2758  droll .0444  dpitch .1362  dyaw .3200  gripper .4460   mean .2700
```

与修复前那次（§8–9）逐项对比：

| 维度 | 仿射 修复前 | 仿射 修复后 | 非线性 修复前 | 非线性 修复后 |
|---|---|---|---|---|
| dx | .0241 | .0241 | .3290 | .3298 |
| dy | .0365 | .0365 | .3324 | .3380 |
| dz | .0655 | .0655 | .2732 | .2758 |
| droll | .0116 | .0130 | .0814 | .0444 |
| dpitch | .0078 | .0078 | .1392 | .1362 |
| dyaw | .0328 | .0317 | .3393 | .3200 |
| gripper | .0996 | .0996 | .4465 | .4460 |
| **均值** | **.0397** | **.0397** | **.2773** | **.2700** |

在*每一个*维度上都相同（在 MLP 训练噪声范围内），不只是旋转维。这干净地证伪了“合成 bug 解释了真实数据的旋转缺陷” —— 修复确实有效（§10 的合成测试），只是它在这里不是瓶颈。

**免费的后续计算**（无需代码/重跑 —— 只是对已报告的逐维数字取平均）：限制到 5 个视觉上明显的维度 dx+dy+dz+dyaw+gripper：`(.3298+.3380+.2758+.3200+.4460)/5 = 0.342`。仍远低于 0.6。所以“roll/pitch 不可观测，其余都没问题”这个重新表述本身救不了闸门 —— 信号在全线都是弱到中等（0.27–0.45），而不是“6 个好维度 + 2 个坏维度”。

**这让 v2 recon/R² 矛盾的三种原始解释（§8）落到何处：**
- 复制捷径 / z 坍缩（§6–7 的原因）—— 已修，排除（recon 动了，var 健康）。
- 旋转合成数学（§10）—— 已修，在真实数据上排除（本节）。
- 剩余候选：z 所解释的约 15–25% 差分能量，实质上被机器人自身指令动作*之外*的东西主导（物体运动、接触动态、场景上下文），而这些恰好与视觉上大的维度（平移、yaw、gripper）中等相关、与视觉上细微的维度（roll、pitch）弱相关 —— 即主要是污染假设，而非可修的 bug。

**向用户提出的决策点**（三个选项，权衡如上推理）：
1. 加大 LAM（更大的 `--lam_dim`/`--enc_depth`/`--dec_depth`，更多 iters）—— 便宜（约 1 个 GPU 日），在更大改动前先排查容量/训练时长，不改变论文主张。
2. CLAM 式少样本 grounding —— 在 LAM 训练里混入少量真实动作标注数据作为辅助监督损失。收益可能更大，但把论文核心主张从“完全无动作标注”变成“无动作标注 + 轻度 grounding”（仍可辩护，也有已发表先例，但确实是表述上的真实转向）。
3. 转向创新点五（针对夹爪的混合离散-连续流匹配）—— 止损、重新部署精力；它有一个便宜的当天动机验证（在现有 SimVLA baseline checkpoint 上统计接触帧的夹爪预测直方图）。

等待用户决定 —— 见顶部“当前状态”。

---

## commit 对照

| Commit | 内容 |
|---|---|
| `a45fa99` | 初始完整流水线（latent_action 包 + 集成） |
| `9cb6158` | multiprocessing file_system 共享策略（部分修复） |
| `bc6cdc3` | 原始 uint8 IPC + GPU 端预处理（真正的 shm 修复） |
| `df5185d` | 在 `run.py` 中把 `--iters` 转发给 train-lam 阶段 |
| `b04c615` | `reserve_gpu.py` —— 纯显存占用守护脚本 |
| `ed75b43` | LAM v2 —— 差分预测、stride 4、窗口对齐的探针目标 |
| `70a1e4e` | 非线性（MLP）诊断探针 |
| `b40c323` | 探针目标的 SO(3) 旋转合成修复 |

## Go/no-go 闸门判据（自项目开始未变）

- 仿射探针均值 R² ≥ 0.6 → 通过，进入 `pretrain`/`finetune`。
- 合理调参后仍 < 0.6 → 该 LAM 设计的仿射可辨识性前提不成立；升级（架构改动）或转向。
- 非线性探针仅供诊断 —— 它本身不能通过闸门，因为下游适配器（`LiberoZAdapterActionSpace`）目前在两个方向都是严格仿射映射。

---

# 参考（稳定内容 —— 框架而非流水账）

## 论文骨架（摘要结构）

**科学问题（缺口）。** 在 VLA 中，视觉-语言骨干继承了互联网规模的预训练，但生成式动作头（流匹配专家）却是在几百小时机器人演示上从零训练的。潜动作预训练意在通过从无动作标注视频中学习动作来弥合这一不对称，但现有工作要么把潜动作量化成离散码本（[LAPA](https://arxiv.org/abs/2410.11758)、[UniVLA](https://arxiv.org/abs/2505.06111)）—— 为细粒度连续控制设了上限 —— 要么在连续情形（[CLAM](https://arxiv.org/abs/2505.04999)、[villa-X](https://arxiv.org/abs/2507.23682)）中纯属经验，没有关于潜空间*何时*可恢复的理论。

**核心挑战。** 在无动作标注下，连续潜动作空间是否**可辨识至真实动作空间的仿射变换**？障碍在于相机自运动、光照、物体动态都会产生与机器人自身动作混淆的特征空间变化。

**方法（设计如此）。** 冻结 SigLIP 特征上的逆动力学编码器 + 前向解码器，配以 (i) VICReg 方差/协方差白化和 (ii) 共享自运动不变性正则，声称能把潜空间钉到仿射可辨识 —— 从而一个探针初始化的*冻结仿射适配器*就足以迁移一个纯在潜动作上预训练的流专家。

**关键实验（闸门）。** 若仿射可辨识性主张成立，岭回归探针 z↔a 应以高 R²（≥0.6）恢复真实动作。它没有（见下方结论）—— 该主张在当前形式下对本数据/设计被证伪。

## 本研究线为何暂停（结论）

go/no-go 闸门 —— 仿射探针均值 R² ≥ 0.6 —— 从未通过：

- **v1**（stride-1，绝对重建）：R² = 0.03。原因：在 10 Hz/128 px 下，一步运动在粗糙特征里几乎不可见，解码器抄了上一帧，z 坍缩到场景身份。
- **v2**（差分预测，stride 4）：recon 从 1.0 降到约 0.75（z 解释了真实特征变化方差），但仿射探针 R² 停在 0.04，*非线性*（MLP）探针也仅到 0.27 —— z 携带真实但微弱、非仿射、部分被污染的动作信息。
- 发现、修复并确认**不是**瓶颈的两个结构性 bug：z 坍缩（差分预测修复）和探针目标里的旋转合成错误（SO(3) 修复；合成数据上验证它能把完美仿射关系做成 R²≈0，但对真实数字改变约为 0）。
- 即便把非线性探针限制到 5 个视觉上可观测的维度（排除 roll/pitch），也仅到 R² ≈ 0.34。

**结论：** 对本分辨率/骨干下的 LIBERO，完全无动作标注的连续潜动作达不到可用程度的仿射可辨识性。复活本研究线需要：要么 CLAM 式少样本 grounding（混入一点真实动作监督 —— 对“完全无动作标注”主张是真实改动），要么非线性适配器（削弱可辨识性贡献）。用户选择转向创新点三，而非采纳其中任一。

## 哪些成果沉淀给其它研究线

- **差分预测 + 归一化损失（1.0 = 无用）** 约定 —— 被创新点三的 PrefixStateUpdater 直接复用。
- **shm 安全的数据路径**（`data.py::gpu_preprocess`、`_demo_frames_raw`）—— 被创新点三复用。
- **诊断优先的纪律**：在投入 GPU-周级下游训练前，先在便宜的机制级指标（探针 R²）上把关。

## 代码地图

| 文件 | 角色 | 关键部分 |
|---|---|---|
| `config.py` | 环境/工作区 | `$SIMVLA_CHECKPOINTS` 下的 `Workspace` 布局；`latest_checkpoint` |
| `make_splits.py` | Stage 0b | demo 级 p1/p10 低数据划分（`demos` 白名单） |
| `data.py` | 数据 | `FramePairDataset`；`gpu_preprocess`、`_demo_frames_raw`（shm 安全，与创新点三共享）；`iter_demos` |
| `models.py` | LAM | `FrozenVisionBackbone`；`LatentActionModel`（差分前向）；`variance_covariance_reg`、`shared_ego_augment`；`VectorQuantizerEMA`（离散消融） |
| `train_lam.py` | Stage 1 | LAM 训练；记录 `recon`/`var`/`cov`/`inv`/`delta_energy` |
| `label_z.py` | Stage 1b | 逐步 z 标注（+ `--ego_aug` 压力）；从 ckpt 自动检测 stride |
| `probe.py` | Stage 1c | 仿射 + `--nonlinear` MLP 探针；SO(3) 窗口目标；go/no-go 闸门；适配器初始化 |
| `run.py` | 驱动 | `python -m latent_action.run <stage>`；经 accelerate 多卡 |
| （集成） | — | `models/action_hub.py::LatentZActionSpace`/`LiberoZAdapterActionSpace`；`datasets/domain_handler/libero_z.py` |

## 本文档的维护约定

每次实验/修复往时间线日志追加一条编号/带日期的条目，格式同上（背景 → 命令 → 结果 → 诊断）。每次都更新顶部“当前状态”，使读者只看那一段即可了解当前状态；下方日志供细节与审计。框架变化时同步“参考”区块。有对应代码改动时一起提交。
