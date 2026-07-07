# 研究总览 —— 基于 SimVLA 的 VLA 架构创新

本仓库中所有在 SimVLA 基线之上开展的研究工作的统一入口。共有两条相互独立的研究线，各自维护一份实时进度日志。请先阅读本文件了解全局，再进入各自的详细日志。

| 研究线 | 目录 | 状态 | 日志 |
|---|---|---|---|
| **创新点三** —— 流式递归前缀编码 | `streaming_prefix/` | **进行中** —— 时延收益已确认，漂移修复进行中 | [`streaming_prefix/PROGRESS.md`](streaming_prefix/PROGRESS.md) |
| **创新点四** —— 可辨识连续潜动作预训练 | `latent_action/` | **暂停** —— 探针闸门未通过（结论见其日志） | [`latent_action/PROGRESS.md`](latent_action/PROGRESS.md) |

两条线都建立在同一个冻结的 SmolVLM-500M 骨干和 SimVLA 已有的同一套 LIBERO 数据流水线之上，并且都刻意**复用**现有模块（`models/transformer_smolvlm.py`、`datasets/…`、`models/action_hub.py`）而非另起炉灶。

---

## 两条线所依托的 SimVLA 基线

SimVLA（见本仓库 `readme.md`）是一个紧凑的 VLA 模型：

```
   多视角图像 ─────────┐
                       ├─► SmolVLM-500M（SigLIP 视觉 + 连接器 +
   语言指令 ───────────┘      Idefics3 语言模型）  ──► vlm_features
                                                             │
                                    本体感知 proprio ──┐      │
                              加噪动作 x_t ────────────┼──────┤
                                       流时间 t ───────┘      ▼
                          SmolVLMActionTransformer（流匹配动作头）
                                                             │
                                                       速度场 v_t
                                              （Euler 积分还原为动作）
```

- 骨干：`models/modeling_smolvlm_vla.py::forward_vlm_efficient`
  （视觉塔 → 连接器 → 融合图文 token → 语言模型前向 → `vlm_features`）。
- 动作头：`models/transformer_smolvlm.py::SmolVLMActionTransformer`
  （流匹配；两种条件注入模式 —— concat 和 AdaLN）。
- 动作空间 + 归一化：`models/action_hub.py`。
- 数据：LIBERO HDF5，经 `datasets/domain_handler/libero_hdf5.py`。

两条研究线攻击这条流水线上两个不同的结构性问题：

- **创新点三** 攻击*推理数据流*：尽管相邻帧几乎不变，骨干每个控制步都从头重跑一遍。能否用一个可学习的递归状态更新替换掉其中最昂贵的部分？
- **创新点四** 攻击*动作头初始化*：流匹配动作头是在机器人演示上从零训练的，而骨干却享有互联网规模的预训练。能否让动作头通过可辨识的潜动作在无动作标注的视频上预训练？

---

## 共享基础设施（构建一次，两条线复用）

在构建创新点四时搭建，后被创新点三复用：

- **`reserve_gpu.py`** —— 面向共享服务器的“纯显存占用”守护脚本（把其他用户挡在你正在训练的 GPU 之外，且不偷占 SM 算力）。用法见其 docstring。
- **`latent_action/data.py::gpu_preprocess`** 和 `_demo_frames_raw` ——
  DataLoader 的 worker 只传输原始 uint8 帧（约 100 KB/样本），resize 和归一化在 GPU 上完成，从而绕开 Docker 默认 64 MB 的 `/dev/shm` 在“传输预处理后浮点张量”时引发的 bus error。两条线都用它。
- **`paths.env` 约定** —— 所有脚本从环境变量读取路径
  （`SIMVLA_SMOLVLM_MODEL`、`LIBERO_DATASETS`、`SIMVLA_CHECKPOINTS`、`WANDB_*`、`CUDA_DEVICES`、`NUM_GPUS`、`SIMVLA_Z_DIM`）。见 `paths.env.example`。

## 硬件 / 环境

- 共享服务器，8× NVIDIA A800-SXM4-80GB；工作固定在 GPU 6，通过 `CUDA_DEVICES=6` 指定。
- 本地 SmolVLM-500M 位于 `$SIMVLA_SMOLVLM_MODEL`；LIBERO（4 个子集，约 2000 条 demo）位于 `$LIBERO_DATASETS`。
- Docker 容器，`/dev/shm` 较小 —— 见上文 gpu_preprocess 说明。
- W&B 项目 `simvla`（用户 `xxshyj`）。

## 反复出现的工程教训（适用于本仓库未来任何研究线）

1. **小 `/dev/shm` + 多 worker DataLoader = SIGBUS。** 通过 IPC 传原始 uint8，在 GPU 上做预处理。（创新点四，commit `bc6cdc3`。）
2. **当输入几乎不变时，预测“差分”优于预测“绝对值”。** 一个能抄输入的模型一定会抄，导致条件信号得不到梯度。改为预测*变化量*，并把损失归一化，使 1.0 = “预测无变化”。LAM（创新点四）和 PrefixStateUpdater（创新点三）都用了这一招。
3. **Teacher forcing ≠ 部署。** 任何在推理时递归作用于自身输出的模型，都必须*递归地*训练（rollout + scheduled sampling），否则会漂移。（创新点三，Phase C。）
4. **测中间量，别只测最终指标。** LIBERO 成功率已饱和；这里每一个主张在动成功率之前，都先在机制级诊断指标上验证（探针 R²、相对 1.0 基线的 recon、漂移曲线、时延拆解）。

## Git / 溯源

所有工作在分支 `claude/vla-research-innovations-wjijet` 上进行。提交在 GitHub 上显示为 “Unverified”（执行容器内无签名密钥）；作者/提交者身份是正确的。各线的 commit 表见各自日志。
