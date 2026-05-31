# 3. 方法

## 3.1 整体框架

我们提出 **SimVLA**，一种面向机器人操作任务的视觉-语言-动作（Vision-Language-Action, VLA）模型。给定自然语言指令和多相机观测图像，SimVLA 以端到端方式预测一段未来动作序列（action chunk）。模型由三个核心组件构成：（1）用于视觉与语言联合理解的 VLM 骨干网络；（2）以 VLM 特征为条件、基于扩散风格的动作变换器；（3）以流匹配（Flow Matching）为基础的训练目标，实现高质量、高效率的动作生成。

---

## 3.2 视觉-语言骨干网络

我们采用 **SmolVLM-500M-Instruct** 作为视觉-语言骨干网络。该模型将最多 $V=3$ 路相机图像与自然语言任务指令联合编码。每张图像首先经双三次插值缩放至 $512 \times 512$ 像素，再以 ImageNet 统计量进行归一化（$\mu = [0.485, 0.456, 0.406]$，$\sigma = [0.229, 0.224, 0.225]$）。多视角图像在序列维度拼接后送入模型，输出统一的视觉-语言特征序列 $\mathbf{z} \in \mathbb{R}^{T_\text{vlm} \times d_\text{vlm}}$，其中 $d_\text{vlm} = 576$。

训练初期（前 $10^3$ 步），骨干网络参数保持冻结，待动作头建立稳定梯度后再开放联合微调。

---

## 3.3 动作变换器

动作变换器接收加噪动作序列、机器人本体感受状态以及 VLM 条件特征，预测去噪速度场，遵循流匹配范式。

**输入编码。** 在每个去噪步，令 $\mathbf{x}_t \in \mathbb{R}^{T_a \times d_a}$ 为带噪动作序列，其中 $T_a = 10$ 为预测的动作步数，$d_a = 7$ 为 LIBERO 关节空间动作维度（末端执行器 $\Delta x, \Delta y, \Delta z, \Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}$ 及夹爪指令）。本体感受状态 $\mathbf{s} \in \mathbb{R}^{d_s}$（$d_s = 8$：末端位置、姿态、夹爪状态）和正弦时间嵌入 $\mathbf{e}_t \in \mathbb{R}^{32}$ 沿动作序列维度平铺后与 $\mathbf{x}_t$ 拼接，经线性投影得到初始隐表示：

$$\mathbf{h}_0 = \text{Linear}([\mathbf{x}_t \| \tilde{\mathbf{s}} \| \tilde{\mathbf{e}}_t]) \in \mathbb{R}^{T_a \times d_h}$$

其中 $d_h = 768$ 为变换器隐层维度，$[\cdot\|\cdot]$ 表示通道维拼接。

**VLM 条件融合。** VLM 输出 $\mathbf{z}$ 经线性层投影至 $d_h$ 维后追加至动作 token 序列，形成长度为 $T_a + T_\text{vlm}$ 的联合序列：

$$\mathbf{H} = [\mathbf{h}_0 \| \mathbf{W}_z \mathbf{z}] + \mathbf{E}_\text{pos}$$

其中 $\mathbf{E}_\text{pos}$ 为可学习位置嵌入。

**变换器主干。** $\mathbf{H}$ 经 $L = 12$ 个标准 Pre-LayerNorm 变换器块处理，每块包含 $N_h = 12$ 个注意力头（头维度 64）、MLP 扩展比 4（$d_\text{mlp} = 3072$）、GELU 激活函数及 Dropout（0.1）。最终取前 $T_a$ 个动作位置的输出，经线性映射层得到速度预测 $\hat{\mathbf{v}} \in \mathbb{R}^{T_a \times d_a}$。

---

## 3.4 流匹配训练目标

我们采用**条件流匹配**（Conditional Flow Matching, CFM）框架训练动作变换器。给定归一化后的真实动作序列 $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$ 及高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，定义线性插值：

$$\mathbf{x}_t = t\,\boldsymbol{\epsilon} + (1-t)\,\mathbf{a}, \quad t \sim \text{Beta}(1.5,\; 1.0)$$

其中 $t$ 截断至 $[0.001, 0.999]$。回归目标（速度场）为：

$$\mathbf{u}_t = \boldsymbol{\epsilon} - \mathbf{a}$$

训练损失为预测速度与目标速度的均方误差：

$$\mathcal{L}_\text{flow} = \mathbb{E}_{t,\boldsymbol{\epsilon}} \left\| \hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{z}, \mathbf{s}) - \mathbf{u}_t \right\|_2^2$$

**动作归一化。** 训练前，所有动作维度以数据集统计量进行 Z-score 归一化：

$$\mathbf{a}_\text{norm} = \frac{\mathbf{a} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}, \quad \epsilon = 10^{-6}$$

本体感受状态采用相同方式归一化。

---

## 3.5 自适应动作分块（可选）

为使模型在关键运动时刻（如接触、方向翻转）投入更多学习资源，我们提出**自适应动作分块**（Adaptive Action Chunking），以每步动作变化率对流匹配损失进行重加权。对于真实动作序列 $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$，计算归一化的逐步变化率：

$$r_\tau = \frac{\|\mathbf{a}_{\tau+1} - \mathbf{a}_\tau\|_2}{\max_{\tau'}\|\mathbf{a}_{\tau'+1} - \mathbf{a}_{\tau'}\|_2}, \quad \tau = 1, \ldots, T_a - 1$$

各步的损失权重定义为 $w_\tau = 0.5 + r_\tau \in [0.5, 1.5]$，加权流匹配损失为：

$$\mathcal{L}_\text{flow}^* = \mathbb{E}_{t,\boldsymbol{\epsilon}} \left\| \sqrt{w_\tau}\left(\hat{\mathbf{v}}_\tau - \mathbf{u}_{t,\tau}\right) \right\|_2^2$$

此外，我们引入辅助**动作边界预测头**——一个作用于动作特征的两层 MLP（$d_h \to d_h/2 \to 1$，SiLU 激活，输出层零初始化），对 stop-gradient 的变化率 $r_\tau$ 进行回归：

$$\mathcal{L}_\text{boundary} = \text{MSE}(\hat{r}_\tau,\; r_\tau^\text{stop-grad})$$

总训练损失为：

$$\mathcal{L} = \mathcal{L}_\text{flow}^* + \lambda\,\mathcal{L}_\text{boundary}, \quad \lambda = 0.1$$

关闭此选项时退化为标准均匀 MSE 目标，与原始设置完全兼容。

---

## 3.6 推理

推理阶段，通过 10 步欧拉积分求解学习到的流 ODE 生成动作。从 $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发：

$$\mathbf{x}_{t - \Delta t} = \mathbf{x}_t + \Delta t \cdot \hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{z}, \mathbf{s}), \quad \Delta t = \frac{1}{10}$$

最终 $\mathbf{x}_0$ 经数据集统计量反归一化，得到最终动作序列，以每 5 步重规划一次的频率发送至机器人控制器。

---

## 3.7 训练细节

我们使用 **AdamW** 优化器（$\beta_1 = 0.9$，$\beta_2 = 0.95$，权重衰减为 0，梯度裁剪阈值为 1.0）。学习率在前 2,000 步线性预热至 $10^{-4}$，之后余弦衰减至 $10^{-5}$。批大小为 32，使用 HuggingFace Accelerate 进行多卡分布式训练，最长训练 $10^6$ 步，每 $5 \times 10^4$ 步保存一次检查点。
