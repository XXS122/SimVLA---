# 自适应动作分块 (Adaptive Action Chunking)

## 动机

标准流匹配（Flow Matching）目标对动作序列中的每一步施加均匀的均方误差（MSE）损失，隐含假设序列内所有时刻同等重要。然而在机器人操作中，关键运动时刻（如发生接触、运动方向翻转）往往集中在序列的少数几步上；对这些步施加与平稳步相同的权重，会使模型在精细控制阶段学习不足。为此，我们提出**自适应动作分块**（Adaptive Action Chunking），依据真实动作的逐步变化率对流匹配损失进行重加权，并引入一个辅助边界预测头，使模型显式感知动作序列内部的转折结构。该模块为可选项；关闭时即退化为标准均匀 MSE 目标，与原始训练完全兼容。

## 预备记号

设归一化后的真实动作序列为 $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$，其中 $T_a$ 为动作步数、$d_a$ 为单步动作维度。在条件流匹配框架下，模型预测速度场 $\hat{\mathbf{v}}_\tau$，回归目标为 $\mathbf{u}_{t,\tau} = \boldsymbol{\epsilon}_\tau - \mathbf{a}_\tau$，基线损失为均匀加权的均方误差。本文的创新在于对该损失的逐步加权及辅助边界监督。

## 逐步变化率加权

我们以相邻动作步之间的 L2 距离度量动作的"变化剧烈程度"。逐步变化率定义为：

$$c_\tau = \begin{cases} \|\mathbf{a}_2 - \mathbf{a}_1\|_2, & \tau = 1,\\[2pt] \|\mathbf{a}_\tau - \mathbf{a}_{\tau-1}\|_2, & \tau = 2, \ldots, T_a, \end{cases}$$

即对长度为 $T_a$ 的序列计算 $T_a - 1$ 个相邻步的后向差分 L2 范数，并将首步以第一个差分值复制填充，使变化率长度与动作序列对齐（当 $T_a = 1$ 时变化率退化为全 $0$）。随后对每个样本以其自身最大值进行归一化：

$$\tilde{c}_\tau = \frac{c_\tau}{\max\!\bigl(\max_{\tau'} c_{\tau'},\ \varepsilon\bigr)}, \quad \varepsilon = 10^{-6}, \qquad \tilde{c}_\tau \in [0, 1]$$

归一化后的变化率映射为逐步损失权重：

$$w_\tau = 0.5 + \tilde{c}_\tau \in [0.5,\, 1.5]$$

变化最剧烈的步获得最高权重 $1.5$，变化最平缓的步获得最低权重 $0.5$。加权后的流匹配损失为（权重作用于逐步平方误差）：

$$\mathcal{L}_\text{vel} = \mathbb{E}_{t,\boldsymbol{\epsilon}} \left[ \frac{1}{T_a} \sum_{\tau=1}^{T_a} w_\tau \,\bigl\|\hat{\mathbf{v}}_\tau - \mathbf{u}_{t,\tau}\bigr\|_2^2 \right]$$

## 边界预测头

为使模型显式建模动作序列内部的转折结构，我们在动作特征上附加一个轻量的**边界预测头**（Chunk Boundary Head）。设变换器最后一层归一化后输出的逐步动作特征为 $\mathbf{f}_\tau \in \mathbb{R}^{d_h}$（即用于解码速度场的同一特征），边界头为两层 MLP：

$$\hat{r}_\tau = \mathbf{W}_2\,\operatorname{SiLU}(\mathbf{W}_1 \mathbf{f}_\tau), \quad \mathbf{W}_1 \in \mathbb{R}^{(d_h/2) \times d_h},\ \mathbf{W}_2 \in \mathbb{R}^{1 \times (d_h/2)}$$

其输出层权重与偏置零初始化，使边界预测在训练初期保持中性。边界头以 stop-gradient 的归一化变化率 $\tilde{c}_\tau$ 为回归目标：

$$\mathcal{L}_\text{bnd} = \frac{1}{T_a}\sum_{\tau=1}^{T_a} \bigl(\hat{r}_\tau - \operatorname{sg}(\tilde{c}_\tau)\bigr)^2$$

其中 $\operatorname{sg}(\cdot)$ 为 stop-gradient 算子，确保边界监督不反向影响变化率本身。

## 总目标

总训练损失为加权流匹配损失与边界辅助损失之和：

$$\mathcal{L} = \mathcal{L}_\text{vel} + \lambda\, \mathcal{L}_\text{bnd}, \qquad \lambda = 0.1$$

其中 $\lambda$ 为边界辅助损失权重（默认 $0.1$）。当关闭自适应分块时，所有 $w_\tau \equiv 1$ 且移除边界头，目标退化为标准均匀 MSE，保证与基线设置的向后兼容性。
