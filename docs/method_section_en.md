# 3. Method

## 3.1 Overview

We present **SimVLA**, a Vision-Language-Action (VLA) model for robotic manipulation that couples a pretrained vision-language model (VLM) with a flow-matching action transformer. Given a language instruction and multi-camera observations, SimVLA predicts a chunk of future actions in an end-to-end fashion. The model consists of three components: (1) a VLM backbone for visual and language grounding, (2) a diffusion-style action transformer that decodes actions conditioned on VLM features, and (3) a flow-matching training objective that enables efficient, high-quality action generation.

---

## 3.2 Vision-Language Backbone

We adopt **SmolVLM-500M-Instruct** as our vision-language backbone. The model encodes up to $V=3$ camera views jointly alongside the natural-language task instruction. Each image is resized to $512 \times 512$ pixels via bicubic interpolation and normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). Multiple views are concatenated along the sequence dimension before being fed to the model, producing a unified visual-language feature sequence $\mathbf{z} \in \mathbb{R}^{T_\text{vlm} \times d_\text{vlm}}$, where $d_\text{vlm} = 576$.

During early training, the VLM backbone parameters are frozen (for the first $10^3$ steps) to allow the action head to bootstrap stable gradients before joint fine-tuning commences.

---

## 3.3 Action Transformer

The action transformer takes noisy actions, robot proprioception, and VLM conditioning features as input and predicts the denoising velocity field, following the flow-matching paradigm.

**Input encoding.** At each denoising step, let $\mathbf{x}_t \in \mathbb{R}^{T_a \times d_a}$ denote the noisy action chunk, where $T_a = 10$ is the number of predicted action steps and $d_a = 7$ for the LIBERO joint-space action (end-effector $\Delta x, \Delta y, \Delta z$, $\Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}$, and gripper command). The proprioceptive state $\mathbf{s} \in \mathbb{R}^{d_s}$ ($d_s = 8$: end-effector position, orientation, gripper state) and a sinusoidal time embedding $\mathbf{e}_t \in \mathbb{R}^{32}$ are tiled along the action-sequence dimension and concatenated with $\mathbf{x}_t$:

$$\mathbf{h}_0 = \text{Linear}([\mathbf{x}_t \| \tilde{\mathbf{s}} \| \tilde{\mathbf{e}}_t]) \in \mathbb{R}^{T_a \times d_h}$$

where $d_h = 768$ is the transformer hidden size and $[\cdot \| \cdot]$ denotes channel-wise concatenation.

**VLM conditioning.** The VLM output $\mathbf{z}$ is projected to $d_h$ dimensions via a linear layer and appended to the action token sequence, yielding a joint sequence of length $T_a + T_\text{vlm}$:

$$\mathbf{H} = [\mathbf{h}_0 \| \mathbf{W}_z \mathbf{z}] + \mathbf{E}_\text{pos}$$

where $\mathbf{E}_\text{pos}$ is a learned positional embedding.

**Transformer blocks.** $\mathbf{H}$ is processed by $L = 12$ standard Pre-LayerNorm transformer blocks, each with $N_h = 12$ attention heads (head dimension 64), MLP expansion ratio 4 ($d_\text{mlp} = 3072$), GELU activation, and dropout 0.1. After the final layer, only the $T_a$ action positions are retained and projected by a linear head to produce the velocity prediction $\hat{\mathbf{v}} \in \mathbb{R}^{T_a \times d_a}$.

---

## 3.4 Flow Matching Training Objective

We train the action transformer using the **Conditional Flow Matching** (CFM) framework. Given a ground-truth normalized action chunk $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$ and Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, we define the interpolant:

$$\mathbf{x}_t = t\,\boldsymbol{\epsilon} + (1-t)\,\mathbf{a}, \quad t \sim \text{Beta}(1.5,\, 1.0)$$

where $t$ is clipped to $[0.001, 0.999]$. The regression target (velocity field) is:

$$\mathbf{u}_t = \boldsymbol{\epsilon} - \mathbf{a}$$

The training loss is the mean squared error between the predicted and target velocity fields:

$$\mathcal{L}_\text{flow} = \mathbb{E}_{t,\boldsymbol{\epsilon}} \left\| \hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{z}, \mathbf{s}) - \mathbf{u}_t \right\|_2^2$$

**Action normalization.** Prior to training, all action dimensions are normalized to zero mean and unit variance using dataset statistics:

$$\mathbf{a}_\text{norm} = \frac{\mathbf{a} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}, \quad \epsilon = 10^{-6}$$

Proprioceptive states are normalized identically.

---

## 3.5 Adaptive Action Chunking (Optional)

To focus learning on critical transition moments (e.g., contacts and direction reversals), we optionally apply **Adaptive Action Chunking** by reweighting the flow-matching loss with per-step change rates. For a ground-truth action chunk $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$, we compute the normalized step-wise change rate:

$$r_\tau = \frac{\|\mathbf{a}_{\tau+1} - \mathbf{a}_\tau\|_2}{\max_{\tau'}\|\mathbf{a}_{\tau'+1} - \mathbf{a}_{\tau'}\|_2}, \quad \tau = 1, \ldots, T_a - 1$$

Each step is assigned a loss weight $w_\tau = 0.5 + r_\tau \in [0.5, 1.5]$, so the weighted flow-matching loss becomes:

$$\mathcal{L}_\text{flow}^* = \mathbb{E}_{t,\boldsymbol{\epsilon}} \left\| \sqrt{w_\tau}\left(\hat{\mathbf{v}}_\tau - \mathbf{u}_{t,\tau}\right) \right\|_2^2$$

An auxiliary **chunk boundary head** — a two-layer MLP ($d_h \to d_h/2 \to 1$, SiLU activation, zero-initialized output layer) applied to the action features — regresses the detached change rate $r_\tau$ as a boundary score:

$$\mathcal{L}_\text{boundary} = \text{MSE}(\hat{r}_\tau,\; r_\tau^\text{stop-grad})$$

The total loss is $\mathcal{L} = \mathcal{L}_\text{flow}^* + \lambda\,\mathcal{L}_\text{boundary}$ with $\lambda = 0.1$. Disabling this option recovers the standard uniform-MSE objective, remaining fully backward compatible.

---

## 3.6 Inference

At inference time, actions are generated by solving the learned flow ODE with 10 Euler steps. Starting from $\mathbf{x}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathbf{x}_{t - \Delta t} = \mathbf{x}_t + \Delta t \cdot \hat{\mathbf{v}}(\mathbf{x}_t, t, \mathbf{z}, \mathbf{s}), \quad \Delta t = \frac{1}{10}$$

The resulting $\mathbf{x}_0$ is denormalized using the stored dataset statistics to recover the final action chunk, which is then sent to the robot controller at a replanning interval of 5 steps.

---

## 3.7 Training Details

We train SimVLA with the **AdamW** optimizer ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $= 0$, gradient clipping $= 1.0$). The learning rate follows a linear warmup over 2,000 steps to $10^{-4}$, followed by cosine decay to $10^{-5}$. We use a batch size of 32 and train for up to $10^6$ iterations using HuggingFace Accelerate for multi-GPU distributed training. Checkpoints are saved every 50,000 steps.
