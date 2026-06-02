# Method Section Draft — Change-Rate-Weighted Adaptive Action Chunking

> Draft of the Method section for the SimVLA adaptive-chunking contribution.
> English prose intended for direct inclusion in the paper; inline `# 注释` are
> Chinese notes for the authors and should be removed before submission.

---

## 3. Method

### 3.1 Preliminaries: Flow-Matching Action Chunking

SimVLA predicts a chunk of $T$ future actions
$\mathbf{a} = (a_1,\dots,a_T)$, $a_i \in \mathbb{R}^{D}$, conditioned on
fused vision-language features $\mathbf{z}$ produced by the SmolVLM backbone
and the current proprioceptive state $s$. The action head is trained with a
conditional flow-matching objective.

Given a clean (normalized) action chunk $\mathbf{a}_0$ and Gaussian noise
$\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$, we form the linear interpolant at
flow time $\tau\in(0,1]$,

$$
\mathbf{x}_\tau = \tau\,\boldsymbol{\epsilon} + (1-\tau)\,\mathbf{a}_0,
\qquad
\mathbf{u}_\tau = \boldsymbol{\epsilon} - \mathbf{a}_0,
$$

where $\mathbf{u}_\tau$ is the target velocity field. The network
$v_\theta(\mathbf{x}_\tau, \tau, \mathbf{z}, s)$ regresses this velocity, and
the standard objective averages the squared error **uniformly** over all
chunk steps and action dimensions:

$$
\mathcal{L}_{\text{FM}}
= \mathbb{E}_{\tau,\boldsymbol{\epsilon}}
\Big[\tfrac{1}{TD}\textstyle\sum_{t=1}^{T}\sum_{d=1}^{D}
\big(v_\theta^{(t,d)} - u_\tau^{(t,d)}\big)^2 \Big].
\tag{1}
$$

# 注：τ 在代码里用 Beta(1.5,1) 采样，与本节推导无关，可在实验细节里提。

### 3.2 Motivation: Not All Steps Are Equal

The uniform average in Eq. (1) implicitly assumes every timestep within a
chunk is equally important to imitate. In manipulation this is rarely true.
During **free-space transit** the end-effector moves smoothly and a small
prediction error is harmless; during **contact-rich or direction-reversing**
moments (grasping, insertion, placing) the same magnitude of error can flip a
success into a failure. A model trained with the uniform objective therefore
spends representational capacity matching easy, low-stakes steps as hard as it
matches the few decisive ones.

We propose to make the chunking objective **change-rate aware**: the training
signal is concentrated on steps where the demonstrated trajectory changes
rapidly, and the model is additionally asked to *predict* this change rate,
yielding an explicit notion of "where the decisive moments are" inside a chunk.

### 3.3 Change-Rate Weighting (CRW)

For each ground-truth (normalized) chunk we measure the local change rate as
the norm of consecutive action differences,

$$
\delta_t = \lVert a_{0,\,t+1} - a_{0,\,t}\rVert_2,\quad t=1,\dots,T-1,
\qquad \delta_1 \leftarrow \delta_1 \;\text{(repeat for } t{=}0\text{)},
$$

so that $\delta \in \mathbb{R}^{T}$ aligns with the $T$ chunk steps.
# 注：代码中是把 δ_1 复制到首位，保证长度对齐 T。

We normalize per chunk and map to a bounded weight,

$$
\hat{\delta}_t = \frac{\delta_t}{\max_{t'}\delta_{t'} + \varepsilon}\in[0,1],
\qquad
w_t = w_{\min} + (w_{\max}-w_{\min})\,\hat{\delta}_t,
\tag{2}
$$

with $[w_{\min}, w_{\max}] = [0.5, 1.5]$ by default. The bounded range is
deliberate: $w_{\min}>0$ guarantees smooth steps are never fully ignored
(avoiding trajectory drift), while $w_{\max}/w_{\min}=3$ gives decisive steps
up to a threefold larger gradient. The weighted flow-matching loss is

$$
\mathcal{L}_{\text{CRW}}
= \mathbb{E}_{\tau,\boldsymbol{\epsilon}}
\Big[\tfrac{1}{TD}\textstyle\sum_{t=1}^{T} w_t \sum_{d=1}^{D}
\big(v_\theta^{(t,d)} - u_\tau^{(t,d)}\big)^2 \Big].
\tag{3}
$$

The weights $w_t$ are computed from the targets and treated as constants
(no gradient flows through them), so CRW is a pure re-weighting of Eq. (1)
and adds **zero** inference cost.

### 3.4 Auxiliary Boundary Prediction Head

Re-weighting shapes *where* the model is accurate, but the change rate itself
is a useful, transferable signal: it marks the natural sub-segment boundaries
of a chunk. We therefore attach a lightweight **boundary head** $g_\phi$ — a
two-layer MLP — on top of the decoded per-step action features
$h_t\in\mathbb{R}^{H}$ (the hidden states just before the velocity readout):

$$
b_t = g_\phi(h_t)\in\mathbb{R},\qquad t=1,\dots,T.
$$

It is supervised to regress the (detached) normalized change rate,

$$
\mathcal{L}_{\text{bnd}}
= \tfrac{1}{T}\textstyle\sum_{t=1}^{T}\big(b_t - \operatorname{sg}[\hat{\delta}_t]\big)^2,
\tag{4}
$$

where $\operatorname{sg}[\cdot]$ is the stop-gradient. The head's output layer
is zero-initialized so that it does not perturb the action head early in
training. Predicting $\hat\delta_t$ forces the shared trunk to encode an
explicit, per-step sense of motion abruptness, which (i) regularizes the action
features and (ii) provides a ready-to-use boundary signal for variable-length
execution at deployment (see §3.6).

### 3.5 Training Objective

The full objective combines the weighted flow-matching loss with the auxiliary
boundary loss,

$$
\mathcal{L} = \mathcal{L}_{\text{CRW}} + \lambda\,\mathcal{L}_{\text{bnd}},
\tag{5}
$$

with $\lambda = 0.1$ by default. Setting $w_t\equiv 1$ and $\lambda = 0$
recovers the original SimVLA objective exactly, so the method is a strict,
opt-in generalization that is backward compatible with existing checkpoints.

### 3.6 Inference (Optional Variable-Length Execution)

At deployment the velocity head is integrated as usual to produce the action
chunk; the boundary head adds no cost to this path. Optionally, the predicted
boundary scores $\{b_t\}$ can drive **variable-length re-planning**: execute
the chunk until the first step whose score exceeds a threshold
$b_t > \beta$ (a high-change / decisive moment), then re-observe and re-plan.
This lets the policy commit to long open-loop segments during smooth motion
while re-planning frequently around contacts, trading compute for reactivity
without retraining. # 注:这是后续可做的推理扩展,主实验先用定长执行即可。

---

## Implementation Notes (for the experiments section)

- Backbone: SmolVLM-500M-Instruct; action head as in SimVLA (concat / AdaLN).
- Defaults: $[w_{\min},w_{\max}]=[0.5,1.5]$, $\lambda=0.1$, $\varepsilon=10^{-6}$.
- $\delta_t$ is computed in the normalized action space; both CRW and the
  boundary target share the same $\hat\delta_t$.
- Flags: `--use_adaptive_chunking`, `--chunk_loss_weight <λ>`.
- Added parameters: only the two-layer boundary head ($H\!\to\!H/2\!\to\!1$);
  negligible relative to the backbone.
