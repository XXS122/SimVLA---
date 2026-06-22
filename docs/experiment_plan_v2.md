# 实验计划 v2 —— 冲刺可发表的 TDS 论文（中文版）

> 接下来要做的实验路线图，已落实到两台机器。
> 机器：**sapi**（路径 `/datasets/...`，训练了 uniform）、**yyk**（路径 `/workspace/...`，训练了 tds）。
> 配合 `docs/results_log.md`（实时数据）和 `docs/tds_design.md`（方法本身）一起看。

---

## 0. 现状（已完成）

**训练（两个 run 都到 10 万步）：**
- **sapi**：uniform 全套 checkpoint —— `ckpt-20000 / 40000 / 60000 / 80000 / 100000`
- **yyk**：tds 全套 checkpoint —— `ckpt-20000 / 40000 / 60000 / 80000 / 100000`（正在续训到 20 万）

**核心结果（已同机验证）：** 在 40k 步、sapi 同机评估下，
TDS **89.75** vs uniform **61.9** = **+27.9** 平均。TDS@40k ≈ uniform@100k
⇒ 约 **2.4 倍样本效率**。

**实验 0（难度预测失败率）：** 在未饱和的 40k checkpoint 上显著（Spearman ρ=−0.315, p<0.05）。

---

## 1. 逻辑：两个主张，对应不同对照

| 主张 | 由什么证明 | 状态 |
|---|---|---|
| **A.** 多练难任务有用 | TDS vs **uniform** | ✅ 已完成（+27.9） |
| **B.** *免费*的数据难度 = *花钱*的模型难度 | TDS vs **实测难度** | ❌ **待补，最关键** |
| **C.** 增益来自信号的*方向*，不是任意扰动 | TDS vs **反转**权重 | ❌ 待补（便宜） |
| **D.** uniform 下限防止简单任务遗忘 | TDS vs **无clip** | ❌ 待补（便宜） |

主张 **B** 是审稿人必问的（"为什么不直接用 loss/成功率找难任务？"）。没有它，创新点就暴露了。
好在 B/C/D 都很便宜——**同一套训练流程，只换权重 csv**。

---

## 2. 两台机器的分工

| 机器 | 角色 | 现有资产 | 接下来干什么 |
|---|---|---|---|
| **sapi** | 评估主力 + 同机对比 | uniform 全套 + 已拷入 TDS 40k/100k | ① 补评 uniform 20k/60k/80k（凑曲线）② 已在跑 tds100k_onA |
| **yyk** | 训练主力 | tds 全套 + 在训 200k | ① 续训 200k ② 训完后评 TDS 20k/60k/80k ③ 跑对照实验训练 |

**为什么这么分**：主表的 40k/100k 用 **sapi 同机**最干净（已做/在做）；效率曲线的中间点（20/60/80k）让两台**并行各评自己原生的 checkpoint** 提速，再用同机的 40k/100k 当锚点校准两机微小偏差（验证已证两机一致，差异 <3 点，远小于 +27.9）。新的对照实验是新训练，交给训练机 **yyk**。

---

## 3. 优先级矩阵

| # | 实验 | 证明 | 论文产物 | 跑在哪 | 成本 |
|---|---|---|---|---|---|
| 1 | **效率曲线**（两个 run 评 20/40/60/80/100k） | A，量化提速 | 图 1（招牌） | sapi+yyk 并行 | 只评估（还差约 6 个点） |
| 2 | **实测难度 baseline** | **B（创新点防守）** | 主表一行 | yyk 训 + 评 | 1 次 40k 训练 + 评估 |
| 3 | **反转权重** 消融 | C | 消融表 | yyk 训 + 评 | 1 次 40k 训练 + 评估 |
| 4 | **无clip** 消融 | D | 消融表 | yyk 训 + 评 | 1 次 40k 训练 + 评估 |
| 5 | 课程式（easy→hard）*可选* | 替代调度 | 消融表 | yyk | 1 次训 + 评 |
| 6 | CRW / CRW+TDS 协同 *可选* | 三尺度框架 | 框架表 | yyk | 2 次训 + 评 |
| 7 | CALVIN 第二 benchmark *可选* | 泛化 | 第二主表 | —— | 大工程 |

**先做 1–4。** 5–7 是增强（5 便宜；6 扩展叙事；7 只在冲顶会且有空余时间时做——按 DoReMi 先例，LIBERO 单榜可发）。

---

## 4. 具体协议（精确命令）

> 所有对照/消融训练**只需 40k 步**（招牌对比点），**不要**浪费时间跑 100k。
> 除权重 csv 外，所有设置与 uniform/tds 完全一致。
> 通用参数：`batch=64`、`lr_coef=0.1`、`seed=0`、`NUM_WORKERS=0`（或扩 shm 后用 4）、`WANDB_MODE=offline`。

### 实验 1 —— 效率曲线（只评估）

把两个 run 的每个 checkpoint 都评一遍。已有：uniform 40k/100k、TDS 40k（sapi 同机）、TDS 100k（在跑）。还差：uniform 20k/60k/80k、TDS 20k/60k/80k。

**sapi 评 uniform 的剩余档（原生，不用拷贝）：**
```bash
cd /datasets/code/SimVLA---
pkill -f serve_smolvlm
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/exp_uniform/ckpt-60000 \
    --norm_stats ./norm_stats/libero_norm.json --port 8102 &
# 等 listening，tmux 另一窗格（先 conda activate）：
cd evaluation/libero && bash run_eval_seq.sh 8102 20 u60k 0 --no_video
# 重建合并 csv（自动合并不可靠）：
echo "task_name,sr" > u60k_sr_all.csv
grep -hv "^task_name" u60k_per_task_*.csv | grep -v "^$" >> u60k_sr_all.csv
```
（20k、80k 同理，把 `60000`/`u60k` 换成 `20000`/`u20k`、`80000`/`u80k`。）

**yyk 评 TDS 的剩余档（原生，不用拷贝，等 200k 训完或暂停时）：**
```bash
cd /workspace/hyj/code/SimVLA---
pkill -f serve_smolvlm
CUDA_VISIBLE_DEVICES=<空闲卡> python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/exp_tds/ckpt-60000 \
    --norm_stats ./norm_stats/libero_norm.json --port 8103 &
cd evaluation/libero && bash run_eval_seq.sh 8103 20 tds60k <空闲卡> --no_video
```

**产物**：画"平均成功率 vs 训练步数"图，uniform 一条线、TDS 一条线。TDS 在上方/左侧 = 样本效率胜。用 DoReMi 式说法报告："TDS 在约 4.X 万步就达到 uniform 10 万步的成功率 = Y 倍提速"。

### 实验 2 —— 实测难度 baseline（最关键对照，在 yyk 训）

用 uniform 实测的逐任务成功率构造采样权重（"问模型"那一派），再同样训练、和 TDS 比。

```bash
# 1) 从 uniform 的逐任务成功率造权重（模型耦合难度）
#    u40k_sr_all.csv 在 sapi，先拷到 yyk，或在 sapi 上生成 csv 再拷
python make_empirical_weights.py \
    --sr_csv evaluation/libero/u40k_sr_all.csv \
    --temperature 2.0 --clip_rho 0.5 \
    --out task_difficulty_empirical.csv

# 2) 完全照 TDS 训练，只换 csv（40k）
TDS_WEIGHTS_CSV=./task_difficulty_empirical.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_empirical

# 3) 评 40k，和 TDS@40k、uniform@40k 比
```

**怎么读结果**：
- TDS ≈ exp_empirical ⇒ **免费难度打平花钱难度** → 创新点成立（省掉整条评估回路 + 一次 baseline 训练）。
- 论文里写明：实测难度需要 训→评→重算权重→再训（约 2 倍算力 + 一整轮评估）；TDS 只要 CPU 几分钟。

### 实验 3 —— 反转权重消融（方向对照，在 yyk 训）

改成多采样**简单**任务。如果增益来自任意非均匀扰动,这样也该有用;实际应该变差(尤其 Goal/Long)。

```bash
# 用同样的转变统计，--invert 翻转方向（这个开关我已经加好了）
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --alpha 0 --beta 0.5 --gamma 0.5 --invert \
    --out task_difficulty_inverted.csv

TDS_WEIGHTS_CSV=./task_difficulty_inverted.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_inverted
```

**预期**：Long/Goal 相对 uniform 下降 → 信号的方向携带信息（不是正则化噪声）。

### 实验 4 —— 无clip 消融（下限对照，在 yyk 训）

```bash
python transition_density_stats.py --data_root $LIBERO_DATASETS \
    --suites libero_spatial libero_object libero_goal libero_10 \
    --alpha 0 --beta 0.5 --gamma 0.5 --clip_rho 0 \
    --out task_difficulty_noclip.csv

TDS_WEIGHTS_CSV=./task_difficulty_noclip.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_noclip
```

**预期**：Spatial/Object 下降（简单任务被饿死），而有 clip 的 TDS 不掉 → 下限控制失败模式。

### 实验 5 —— 课程式 *（可选，yyk）*

用 `tds_sampling.mix_with_uniform` 生成 uniform→TDS 的中间权重 csv，分阶段训。预期：不优于静态 TDS → 主方案用静态版。

### 实验 6 —— CRW / 协同 *（可选，三尺度框架，yyk）*

```bash
# 只开 CRW + 边界头（步级）
USE_ADAPTIVE_CHUNKING=true \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_crw

# CRW + TDS（协同：TDS 是否让边界头学得更好？）
USE_ADAPTIVE_CHUNKING=true TDS_WEIGHTS_CSV=./task_difficulty_2comp.csv \
NUM_WORKERS=0 ITERS=40000 SAVE_INTERVAL=20000 WANDB_MODE=offline \
  bash train_smolvlm_small.sh 64 0.1 ./runs/exp_all
```
比较 exp_crw 与 exp_all 的边界头 AUROC，报告协同效应。

### 实验 7 —— CALVIN *（可选，冲顶会增强）*

大工程（新数据格式 + 新评估器）。只在冲顶会且有富余时间时做，否则写进 future work——LIBERO 单榜可发。

---

## 5. 操作避坑速查（我们已经踩过的）

- **从 yyk 拷 checkpoint 到 sapi**：config.json 里写死了 yyk 的 backbone 路径，加载前必须改：
  ```bash
  sed -i 's|/workspace/hyj/simvla/model/smolvlm-500M|/datasets/models/smolvlm/SmolVLM-500M-Instruct|g' \
      runs/<exp>/ckpt-<N>/config.json
  ```
  （或一劳永逸：在 sapi 上 `ln -s /datasets/models/smolvlm/SmolVLM-500M-Instruct /workspace/hyj/simvla/model/smolvlm-500M`）
- **Docker `/dev/shm` 太小 → DataLoader Bus error**：用 `NUM_WORKERS=0`（或 `mount -o remount,size=32g /dev/shm` 后用 `NUM_WORKERS=4`）。
- **评估秒退 / `(no per-task csvs)`**：server 没在那个端口起，或 conda 环境没激活。先起 server、等 `listening`、再评。
- **评估放 tmux 里跑**；误按 Ctrl+C 会杀掉客户端（server 还在）。
- **每个 run 都用 20 trials**（和已有数字一致）。
- **每台机器只有一张可用卡 ⇒ 不能同时训练和评估**（会抢资源）。串行,或一台训练时另一台评估。
- **`<前缀>_sr_all.csv` 一律手动重建**（自动合并有 bug）。

---

## 6. 要填的论文表格

**表 1 —— 主对比 @ 40k（sapi 同机）：**

| 方法 @ 40k | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| uniform | 73.0 | 95.5 | 38.5 | 40.5 | 61.9 |
| 实测难度（实验2） | — | — | — | — | — |
| **TDS（本文）** | 97.5 | 98.5 | 92.0 | 71.0 | **89.75** |

**表 2 —— 消融 @ 40k**：TDS / 反转 / 无clip /（课程式）。

**图 1 —— 效率曲线**：平均 SR vs 步数，uniform vs TDS（实验1）。

**图 2 —— 实验0 散点图**：难度 d_k vs 逐任务 SR（ρ=−0.32）。

**（可选）表 3 —— 框架**：uniform / +CRW / +TDS / +两者（实验6）。

---

## 7. 设计依据（有调研支撑）

- **DoReMi**（数据混合重加权，NeurIPS'23）：单数据集（The Pile）+ 深度分析就够；
  招牌指标"2.6× 更少步数达到 baseline"；对照=uniform/默认权重；用了*代理模型*（TDS 更便宜）。
  还有：重加权中等难度域产生*正迁移、抬升所有域*——这解释了我们 Spatial +24.5。
- **优先经验回放 / OHEM / Re-Mix**：难度来自*模型*（TD-error / loss / 参考模型）——我们的实测难度 baseline 代表这一派。
- **课程学习 / RFCL / PLR**：标准对照=非课程（uniform）版；标准指标=效率曲线（SR vs 步数）。
- **VLA benchmark（2026）**：LIBERO 必报；CALVIN 对*模型*论文越来越被期待，但方法论文用 LIBERO 单榜 + 硬核对照消融即可。

参考文献：DoReMi (arXiv:2305.10429)；OpenVLA-OFT (arXiv:2502.19645)；
Reverse-Forward Curriculum (ICLR'24)；LIBERO (NeurIPS'23)。
