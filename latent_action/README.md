# 可辨识连续潜动作流匹配预训练

构建于 SimVLA 之上的研究流水线。待检验的科学主张：由带 (i) 方差/协方差白化和 (ii) 共享自运动不变性正则的逆动力学编码器，从无动作标注视频中提取的潜动作，**可辨识至真实动作空间的仿射变换** —— 因此一个探针初始化的冻结仿射适配器就足以把一个纯在潜动作上预训练的流匹配动作专家迁移到真实机器人控制，且收益集中在低数据区间。

> 状态说明：本流水线已完整实现并端到端验证，但 go/no-go 探针闸门在 LIBERO 上未通过（详见 `PROGRESS.md`）。下面的配方保留完整，供复现或在其它数据集上重试。

## 准备

```bash
cp paths.env.example paths.env   # 编辑路径，切勿提交 paths.env
source paths.env
```

以下所有内容都写入 `$SIMVLA_CHECKPOINTS`：

```
$SIMVLA_CHECKPOINTS/
  metas/            libero_train.json, libero_train_p1.json, libero_train_p10.json
  norm_stats/       libero_norm.json
  lam/              lam_final.pt, lam_config.json
  z_labels/         z_labels.h5, z_labels_stress.h5, libero_train_z.json
  probe/            probe.npz (+ .json report), probe_stress.npz
  runs/             baseline_* / pretrain_flow / finetune_*
```

## 流水线

### Stage 0 —— 元数据、归一化统计、基线对照线

```bash
./run_pipeline.sh meta                      # 扫描 $LIBERO_DATASETS
./run_pipeline.sh norm-stats
./run_pipeline.sh splits                    # p1 (1%) 和 p10 (10%) demo 级划分

./run_pipeline.sh baseline --split p100     # 20 万 iters（默认）
./run_pipeline.sh baseline --split p10      # 6 万
./run_pipeline.sh baseline --split p1       # 2 万
```

这三个基线是主图的“随机初始化”那条线（成功率 vs demo 数）。

### Stage 1 —— 潜动作模型 + go/no-go 闸门

```bash
./run_pipeline.sh train-lam                       # 单张 A100 约 2-3 天
./run_pipeline.sh label                           # z 标签 + 预训练元数据
./run_pipeline.sh probe                           # 均值 R^2 >= 0.6 才 PASS
```

**早期健康检查（train-lam 头一小时）：** `recon` 是归一化的 —— 1.0 表示解码器预测零变化、z 无用；它必须明显降到 1 以下（例如 ≤0.7）。若约 1 小时后仍贴着 1.0，直接终止并升级（更大的 `--stride`、更大的 `--lam_dim/--enc_depth`），不要干等探针。

**停止规则：** 若在合理调整 `--var_coef/--cov_coef/--inv_coef` 后 `probe` 报告均值 R² < 0.6，可辨识性前提不成立 —— 就此停止并转向。

理论预言的证伪测试（自运动压力）：

```bash
./run_pipeline.sh label --ego_aug                 # 压力 z 标签
./run_pipeline.sh probe --stress                  # inv_coef>0 时 R^2 应保持高，
                                                  # 为 0 时应崩溃
```

LAM 的离散（VQ）消融 —— LAPA/UniVLA 式量化潜动作的同规模替身：

```bash
./run_pipeline.sh train-lam --use_vq --output_dir $SIMVLA_CHECKPOINTS/lam_vq
```

### Stage 2 —— 流专家预训练 + 下游微调

```bash
./run_pipeline.sh pretrain                        # action_mode=latent_z，10 万 iters

# 预训练 vs 从零，三种数据规模（主图）
./run_pipeline.sh finetune --split p1
./run_pipeline.sh finetune --split p1  --from_scratch
./run_pipeline.sh finetune --split p10
./run_pipeline.sh finetune --split p10 --from_scratch
./run_pipeline.sh finetune --split p100
./run_pipeline.sh finetune --split p100 --from_scratch
```

`finetune` 以 `action_mode=libero_z_adapter` 运行：流专家继续在 z 空间生成；一个**冻结的**仿射适配器（从 `probe.npz` 初始化）把动作映入/映出。`--train_adapter` 解冻它（消融）。种子：追加 `--seed 1` 等（转发给 train_smolvlm.py）。

### 评测

```bash
./run_pipeline.sh serve --ckpt $SIMVLA_CHECKPOINTS/runs/finetune_p1_pretrained/ckpt-20000
# 然后从 evaluation/libero/ 运行 LIBERO 客户端（见其 README）
```

## 多卡 / 续训 / wandb

- 环境里 `NUM_GPUS>1` 会把训练阶段切换为 `accelerate launch --num_processes $NUM_GPUS`。
- `SIMVLA_RESUME_CKPT=<ckpt 目录>` 从该 checkpoint 续训（自动加上 `--models <dir> --resume`）。
- 只要 `WANDB_API_KEY` 非空即启用 wandb 记录；项目名来自 `WANDB_PROJECT`。

## 论文产物对照表

| 图/表 | 产出数字的命令 |
|---|---|
| 主图：成功率 vs demo 数（喇叭口） | 6 次 `finetune` 运行 + `baseline` 运行，经 `serve` 评测 |
| 探针 R² 表（逐动作维） | `probe` → `probe/probe.json` |
| 自运动压力（理论证伪） | `label --ego_aug` + `probe --stress`，LAM 在有/无 `--inv_coef 0` 下训练 |
| 连续 vs 离散潜动作 | `train-lam --use_vq` + 重标 + 重探针 + 重预训练 |
| 收敛速度 | `finetune_*` 运行的 wandb 曲线（达到固定成功率所需步数） |
| 适配器消融 | `finetune --train_adapter` vs 默认冻结 |
