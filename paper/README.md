# AAAI 2026 论文草稿（SimVLA Transition-Aware Training）

## 文件

| 文件 | 说明 |
|---|---|
| `simvla_aaai2026.tex` | 论文主文件（单文件，符合 AAAI 单 tex 要求） |
| `references.bib` | 参考文献（真实可查的 VLA/数据加权文献） |
| `figures/placeholder.png` | **需要你放入**：所有图暂时共用这一张占位图 |

## 编译

需要 AAAI-26 author kit 里的 `aaai2026.sty` 和 `aaai2026.bst`（官网下载，
放在本目录），然后：

```bash
pdflatex simvla_aaai2026
bibtex   simvla_aaai2026
pdflatex simvla_aaai2026
pdflatex simvla_aaai2026
```

## 投稿前要填的占位（全文搜 `TODO` 和 `\na{}`）

1. **图**（4 处，全部指向 `figures/placeholder.png`）：
   - Fig 1 teaser：一个信号 → 三个时间尺度
   - Fig 2 架构图（双栏 `figure*`）
   - Fig 3 实验 0a 散点图——直接用 `transition_density_stats.py` 生成的
     `density_vs_sr.png`
   - Fig 4 边界分数定性可视化
2. **表**（5 张，`--` 为待填数字，mean±std）：
   - Table 1 主表：published reference 行已填（OpenVLA 论文报告值，带 †）；
     ours 5 行待填
   - Table 2 实验 0a/0b 相关性（跑完假设检验直接抄终端输出）
   - Table 3 TDS 消融（反转/无 floor/课程式/loss-based 对照）
   - Table 4 步级/块级消融（High-Δ MSE、Jerk 诊断列）
   - Table 5 跨尺度协同（边界 AUROC、BGE 增益 × TDS 开关）
3. **BGE 小节**：按最终推理实现核对描述（tex 内有 TODO 注释标记）。
4. 实验 setup 里的 trials/seeds/模型参数量/耗时。

## 风格说明

结构与 OpenVLA / Octo / π0 等顶会 VLA 论文对齐：
Intro（贡献列表+teaser 图）→ Related Work（3 小节）→ Method（信号定义 + 三组件
+ 算法框）→ Experiments（Q1–Q5 研究问题组织，主表+消融+协同）→ Limitations →
Conclusion。主表沿用 LIBERO 惯例（行=方法，列=Spatial/Object/Goal/Long/Avg）。
